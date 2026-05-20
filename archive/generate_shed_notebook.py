import nbformat as nbf
import nbformat.v4 as nbfv4
import os

os.makedirs('notebooks', exist_ok=True)
os.makedirs('figures', exist_ok=True)

nb = nbfv4.new_notebook()

# Add cells
cells = []
cells.append(nbfv4.new_markdown_cell("# SHED 2024 BNPL Econometric Analysis"))

cells.append(nbfv4.new_code_cell("""
import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading data...")
z = zipfile.ZipFile('../data/raw/public2024.csv.zip')
df = pd.read_csv(z.open('public2024.csv'), low_memory=False)

# Map target variable
df['bnpl_use'] = df['BNPL1'].apply(lambda x: 1 if x == 'Yes' else 0)

# Map demographics
df['age_cat'] = df['ppagecat']
df['income_cat'] = df['inc_4cat_50k']
df['race_cat'] = df['race_5cat']
df['educ_cat'] = df['educ_4cat']

# Clean string values to categories if needed
demographics = ['age_cat', 'income_cat', 'race_cat', 'educ_cat']
df = df.dropna(subset=demographics)
"""))

cells.append(nbfv4.new_markdown_cell("## Analysis 1: Weighted Descriptive Comparison + Chi-Square"))
cells.append(nbfv4.new_code_cell("""
def weighted_prop(df, group_col, target_col, weight_col='weight_pop'):
    grouped = df.groupby(group_col)
    res = {}
    for name, group in grouped:
        weighted_mean = np.average(group[target_col], weights=group[weight_col].fillna(0))
        res[name] = weighted_mean
    return pd.Series(res)

print("--- Weighted BNPL Usage by Demographic ---")
for demo in demographics:
    print(f"\\nBy {demo}:")
    props = weighted_prop(df, demo, 'bnpl_use')
    print(props)
    
    # Chi-square test (unweighted for simplicity of stat test, or pseudo-weighted)
    # Using cross tab for unweighted chi2
    crosstab = pd.crosstab(df[demo], df['bnpl_use'])
    chi2, p, dof, ex = chi2_contingency(crosstab)
    print(f"Chi-square p-value for {demo}: {p:.4e}")
    
    # Plot
    plt.figure(figsize=(8,4))
    sns.barplot(x=props.index, y=props.values, palette='viridis')
    plt.title(f'Weighted BNPL Usage by {demo}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'../figures/bnpl_by_{demo}.png')
    plt.close()
"""))

cells.append(nbfv4.new_markdown_cell("## Analysis 2: Composite Vulnerability Index (PCA + Cronbach's Alpha)"))
cells.append(nbfv4.new_code_cell("""
# Map fragility variables
# EF1: Can handle $400 emergency (Yes -> 0, No -> 1)
df['fragile_EF1'] = df['EF1'].apply(lambda x: 1 if x == 'No' else 0)

# atleast_okay: Doing okay financially (Yes -> 0, No -> 1)
df['fragile_okay'] = df['atleast_okay'].apply(lambda x: 1 if x == 'No' else 0)

# E2: Carried credit card balance (Yes -> 1, No -> 0)
df['fragile_E2'] = df['E2'].apply(lambda x: 1 if x == 'Yes' else 0)

# K0: Overdrawn checking (Yes -> 1, No -> 0)
df['fragile_K0'] = df['K0'].apply(lambda x: 1 if x == 'Yes' else 0)

fragility_cols = ['fragile_EF1', 'fragile_okay', 'fragile_E2', 'fragile_K0']

# Calculate Cronbach's Alpha manually
def cronbach_alpha(df):
    k = df.shape[1]
    var_sum = df.var(ddof=1).sum()
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0: return 0
    return (k / (k - 1)) * (1 - (var_sum / total_var))

alpha = cronbach_alpha(df[fragility_cols])
print(f"Cronbach's Alpha for fragility items: {alpha:.3f}")

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[fragility_cols])

pca = PCA(n_components=1)
df['vulnerability_score'] = pca.fit_transform(X_scaled)

print(f"PCA explained variance ratio (PC1): {pca.explained_variance_ratio_[0]:.3f}")

plt.figure(figsize=(8,4))
sns.histplot(df['vulnerability_score'], bins=30, kde=True, color='purple')
plt.title('Distribution of Vulnerability Score (PC1)')
plt.savefig('../figures/vulnerability_score_dist.png')
plt.close()
"""))

cells.append(nbfv4.new_markdown_cell("## Analysis 3: OLS Linear Probability Model with Interaction Terms"))
cells.append(nbfv4.new_code_cell("""
# Convert demographics to categorical codes for regression
for col in demographics:
    df[col] = df[col].astype('category')

# OLS model
# We use a Linear Probability Model: bnpl_use ~ vulnerability_score * age_cat + income_cat + race_cat + educ_cat
# Drop NA for regression
df_reg = df.dropna(subset=['bnpl_use', 'vulnerability_score'] + demographics)

formula = 'bnpl_use ~ vulnerability_score * age_cat + income_cat + race_cat + educ_cat'
model = smf.ols(formula=formula, data=df_reg).fit()
print(model.summary().tables[1])

# Plot the interaction directly instead of using lmplot on the whole DF to avoid seaborn bugs
plt.figure(figsize=(10,6))
sns.lmplot(x='vulnerability_score', y='bnpl_use', hue='age_cat', data=df_reg.sample(5000, random_state=42), scatter=False, aspect=1.5)
plt.title('Predicted BNPL Use vs Vulnerability by Age')
plt.tight_layout()
plt.savefig('../figures/ols_interaction_age.png')
plt.close('all')
"""))

cells.append(nbfv4.new_markdown_cell("## Analysis 4: Propensity Score Matching"))
cells.append(nbfv4.new_code_cell("""
# Define 'Fragile' vs 'Non-Fragile' (Median split of vulnerability score)
median_vuln = df_reg['vulnerability_score'].median()
df_reg['is_fragile'] = (df_reg['vulnerability_score'] > median_vuln).astype(int)

# Logistic regression for propensity score based on demographics
formula_psm = 'is_fragile ~ age_cat + income_cat + race_cat + educ_cat'
logit_model = smf.logit(formula=formula_psm, data=df_reg).fit(disp=0)
df_reg['propensity_score'] = logit_model.predict(df_reg)

# Nearest Neighbor Matching (Simplified 1:1)
fragile_df = df_reg[df_reg['is_fragile'] == 1].reset_index(drop=True)
control_df = df_reg[df_reg['is_fragile'] == 0].reset_index(drop=True)

# For speed on large dataset, sample if needed, but NN with sklearn is fast
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(control_df[['propensity_score']])

distances, indices = nn.kneighbors(fragile_df[['propensity_score']])

matched_control_df = control_df.iloc[indices.flatten()].reset_index(drop=True)

print("--- Propensity Score Matching Results ---")
print(f"Fragile BNPL Use Rate (Unmatched): {fragile_df['bnpl_use'].mean():.4f}")
print(f"Non-Fragile BNPL Use Rate (Unmatched): {control_df['bnpl_use'].mean():.4f}")
print("---")
print(f"Fragile BNPL Use Rate (Matched): {fragile_df['bnpl_use'].mean():.4f}")
print(f"Non-Fragile BNPL Use Rate (Matched Controls): {matched_control_df['bnpl_use'].mean():.4f}")

# T-test for difference in matched groups
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(fragile_df['bnpl_use'], matched_control_df['bnpl_use'])
print(f"T-test p-value (Matched groups): {p_val:.4e}")

# Visualization of matched outcome
plt.figure(figsize=(6,5))
sns.barplot(x=['Fragile (Matched)', 'Non-Fragile (Matched)'], 
            y=[fragile_df['bnpl_use'].mean(), matched_control_df['bnpl_use'].mean()],
            palette='Set2')
plt.title('BNPL Usage Rate: Matched Fragile vs Non-Fragile')
plt.ylabel('Proportion Using BNPL')
plt.savefig('../figures/psm_results.png')
plt.close()
"""))

nb.cells = cells
nbf.write(nb, 'notebooks/05_shed_econometric_analysis.ipynb')
print("Notebook generated successfully.")
