import os
import json
import nbformat as nbf
import nbformat.v4 as nbfv4
import shutil

def split_1_2():
    nb = nbf.read('[In Progress]Analysis/BNPL_Analysis_1_and_2.ipynb', as_version=4)
    
    # Analysis 1 is cells 0 to 18. We should copy imports (cell 2, 3) to Analysis 2 as well.
    cells_a1 = nb.cells[:19]
    # Update title
    cells_a1[0]['source'] = ['# BNPL Project — Analysis 1\n', '## Weighted Descriptive Comparison\n']
    
    cells_a2 = [
        nb.cells[0].copy(), # Title block
        nb.cells[2].copy(), # Imports
        nb.cells[3].copy(), # Load data
        nb.cells[7].copy(), # Build dataframe
        nb.cells[8].copy()
    ]
    
    # Replace title
    cells_a2[0]['source'] = ['# BNPL Project — Analysis 2\n', '## Composite Vulnerability Index\n']
    
    cells_a2.extend(nb.cells[19:])
    
    nb1 = nbfv4.new_notebook()
    nb1.cells = cells_a1
    
    nb2 = nbfv4.new_notebook()
    nb2.cells = cells_a2
    
    nbf.write(nb1, 'notebooks/05a_shed_analysis_1_descriptive.ipynb')
    nbf.write(nb2, 'notebooks/05b_shed_analysis_2_vulnerability_index.ipynb')

def create_a3():
    nb = nbfv4.new_notebook()
    cells = []
    cells.append(nbfv4.new_markdown_cell("# BNPL Project — Analysis 3\n## OLS Linear Probability Model (Rigorous WLS + HC3)"))
    cells.append(nbfv4.new_code_cell("""import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
z = zipfile.ZipFile('../data/raw/public2024.csv.zip')
df = pd.read_csv(z.open('public2024.csv'), low_memory=False)

# Recreate vulnerability score for this notebook (or load if we saved it, but we'll recreate for standalone execution)
df['bnpl_use'] = df['BNPL1'].apply(lambda x: 1 if x == 'Yes' else 0)

df['age_cat'] = df['ppagecat']
df['income_cat'] = df['inc_4cat_50k']
df['race_cat'] = df['race_5cat']
df['educ_cat'] = df['educ_4cat']

# Re-run a simplified PCA here to get vulnerability_score since we didn't save it to a file
df['fragile_EF1'] = df['EF1'].apply(lambda x: 1 if x == 'No' else 0)
df['fragile_okay'] = df['atleast_okay'].apply(lambda x: 1 if x == 'No' else 0)
df['fragile_E2'] = df['E2'].apply(lambda x: 1 if x == 'Yes' else 0)
df['fragile_K0'] = df['K0'].apply(lambda x: 1 if x == 'Yes' else 0)

fragility_cols = ['fragile_EF1', 'fragile_okay', 'fragile_E2', 'fragile_K0']
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sub = df.dropna(subset=fragility_cols + ['weight'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sub[fragility_cols])
pca = PCA(n_components=1)
sub['vulnerability_score'] = pca.fit_transform(X_scaled)

# Ensure positive means more fragile (if EF1 etc are positive, check correlation)
if sub[['vulnerability_score', 'fragile_EF1']].corr().iloc[0,1] < 0:
    sub['vulnerability_score'] *= -1

df = df.merge(sub[['vulnerability_score']], left_index=True, right_index=True, how='left')
"""))
    cells.append(nbfv4.new_markdown_cell("### Weighted Least Squares (WLS) with HC3 Robust Standard Errors"))
    cells.append(nbfv4.new_code_cell("""demographics = ['age_cat', 'income_cat', 'race_cat', 'educ_cat']
for col in demographics:
    df[col] = df[col].astype('category')

# Drop NA for regression
df_reg = df.dropna(subset=['bnpl_use', 'vulnerability_score', 'weight'] + demographics)

# Fit WLS model using survey weights
formula = 'bnpl_use ~ vulnerability_score * age_cat + income_cat + race_cat + educ_cat'

# We use smf.wls to account for the survey weights and cov_type='HC3' to correct for heteroskedasticity 
# inherent in the Linear Probability Model.
model = smf.wls(formula=formula, data=df_reg, weights=df_reg['weight']).fit(cov_type='HC3')
print(model.summary().tables[1])

plt.figure(figsize=(10,6))
sns.lmplot(x='vulnerability_score', y='bnpl_use', hue='age_cat', data=df_reg.sample(5000, random_state=42), scatter=False, aspect=1.5)
plt.title('Predicted BNPL Use vs Vulnerability by Age (WLS Model)')
plt.tight_layout()
plt.savefig('../figures/ols_interaction_age_wls.png')
plt.show()
"""))
    nb.cells = cells
    nbf.write(nb, 'notebooks/05c_shed_analysis_3_ols_model.ipynb')

def create_a4():
    nb = nbfv4.new_notebook()
    cells = []
    cells.append(nbfv4.new_markdown_cell("# BNPL Project — Analysis 4\n## Propensity Score Matching (Caliper + Covariate Balance)"))
    cells.append(nbfv4.new_code_cell("""import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
z = zipfile.ZipFile('../data/raw/public2024.csv.zip')
df = pd.read_csv(z.open('public2024.csv'), low_memory=False)

df['bnpl_use'] = df['BNPL1'].apply(lambda x: 1 if x == 'Yes' else 0)
df['age_cat'] = df['ppagecat']
df['income_cat'] = df['inc_4cat_50k']
df['race_cat'] = df['race_5cat']
df['educ_cat'] = df['educ_4cat']

# Re-run a simplified PCA here to get vulnerability_score
df['fragile_EF1'] = df['EF1'].apply(lambda x: 1 if x == 'No' else 0)
df['fragile_okay'] = df['atleast_okay'].apply(lambda x: 1 if x == 'No' else 0)
df['fragile_E2'] = df['E2'].apply(lambda x: 1 if x == 'Yes' else 0)
df['fragile_K0'] = df['K0'].apply(lambda x: 1 if x == 'Yes' else 0)

fragility_cols = ['fragile_EF1', 'fragile_okay', 'fragile_E2', 'fragile_K0']
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sub = df.dropna(subset=fragility_cols + ['weight'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sub[fragility_cols])
pca = PCA(n_components=1)
sub['vulnerability_score'] = pca.fit_transform(X_scaled)
if sub[['vulnerability_score', 'fragile_EF1']].corr().iloc[0,1] < 0:
    sub['vulnerability_score'] *= -1

df = df.merge(sub[['vulnerability_score']], left_index=True, right_index=True, how='left')
demographics = ['age_cat', 'income_cat', 'race_cat', 'educ_cat']
df_reg = df.dropna(subset=['bnpl_use', 'vulnerability_score', 'weight'] + demographics).copy()
"""))
    cells.append(nbfv4.new_markdown_cell("### Propensity Score Calculation & Caliper Matching"))
    cells.append(nbfv4.new_code_cell("""# Median split to define 'Fragile' vs 'Non-Fragile'
median_vuln = df_reg['vulnerability_score'].median()
df_reg['is_fragile'] = (df_reg['vulnerability_score'] > median_vuln).astype(int)

# Logistic regression for propensity score based on demographics, adjusting for survey weights
formula_psm = 'is_fragile ~ age_cat + income_cat + race_cat + educ_cat'
# Generalized Linear Model with Binomial family and survey weights
glm_binom = sm.families.Binomial()
logit_model = smf.glm(formula=formula_psm, data=df_reg, family=glm_binom, freq_weights=df_reg['weight']).fit()
df_reg['propensity_score'] = logit_model.predict(df_reg)
df_reg['logit_ps'] = np.log(df_reg['propensity_score'] / (1 - df_reg['propensity_score']))

fragile_df = df_reg[df_reg['is_fragile'] == 1].reset_index(drop=True)
control_df = df_reg[df_reg['is_fragile'] == 0].reset_index(drop=True)

# Caliper Matching (0.2 Standard Deviations of logit propensity score)
caliper = 0.2 * df_reg['logit_ps'].std()
print(f"Using caliper: {caliper:.4f} logit units")

nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(control_df[['logit_ps']])
distances, indices = nn.kneighbors(fragile_df[['logit_ps']])

# Filter by caliper
valid_matches = distances.flatten() <= caliper
matched_fragile = fragile_df[valid_matches].reset_index(drop=True)
matched_control = control_df.iloc[indices.flatten()[valid_matches]].reset_index(drop=True)

print(f"Matched {len(matched_fragile)} out of {len(fragile_df)} fragile individuals ({(len(matched_fragile)/len(fragile_df)):.1%})")
"""))
    cells.append(nbfv4.new_markdown_cell("### Covariate Balance (Standardized Mean Differences)"))
    cells.append(nbfv4.new_code_cell("""# Calculate SMD for age_cat[T.25-34] as an example to check balance
def get_dummies(df, cols):
    return pd.get_dummies(df[cols], drop_first=True)

X_fragile = get_dummies(fragile_df, demographics)
X_control = get_dummies(control_df, demographics)
X_matched_fragile = get_dummies(matched_fragile, demographics)
X_matched_control = get_dummies(matched_control, demographics)

def calc_smd(df1, df2):
    mean_diff = df1.mean() - df2.mean()
    pooled_sd = np.sqrt((df1.var() + df2.var()) / 2)
    return (mean_diff / pooled_sd).abs()

smd_unmatched = calc_smd(X_fragile, X_control)
smd_matched = calc_smd(X_matched_fragile, X_matched_control)

smd_df = pd.DataFrame({
    'Unmatched': smd_unmatched,
    'Matched': smd_matched
})

print("\\nStandardized Mean Differences (SMD):")
print(smd_df.head(10))

# Plot SMD
smd_df.plot(kind='barh', figsize=(8, 6))
plt.axvline(x=0.1, color='r', linestyle='--', label='0.1 Threshold (Balanced)')
plt.title('Covariate Balance Before and After Matching')
plt.xlabel('Standardized Mean Difference (SMD)')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/psm_covariate_balance.png')
plt.show()

# Final T-test
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(matched_fragile['bnpl_use'], matched_control['bnpl_use'])
print(f"\\n--- Propensity Score Matching Results (Caliper Matching) ---")
print(f"Fragile BNPL Use Rate (Matched): {matched_fragile['bnpl_use'].mean():.4f}")
print(f"Non-Fragile BNPL Use Rate (Matched Controls): {matched_control['bnpl_use'].mean():.4f}")
print(f"Paired T-test p-value (Matched groups): {p_val:.4e}")
"""))
    nb.cells = cells
    nbf.write(nb, 'notebooks/05d_shed_analysis_4_psm.ipynb')

if __name__ == '__main__':
    split_1_2()
    create_a3()
    create_a4()
    
    # Moves and deletes
    os.rename('generate_notebooks.py', 'archive/generate_notebooks.py')
    os.rename('generate_shed_notebook.py', 'archive/generate_shed_notebook.py')
    
    if os.path.exists('test_fk.py'):
        os.remove('test_fk.py')
        
    if os.path.exists('notebooks/05_shed_econometric_analysis.ipynb'):
        os.remove('notebooks/05_shed_econometric_analysis.ipynb')
        
    if os.path.exists('[In Progress]Analysis'):
        shutil.rmtree('[In Progress]Analysis')

    print("Refactoring completed successfully.")
