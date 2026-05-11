import nbformat as nbf
import os
import nbformat.v4 as nbfv4

os.makedirs('notebooks', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# 01_data_collection.ipynb
nb1 = nbfv4.new_notebook()
nb1.cells = [
    nbfv4.new_markdown_cell("# 01 - Data Collection & Preprocessing"),
    nbfv4.new_code_cell("""import pandas as pd
import numpy as np

# Load Mock Data
meta_df = pd.read_csv('../data/raw/meta_ads.csv')
census_df = pd.read_csv('../data/acs/census_fragility.csv')

print("Meta Ad Library Data Shape:", meta_df.shape)
print("Census Data Shape:", census_df.shape)

# Save processed (in this mock they are already somewhat clean)
meta_df.to_csv('../data/processed/meta_ads_clean.csv', index=False)
census_df.to_csv('../data/processed/census_fragility_clean.csv', index=False)
""")
]
nbf.write(nb1, 'notebooks/01_data_collection.ipynb')

# 02_targeting_analysis.ipynb
nb2 = nbfv4.new_notebook()
nb2.cells = [
    nbfv4.new_markdown_cell("# 02 - Targeting Analysis"),
    nbfv4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare

meta_df = pd.read_csv('../data/processed/meta_ads_clean.csv')

# Plot Age Distribution overall
plt.figure(figsize=(10,6))
sns.countplot(data=meta_df, x='target_age', order=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], palette='viridis')
plt.title('Distribution of Ad Targeting by Age Bracket')
plt.savefig('../figures/age_targeting_distribution.png')
plt.show()
""")
]
nbf.write(nb2, 'notebooks/02_targeting_analysis.ipynb')

# 03_spend_by_demographic.ipynb
nb3 = nbfv4.new_notebook()
nb3.cells = [
    nbfv4.new_markdown_cell("# 03 - Spend by Demographic"),
    nbfv4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

meta_df = pd.read_csv('../data/processed/meta_ads_clean.csv')

spend_df = meta_df.groupby('target_age')['spend'].sum().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=spend_df, x='target_age', y='spend', palette='magma')
plt.title('Total Ad Spend by Target Age Bracket')
plt.ylabel('Total Spend ($)')
plt.savefig('../figures/spend_by_demographic.png')
plt.show()
""")
]
nbf.write(nb3, 'notebooks/03_spend_by_demographic.ipynb')

# 04_content_nlp_analysis.ipynb
nb4 = nbfv4.new_notebook()
nb4.cells = [
    nbfv4.new_markdown_cell("# 04 - Content NLP Analysis"),
    nbfv4.new_code_cell("""import sys
sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.nlp_analysis import analyze_text

meta_df = pd.read_csv('../data/processed/meta_ads_clean.csv')

nlp_df = analyze_text(meta_df, 'body')

# Plot Sentiment by Age
plt.figure(figsize=(10,6))
sns.boxplot(data=nlp_df, x='target_age', y='vader_compound', order=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
plt.title('Sentiment (VADER Compound) by Target Age Bracket')
plt.savefig('../figures/sentiment_by_age.png')
plt.show()

# Readability by Age
plt.figure(figsize=(10,6))
sns.barplot(data=nlp_df, x='target_age', y='fk_grade', errorbar=None, palette='coolwarm')
plt.title('Average Flesch-Kincaid Grade Level by Age Bracket')
plt.savefig('../figures/readability_by_age.png')
plt.show()

# Risk Keywords
risk_summary = nlp_df.groupby('target_age')['risk_keyword_count'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.barplot(data=risk_summary, x='target_age', y='risk_keyword_count', palette='Reds')
plt.title('Average Risk Keywords per Ad by Age')
plt.savefig('../figures/risk_keywords_by_age.png')
plt.show()
""")
]
nbf.write(nb4, 'notebooks/04_content_nlp_analysis.ipynb')

print("Notebooks created successfully.")
