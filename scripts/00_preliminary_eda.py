import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Loading data...")
    # Load dataset
    df = pd.read_csv('data/raw/public2024.csv', low_memory=False)

    print("Processing BNPL indicators...")
    # Assume BNPL1 is the primary indicator of using BNPL. 
    # Valid answers might be 'Yes', 'No', or something similar.
    # We will map 'Yes' to 1 and 'No' to 0 for calculation.
    df['used_bnpl'] = df['BNPL1'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # Set up plotting style
    sns.set_theme(style="whitegrid")
    
    # 1. BNPL Usage by Age Group (ppagecat)
    if 'ppagecat' in df.columns:
        print("Plotting BNPL usage by age group...")
        plt.figure(figsize=(10, 6))
        # Group and calculate mean
        age_usage = df.groupby('ppagecat')['used_bnpl'].mean().reset_index()
        # Sort by age category logically
        age_order = sorted(age_usage['ppagecat'].unique())
        
        sns.barplot(data=age_usage, x='ppagecat', y='used_bnpl', order=age_order, palette='viridis')
        plt.title('BNPL Usage Rate by Age Group (2024 SHED)', fontsize=14)
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Proportion Using BNPL', fontsize=12)
        plt.ylim(0, max(age_usage['used_bnpl']) * 1.2)
        
        # Add value labels
        for index, row in age_usage.iterrows():
            plt.text(age_order.index(row['ppagecat']), row['used_bnpl'] + 0.005, 
                     f"{row['used_bnpl']:.1%}", color='black', ha="center")
            
        plt.tight_layout()
        plt.savefig('figures/bnpl_usage_by_age.png')
        plt.close()

    # 2. BNPL Usage by Income Bracket (inc_4cat_50k or ppinc7)
    income_col = 'inc_4cat_50k' if 'inc_4cat_50k' in df.columns else 'ppinc7'
    if income_col in df.columns:
        print(f"Plotting BNPL usage by income bracket ({income_col})...")
        plt.figure(figsize=(10, 6))
        income_usage = df.groupby(income_col)['used_bnpl'].mean().reset_index()
        
        # Determine appropriate order for income if possible
        income_order = sorted(income_usage[income_col].unique())
        
        sns.barplot(data=income_usage, x=income_col, y='used_bnpl', order=income_order, palette='mako')
        plt.title('BNPL Usage Rate by Income Bracket (2024 SHED)', fontsize=14)
        plt.xlabel('Income Bracket', fontsize=12)
        plt.ylabel('Proportion Using BNPL', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(income_usage['used_bnpl']) * 1.2)
        
        # Add value labels
        for index, row in income_usage.iterrows():
            plt.text(income_order.index(row[income_col]), row['used_bnpl'] + 0.005, 
                     f"{row['used_bnpl']:.1%}", color='black', ha="center")
            
        plt.tight_layout()
        plt.savefig('figures/bnpl_usage_by_income.png')
        plt.close()
        
    print("Figures generated successfully in the 'figures/' directory.")

if __name__ == "__main__":
    main()
