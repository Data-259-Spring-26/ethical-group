import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

def generate_mock_census_data():
    print("Generating mock Census ACS data...")
    # Age brackets corresponding to meta ad library
    age_brackets = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    # Financial fragility scores (higher means more fragile)
    # Younger brackets generally have higher fragility in this mock scenario
    fragility_means = [8.5, 7.2, 6.0, 5.5, 4.0, 3.5]
    
    data = []
    np.random.seed(42)
    for i, age in enumerate(age_brackets):
        # We assume multiple regions/PUMAs, let's just make 100 per age group
        for j in range(100):
            fragility = max(0, min(10, np.random.normal(fragility_means[i], 1.5)))
            debt_to_income = max(0, np.random.normal(0.4 - (i*0.05), 0.1))
            data.append({
                'region_id': j,
                'age_bracket': age,
                'financial_fragility_index': round(fragility, 2),
                'debt_to_income_ratio': round(debt_to_income, 2)
            })
            
    df = pd.DataFrame(data)
    df.to_csv('data/acs/census_fragility.csv', index=False)
    print("Saved mock census data to data/acs/census_fragility.csv")

if __name__ == "__main__":
    if CENSUS_API_KEY:
        print("Census API Key found. (In a real scenario, API requests to Census API would happen here)")
    generate_mock_census_data()
