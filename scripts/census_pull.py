import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

def pull_census_data():
    print("Pulling real Census ACS data...")
    YEAR = "2023"  # Using 2023 as 2024 ACS 5-year is usually not released until Dec 2025
    DATASET = "acs/acs5"
    BASE_URL = f"https://api.census.gov/data/{YEAR}/{DATASET}"
    
    # We must split the requests because the Census API allows a maximum of 50 variables per call.
    
    # -------------------------------
    # REQUEST 1: AGE (35 variables)
    # -------------------------------
    variables_age = {
        "B01001_001E": "total_population",
        
        # MALE AGE BRACKETS
        "B01001_007E": "m_18_19",
        "B01001_008E": "m_20_24",
        "B01001_009E": "m_25_29",
        "B01001_010E": "m_30_34",
        "B01001_011E": "m_35_39",
        "B01001_012E": "m_40_44",
        "B01001_013E": "m_45_49",
        "B01001_014E": "m_50_54",
        "B01001_015E": "m_55_59",
        "B01001_016E": "m_60_61",
        "B01001_017E": "m_62_64",
        "B01001_018E": "m_65_66",
        "B01001_019E": "m_67_69",
        "B01001_020E": "m_70_74",
        "B01001_021E": "m_75_79",
        "B01001_022E": "m_80_84",
        "B01001_023E": "m_85_plus",
        
        # FEMALE AGE BRACKETS
        "B01001_031E": "f_18_19",
        "B01001_032E": "f_20_24",
        "B01001_033E": "f_25_29",
        "B01001_034E": "f_30_34",
        "B01001_035E": "f_35_39",
        "B01001_036E": "f_40_44",
        "B01001_037E": "f_45_49",
        "B01001_038E": "f_50_54",
        "B01001_039E": "f_55_59",
        "B01001_040E": "f_60_61",
        "B01001_041E": "f_62_64",
        "B01001_042E": "f_65_66",
        "B01001_043E": "f_67_69",
        "B01001_044E": "f_70_74",
        "B01001_045E": "f_75_79",
        "B01001_046E": "f_80_84",
        "B01001_047E": "f_85_plus",
    }
    
    var_string_age = ",".join(variables_age.keys())
    params_age = {"get": var_string_age, "for": "us:1"}
    if CENSUS_API_KEY:
        params_age["key"] = CENSUS_API_KEY
        
    res_age = requests.get(BASE_URL, params=params_age)
    if res_age.status_code != 200:
        raise Exception(f"Age API request failed: {res_age.text}")
    data_age = res_age.json()
    df_age = pd.DataFrame([data_age[1]], columns=data_age[0])
    df_age.rename(columns=variables_age, inplace=True)
    
    # -------------------------------
    # REQUEST 2: INCOME & POVERTY (20 variables)
    # -------------------------------
    variables_econ = {
        # INCOME DISTRIBUTION
        "B19001_001E": "total_households",
        "B19001_002E": "income_less_10k",
        "B19001_003E": "income_10k_15k",
        "B19001_004E": "income_15k_20k",
        "B19001_005E": "income_20k_25k",
        "B19001_006E": "income_25k_30k",
        "B19001_007E": "income_30k_35k",
        "B19001_008E": "income_35k_40k",
        "B19001_009E": "income_40k_45k",
        "B19001_010E": "income_45k_50k",
        "B19001_011E": "income_50k_60k",
        "B19001_012E": "income_60k_75k",
        "B19001_013E": "income_75k_100k",
        "B19001_014E": "income_100k_125k",
        "B19001_015E": "income_125k_150k",
        "B19001_016E": "income_150k_200k",
        "B19001_017E": "income_200k_plus",

        # POVERTY / FRAGILITY
        "B17001_001E": "poverty_total",
        "B17001_002E": "below_poverty",
        "B17001_031E": "above_poverty",
    }
    
    var_string_econ = ",".join(variables_econ.keys())
    params_econ = {"get": var_string_econ, "for": "us:1"}
    if CENSUS_API_KEY:
        params_econ["key"] = CENSUS_API_KEY
        
    res_econ = requests.get(BASE_URL, params=params_econ)
    if res_econ.status_code != 200:
        raise Exception(f"Econ API request failed: {res_econ.text}")
    data_econ = res_econ.json()
    df_econ = pd.DataFrame([data_econ[1]], columns=data_econ[0])
    df_econ.rename(columns=variables_econ, inplace=True)
    
    # -------------------------------
    # Combine and Process
    # -------------------------------
    df = pd.merge(df_age, df_econ, on="us")

    # Convert numeric columns
    for col in df.columns:
        if col != "us":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Calculate custom metrics based on buckets
    df["age_18_24"] = df["m_18_19"] + df["m_20_24"] + df["f_18_19"] + df["f_20_24"]
    df["age_25_34"] = df["m_25_29"] + df["m_30_34"] + df["f_25_29"] + df["f_30_34"]
    df["age_35_44"] = df["m_35_39"] + df["m_40_44"] + df["f_35_39"] + df["f_40_44"]
    df["age_45_54"] = df["m_45_49"] + df["m_50_54"] + df["f_45_49"] + df["f_50_54"]
    df["age_55_64"] = df["m_55_59"] + df["m_60_61"] + df["m_62_64"] + df["f_55_59"] + df["f_60_61"] + df["f_62_64"]
    df["age_65_plus"] = df["m_65_66"] + df["m_67_69"] + df["m_70_74"] + df["m_75_79"] + df["m_80_84"] + df["m_85_plus"] + df["f_65_66"] + df["f_67_69"] + df["f_70_74"] + df["f_75_79"] + df["f_80_84"] + df["f_85_plus"]
    
    df["poverty_rate"] = df["below_poverty"] / df["poverty_total"]
    
    df["low_income_under_50k"] = (
        df[
            [
                "income_less_10k",
                "income_10k_15k",
                "income_15k_20k",
                "income_20k_25k",
                "income_25k_30k",
                "income_30k_35k",
                "income_35k_40k",
                "income_40k_45k",
                "income_45k_50k"
            ]
        ].sum(axis=1)
    )

    df["pct_households_under_50k"] = df["low_income_under_50k"] / df["total_households"]
    
    # Save the dataframe
    df.to_csv('data/acs/census_fragility.csv', index=False)
    print("Saved real census data to data/acs/census_fragility.csv")
    
if __name__ == "__main__":
    pull_census_data()
