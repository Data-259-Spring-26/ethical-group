import os
import pandas as pd
import numpy as np
import random
from dotenv import load_dotenv

load_dotenv()

META_API_TOKEN = os.getenv("META_API_TOKEN")

BNPL_PROVIDERS = ['Klarna', 'Affirm', 'Afterpay', 'Zip', 'Sezzle', 'PayPal Pay Later']

def generate_mock_meta_data(num_records=5000):
    print("Generating mock Meta Ad Library data because API token is missing or for demonstration...")
    
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    age_brackets = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    # Let's bias targeting to 18-24 and 25-34 to simulate the ethical concern
    age_probs = [0.40, 0.35, 0.10, 0.08, 0.04, 0.03]
    
    headlines = [
        "Buy Now, Pay Later with no hidden fees!",
        "Get it now, pay over time.",
        "Split your purchase into 4 easy payments.",
        "Don't wait! Buy today and pay later.",
        "Unlock your purchasing power.",
        "Zero interest when you pay in 4."
    ]
    
    body_texts = [
        "Why wait until payday? Get the things you want now and split your payments. It's fast, easy, and risk-free.",
        "Shop your favorite brands and pay over 6 weeks. No interest, no hard credit check.",
        "Manage your cash flow better. Buy the essentials today and pay in small, manageable installments.",
        "Instant approval! Just select our payment option at checkout.",
        "We help you buy what you want, when you want it. Download the app today.",
        "Don't let tight budgets hold you back. Flexible payment options available now!"
    ]
    
    for i in range(num_records):
        provider = np.random.choice(BNPL_PROVIDERS)
        target_age = np.random.choice(age_brackets, p=age_probs)
        impressions = int(np.random.lognormal(8, 1))
        spend = float(np.random.uniform(50, 5000))
        headline = np.random.choice(headlines)
        body = np.random.choice(body_texts)
        
        # Add some risk vs aspirational text bias based on age targeting
        if target_age in ['18-24', '25-34']:
            if random.random() < 0.7:
                body += " No credit impact! Treat yourself now. Instant decision."
        else:
            if random.random() < 0.5:
                body += " Smart budgeting for your family. No interest fees."
        
        data.append({
            'ad_id': f'ad_{i}',
            'provider': provider,
            'target_age': target_age,
            'impressions': impressions,
            'spend': round(spend, 2),
            'headline': headline,
            'body': body
        })
        
    df = pd.DataFrame(data)
    df.to_csv('data/raw/meta_ads.csv', index=False)
    print("Saved mock data to data/raw/meta_ads.csv")

if __name__ == "__main__":
    if META_API_TOKEN:
        print("API Token found. (In a real scenario, API requests to Facebook Graph API would happen here)")
        # Falling back to mock for safety in demonstration
        generate_mock_meta_data()
    else:
        print("No Meta API token found. Using mock data generator.")
        generate_mock_meta_data()
