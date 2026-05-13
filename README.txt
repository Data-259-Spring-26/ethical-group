BNPL Social Media Targeting: An Ethical Analysis
DATA 259 -- Autumn 2025

Group Members: Max Linton, Alvin He, Charlie

================================================================================
RESEARCH QUESTION
================================================================================

Do BNPL providers disproportionately target financially vulnerable audiences in
their social media advertising -- and if so, does the content of those ads
obscure or downplay the risks of borrowing?

================================================================================
OVERVIEW
================================================================================

This project investigates the advertising practices of major Buy Now, Pay Later
(BNPL) providers on Meta platforms (Facebook/Instagram). Using the Meta Ad
Library API and U.S. Census data, we analyze both the targeting of BNPL ads
(who sees them) and the content of those ads (what they say) to evaluate whether
these practices constitute predatory design under a data ethics lens.

BNPL providers examined: Klarna, Affirm, Afterpay, Zip, Sezzle, PayPal Pay Later

================================================================================
REPOSITORY STRUCTURE
================================================================================

.
├── data/
│   ├── raw/                  # Raw API pulls from Meta Ad Library
│   ├── processed/            # Cleaned DataFrames used in analysis
│   └── acs/                  # U.S. Census / ACS contextual data
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_targeting_analysis.ipynb
│   ├── 03_spend_by_demographic.ipynb
│   ├── 04_content_nlp_analysis.ipynb
│   └── 05_shed_econometric_analysis.ipynb
├── scripts/
│   ├── meta_api_pull.py      # Meta Ad Library API data collection
│   ├── census_pull.py        # ACS data collection via Census API
│   └── nlp_analysis.py       # NLP pipeline (readability, sentiment, keywords)
├── figures/                  # Output visualizations
├── requirements.txt
└── README.txt

================================================================================
DATA SOURCES
================================================================================

1. Meta Ad Library API (Primary)
   Meta's public Ad Library (facebook.com/ads/library) provides programmatic
   access to ad-level data including:
   - Age and gender targeting ranges
   - Estimated audience size and impressions
   - Ad spend ranges
   - Ad creative: headline, body text, image/video copy
   - Duration and region of ad delivery

2. U.S. Census / American Community Survey (Contextual)
   ACS data provides population-level financial fragility indicators (emergency
   savings rates, debt-to-income ratios, financial literacy) broken down by age
   group, allowing us to contextualize targeting patterns beyond a simple "they
   targeted young people" claim.

3. Ad Creative Text (NLP)
   Ad copy returned by the Meta API is analyzed using NLP techniques to test
   whether messaging strategy differs systematically across targeted demographics.

4. Federal Reserve SHED 2024 (Survey of Household Economics and Decisionmaking)
   Used for advanced econometric modeling (PCA, OLS, Propensity Score Matching)
   to firmly establish the link between financial fragility and BNPL usage.

================================================================================
REPRODUCING THE RESULTS
================================================================================

Prerequisites
-------------
Install dependencies:

    pip install -r requirements.txt

You will need:

  - A Meta Developer Account with access to the Ad Library API. Register at
    developers.facebook.com. The API is free but requires account verification.

  - A U.S. Census API key (free). Register at:
    api.census.gov/data/key_signup.html

Store your credentials in a .env file in the root directory (this file is
.gitignored):

    META_API_TOKEN=your_token_here
    CENSUS_API_KEY=your_key_here

Step-by-Step
------------
1. Collect Meta Ad Library data:
       python scripts/meta_api_pull.py
   Pulls all active and recent ads for the six BNPL providers and saves to
   data/raw/.

2. Collect ACS contextual data:
       python scripts/census_pull.py
   Pulls relevant ACS variables and saves to data/acs/.

3. Run analysis notebooks in order:
       notebooks/01_data_collection.ipynb       -- Cleans and structures raw data
       notebooks/02_targeting_analysis.ipynb    -- Age/gender targeting + chi-square tests
       notebooks/03_spend_by_demographic.ipynb  -- Ad spend breakdown by age group
       notebooks/04_content_nlp_analysis.ipynb  -- Readability, sentiment, keywords
       notebooks/05_shed_econometric_analysis.ipynb -- PCA, OLS, and PSM on SHED data

================================================================================
ANALYSIS SUMMARY
================================================================================

Step                  | Method                                    | Output
----------------------|-------------------------------------------|-------------------------
Targeting Analysis    | Chi-square / proportion tests on age      | Age distribution plots
                      | targeting vs. general adult population    | per provider
Spend Intensity       | Aggregated spend by demographic bracket   | Bar charts: ad dollars
                      | per company                               | by age group
Readability           | Flesch-Kincaid scores on ad copy by       | Readability comparison
                      | target age group                          | across demographics
Keyword Analysis      | Frequency counts: risk language vs.       | Keyword frequency
                      | urgency/aspirational language             | heatmaps
Sentiment             | VADER / TextBlob polarity scores          | Sentiment distributions
                      | on ad copy                                | by target group
Vulnerability PCA     | Principal Component Analysis on fragility | Continuous Vulnerability
                      | variables (EF1, atleast_okay, E2, K0)     | Score (PC1)
OLS Linear Model      | Predict BNPL use via vulnerability score  | Summary table & interaction
                      | and demographic interaction terms         | plot vs age
PSM                   | Propensity Score Matching by demographics | Balanced T-test between
                      | to isolate the effect of fragility        | Fragile & Non-Fragile

================================================================================
ETHICS CONNECTION
================================================================================

This project engages three ethics themes from DATA 259:

1. Autonomy and Informed Consent
   BNPL apps collect an average of 14 data types and share data with third
   parties to power targeting, often without users' meaningful awareness.

2. The "Legal but Harmful" Problem
   No law prevents financial products from advertising to the people most likely
   to be harmed by them. This project names that gap.

3. Algorithmic Amplification of Inequality
   Meta's ad delivery algorithm further concentrates impressions within a
   targeting range toward the most "engaged" (often most financially anxious)
   users, making this an algorithmic fairness problem beyond a simple marketing
   one.

================================================================================
DEPENDENCIES
================================================================================

See requirements.txt. Key packages:

    requests
    pandas
    numpy
    matplotlib
    seaborn
    textblob
    vaderSentiment
    textstat
    census
    python-dotenv

================================================================================
DATA AVAILABILITY
================================================================================

Raw API data pulled from the Meta Ad Library is public and is included in
data/raw/. ACS data is public and is included in data/acs/. No private or
personally identifiable data was used in this project.

================================================================================
REFERENCES
================================================================================

CFPB (2023). Consumer Use of Buy Now, Pay Later. Office of Research.

CFPB (2025). BNPL Market Trends and Consumer Impacts.

Mishra et al. (2025). Regulatory and Ethical Challenges in AI-Driven Credit Risk
  Assessment for BNPL. Journal of Business and Management Studies, 7(2), 42-51.

Behera, Astvansh & Kopalle (2025). Buy Now, Pay Later: AI Usage, Inherent
  Tensions, and Implications. Management and Business Review.

Threadgold et al. (2024). Buy Now, Pay Later Technologies and the Gamification
  of Debt. Journal of Cultural Economy.

Richmond Fed (2025). Buy Now, Pay Later: Market Impact and Policy Considerations.
  Economic Brief 25-03.

BrandTotal (2021). Social Ad Snapshot: Buy Now, Pay Later. [Industry report].
