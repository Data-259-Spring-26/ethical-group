BNPL and Financial Vulnerability: An Ethical Analysis
DATA 259 -- Autumn 2025

Group Members: Max Linton, Alvin He, Charlie Yang


RESEARCH QUESTION:

Do BNPL providers disproportionately reach financially vulnerable consumers,
and does the content of BNPL marketing obscure or downplay the risks of
borrowing?


OVERVIEW:

This project answers its research question in two parts.

Part 1 (Analyses 1-4) uses the 2024 Survey of Household Economics and
Decisionmaking (SHED) to test whether financial vulnerability predicts BNPL
use, controlling for demographic confounders. We build a composite vulnerability
index via weighted PCA and estimate its association with BNPL adoption using a
weighted linear probability model and a stratified subgroup comparison.

Part 2 (Analysis 5) uses two text-based datasets to test whether BNPL marketing
language downplays risk. We scraped 417 text blocks from the marketing pages
of six BNPL providers, computed readability (Flesch-Kincaid), urgency-vs-risk
keyword ratios, and VADER sentiment, then cross-referenced those measures
against deception flag rates in 49,140 CFPB consumer complaints filed against
the same providers.

BNPL providers examined: Affirm, Afterpay, Klarna, PayPal, Sezzle, Zip.

REPOSITORY STRUCTURE:

.
├── notebooks/
│   ├── [Final]BNPL_Analysis_1_to_4_combined.ipynb   FINAL - SHED analyses 1-4
│   ├── [Final]Analysis_5_wording_analysis.ipynb     FINAL - text/CFPB analysis
│   ├── 01_data_collection.ipynb                     (legacy, see note below)
│   ├── 02_targeting_analysis.ipynb                  (legacy)
│   ├── 03_spend_by_demographic.ipynb                (legacy)
│   └── 04_content_nlp_analysis.ipynb                (legacy)
│
├── scripts/
│   ├── generate_paper_figures.py   Builds Analysis 5 paper figures (4, 5, 6)
│   ├── export_notebook_figures.py  Builds Analyses 1-4 paper figures (1, 2, 3)
│   ├── meta_api_pull.py            (legacy - unused in final paper)
│   ├── census_pull.py              (legacy - unused in final paper)
│   ├── nlp_analysis.py             (legacy - older NLP pipeline)
│   └── refactor_shed.py            Utility for splitting/regenerating notebooks
│
├── downloads/                      Inputs for Analysis 5 (mirror of ~/Downloads/)
│   ├── collect_bnpl_data.py        Data collection script
│   └── bnpl_website_copy.csv       Scraped marketing copy
│                                   (cfpb_bnpl_complaints.csv not included;
│                                    see Analysis 5 Data Collection below)
│
├── data/
│   ├── Raw/
│   │   └── public2024.csv.zip      SHED 2024 public-use file
│   ├── processed/
│   │   ├── shed_analysis.csv          Cleaned SHED variables for Analyses 1-4
│   │   ├── wls_coefs.json             Saved regression coefficients
│   │   ├── meta_ads_clean.csv         (legacy)
│   │   └── census_fragility_clean.csv (legacy)
│   ├── raw/
│   │   └── meta_ads.csv            (legacy - unused in final paper)
│   └── acs/
│       └── census_fragility.csv    (legacy - unused in final paper)
│
├── figures/
│   ├── paper/                      Publication figures 1-6 (in paper)
│   └── exploratory/                Notebook output plots, EDA, legacy work
│
├── Final Deliverable/
│   ├── Final_Paper.qmd             Quarto source for the paper
│   ├── Final_Paper.pdf             Compiled final paper
│   └── figures/                    Paper figures (copied from figures/paper/)
│
├── Poster/
│   ├── BNPL_Poster [Final].pptx    Final poster source
│   └── BNPL_Poster [Final].pdf     Final poster export
│
├── archive/                        Preliminary EDA, notebook generators
├── BNPL_Methodology.docx           Methodology write-up
├── peer_review_document.{md,pdf,html}  Peer review materials
├── requirements.txt
└── README.txt

NOTE ON LEGACY FILES
The project originally planned to use the Meta Ad Library API and U.S. Census
ACS data to study BNPL ad targeting. However, after data access constraints, we
pivoted to website scraping plus CFPB complaints for the marketing side
analysis, and to SHED 2024 for the demographic and vulnerability side.
Old/legacy files (i.e notebooks 01-04, meta_api_pull.py, census_pull.py, nlp_analysis.py,
and data/raw/meta_ads.csv) are retained for transparency about the project's
evolution but are NOT used by the final paper or poster.

DATA SOURCES (FINAL PAPER):

1. Survey of Household Economics and Decisionmaking (SHED), 2024
   Federal Reserve Board public-use file. 12,295 respondents, post-stratification
   weights. Source of BNPL use indicator (BNPL1) and seven financial-fragility
   indicators used to build the composite vulnerability index.
   Stored at: data/Raw/public2024.csv.zip

2. BNPL Provider Marketing Copy
   417 text blocks scraped from the homepage, "how it works," and FAQ pages of
   six BNPL providers (Affirm, Afterpay, Klarna, PayPal, Sezzle, Zip). Used as
   a proxy for marketing language. Klarna pages required a headless browser
   (Playwright) for JavaScript-rendered content. See "Analysis 5 Data
   Collection" below for details.
   Stored at: downloads/bnpl_website_copy.csv

3. CFPB Consumer Complaints
   Downloaded as a bulk CSV from the CFPB's public consumer complaint database:
       https://www.consumerfinance.gov/data-research/consumer-complaints/
   ("Download all complaint data" link.) Filtered to the six BNPL providers
   and to complaints filed since 2020, yielding 49,140 complaints (24,683
   with consumer narratives). Each narrative is tagged for seven types of
   consumer harm.

   NOTE: The filtered output file (cfpb_bnpl_complaints.csv) is too large to
   upload to GitHub and is therefore NOT included in this repository. To
   obtain it, run collect_bnpl_data.py as described in "Analysis 5 Data
   Collection" below. The script will download the bulk CFPB CSV, filter it
   to the six BNPL providers, and write cfpb_bnpl_complaints.csv to your
   ~/Downloads/ folder.


REPRODUCING THE RESULTS:


Prerequisites
-------------
Install dependencies:

    pip install -r requirements.txt
    playwright install chromium       # only needed if re-scraping Klarna

SHED data
---------
The SHED zip (data/Raw/public2024.csv.zip) is bundled with the repo.

Analysis 5 input data
---------------------
The Analysis 5 notebook reads two CSVs from your local ~/Downloads/ folder:

    bnpl_website_copy.csv      -- included in downloads/, copy to ~/Downloads/:
                                    cp downloads/bnpl_website_copy.csv ~/Downloads/

    cfpb_bnpl_complaints.csv   -- NOT included in the repo (too large for
                                  GitHub). You must generate it locally by
                                  running collect_bnpl_data.py. See "Analysis 5
                                  Data Collection" below.

Notebooks (in order)
--------------------
1. notebooks/[Final]BNPL_Analysis_1_to_4_combined.ipynb
       Analysis 1: Bivariate associations between BNPL use and demographic /
                   vulnerability predictors (Pearson chi-square with Holm
                   correction).
       Analysis 2: Composite financial vulnerability index via weighted PCA
                   on seven SHED items. Includes Cronbach's alpha diagnostic.
       Analysis 3: Weighted OLS linear probability model predicting BNPL use
                   from VULN_PCA1 plus demographic controls.
       Analysis 4: Stratified subgroup comparison (high vs. low vulnerability
                   within age-by-income strata), Welch t-tests with Holm
                   correction.

2. notebooks/[Final]Analysis_5_wording_analysis.ipynb
       Text block measures: Flesch-Kincaid grade level, urgency ratio
       (urgency keywords / urgency + risk keywords), VADER compound sentiment.
       Provider level aggregation, page type comparison (Welch t-tests),
       cross provider comparison against CFPB deception flag rates.

Paper figures
-------------
Regenerate all six paper figures with:

    python scripts/export_notebook_figures.py   # builds figures 1-3 (SHED)
    python scripts/generate_paper_figures.py    # builds figures 4-6 (text)

Outputs are written to figures/paper/ and copied to Final Deliverable/figures/.


ANALYSIS 5 DATA COLLECTION (downloads/):


The downloads/ folder mirrors the ~/Downloads/ directory that the Analysis 5
notebook reads from. To reproduce Analysis 5:

    1. Copy the website copy CSV from this repo into your Downloads folder:
           cp downloads/bnpl_website_copy.csv ~/Downloads/

    2. Generate the CFPB complaints CSV by running the data collection
       script (see instructions below). The script writes
       cfpb_bnpl_complaints.csv directly to ~/Downloads/.

    3. Run the Analysis 5 notebook.

Contents of downloads/
----------------------
    collect_bnpl_data.py        Script that generates both CSVs below.
    bnpl_website_copy.csv       417 marketing text blocks scraped from six
                                BNPL providers' homepage, how-it-works, and
                                FAQ pages.

    cfpb_bnpl_complaints.csv    NOT INCLUDED -- the filtered CFPB file is too
                                large to upload to GitHub. Run the script
                                below to generate it.

How we collected the data
--------------------------
collect_bnpl_data.py runs in two parts.

Part 1 -- Website scraping. Uses requests + BeautifulSoup to fetch homepage,
"how it works," and FAQ pages from Affirm, Afterpay, Klarna, PayPal, Sezzle,
and Zip. Klarna's pages are JavaScript-rendered, so they are fetched through
a headless Chromium browser via playwright. Each text block (at least 25
characters long) is saved with its provider, page type, and HTML tag.

Part 2 -- CFPB complaint filtering. Loads a bulk CSV download of all CFPB
consumer complaints, filters to the six BNPL providers, and to complaints
filed since 2020, and tags each consumer narrative for seven types of
deception language (hidden fees, unexpected late fees, debt accumulation,
credit impact, misleading advertising, loan confusion, belief the service
was free).

The bulk CFPB complaint CSV must be downloaded manually before running
Part 2:

    1. Go to https://www.consumerfinance.gov/data-research/consumer-complaints/
    2. Click "Download all complaint data" (top-right of the complaints table).
    3. Save the file to your ~/Downloads/ folder. The full file is roughly
       8 GB uncompressed and contains ~15 million complaint rows across all
       financial products since 2011; the script filters it down to the
       ~74,000 BNPL-relevant complaints.

To re-run the script
--------------------
    pip install requests beautifulsoup4 pandas playwright
    playwright install chromium
    python downloads/collect_bnpl_data.py

Outputs are written to ~/Downloads/. Once both CSVs are in ~/Downloads/, the
Analysis 5 notebook will run end-to-end.


ANALYSIS SUMMARY:


Analysis             | Method                                  | Key result
---------------------|-----------------------------------------|---------------------
1. Bivariate         | Chi-square on weighted contingency      | All 5 vulnerability
                     | tables, Holm correction                 | indicators significant
2. Composite Index   | Cronbach's alpha (0.40), weighted PCA   | PC1 explains 54% of
                     | on 7 fragility items                    | variance; orient as
                     |                                         | VULN_PCA1
3. Weighted LPM      | WLS regression with demographic         | +7.0 pp BNPL use per
                     | controls                                | 1 SD vulnerability
                     |                                         | (SE 0.004, p < 0.001)
4. Stratified Test   | Welch t-tests within age x income       | 11/33 strata sig.;
                     | strata, Holm correction                 | avg gap +10.1 pp
5. Text Analysis     | Flesch-Kincaid, urgency/risk keywords,  | Homepages more
                     | VADER sentiment; CFPB harm tagging      | positive than FAQs
                     |                                         | (p = 0.049). Sezzle
                     |                                         | highest urgency +
                     |                                         | highest flag rate


ETHICS CONNECTION:


This project engages three ethics themes from DATA 259:

1. Distributional Harm
   BNPL adoption is concentrated among populations who are both more financially
   at risk and less able to absorb the consequences of late fees, fee spirals,
   or other negative effects, such as credit damage. The same combination of light regulation, frictionless
   onboarding, and aspirational marketing that makes BNPL accessible also
   concentrates its risks on those who are least equipped to manage them.

2. Informational Asymmetry
   BNPL providers also hold detailed behavioral data on fee incidence, repayment
   patterns, and credit outcomes. Consumers are shown marketing copy whose
   sentiment is a lot more positive than the same providers' FAQ pages,
   and 18-25% of CFPB complaint narratives across multiple providers contain
   explicit language about being misled.

3. The "Legal but Harmful" Problem
   BNPL operates with substantially less disclosure and underwriting scrutiny
   than traditional credit, as there are no laws that prevent financial products from advertising
   to the populations most likely to be harmed by them. This project documents
   that gap (empirically).


DEPENDENCIES:


See requirements.txt. Key packages:

    pandas, numpy, scipy, statsmodels    Data manipulation, statistics
    matplotlib, seaborn                  Figures
    textstat                             Flesch-Kincaid readability
    vaderSentiment, textblob             Sentiment analysis
    requests, beautifulsoup4             Web scraping
    playwright                           Headless browser (for Klarna)
    python-dotenv                        Environment variable handling


DATA AVAILABILITY:


In terms of data availability, all data used in the final paper is public. The SHED 2024 public-use file is
distributed by the Federal Reserve Board, the CFPB consumer complaints are
distributed by the U.S. Consumer Financial Protection Bureau, and the BNPL website
content was scraped from the providers' public facing marketing pages. No
private or personally identifiable data was used in our project. CFPB
narratives are redacted at source by the CFPB before publication.

The filtered CFPB complaints file (cfpb_bnpl_complaints.csv) is not included
in this repository because of GitHub's file size limits, but can be regenerated
exactly by running downloads/collect_bnpl_data.py.


REFERENCES:

CFPB (2023). Consumer Use of Buy Now, Pay Later. Office of Research.
CFPB (2025). BNPL Market Trends and Consumer Impacts.
Federal Reserve Board (2024). Survey of Household Economics and Decisionmaking
   (SHED), 2024 public-use file.
Mishra et al. (2025). Regulatory and Ethical Challenges in AI-Driven Credit
   Risk Assessment for BNPL. Journal of Business and Management Studies, 7(2),
   42-51.
Behera, Astvansh & Kopalle (2025). Buy Now, Pay Later: AI Usage, Inherent
   Tensions, and Implications. Management and Business Review.
Threadgold et al. (2024). Buy Now, Pay Later Technologies and the Gamification
   of Debt. Journal of Cultural Economy.
Richmond Fed (2025). Buy Now, Pay Later: Market Impact and Policy
   Considerations. Economic Brief 25-03.

AI USE:

As per what the syllabus allows, we used ChatGPT to generate the script for scraping
the BNPL website data, as API timeouts and other issues forced us to ask artifical
intelligence to make a proper Python script to pull the data we needed for our project.

As mentioned in the syllabus:

"Can you use Chat-GPT to write code? If you are trying to accomplish something that isn’t
explicitly taught in this course for your project
(for example, building a dashboard to display a result),
you may use Chat-GPT or similar as a starting point for your project,
provided that you are documenting the prompts you used to generate the code."



We have attached the prompts we used to generate the scraping Python script below:



So my task right now is to find a dataset for the
"Does the content of those ads obscure or downplay the risks of borrowing “
part of our question.
We were going to use a Meta database,
but apparently you need a meta developer API/account to get access to it.
Do you know another way to get access to it or another dataset that we can use?


Would it be possible for you to do option 1 for me?
(Gave us a list of options for analysis, scraping was option 1)

Ok do the playwright approach
