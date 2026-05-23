"""
collect_bnpl_data.py
---------------------
Saves two CSVs to your Downloads folder:

  ~/Downloads/bnpl_website_copy.csv     — marketing copy from 6 BNPL providers
  ~/Downloads/cfpb_bnpl_complaints.csv  — CFPB consumer complaints (BNPL-filtered)

BEFORE RUNNING — download the CFPB data manually (one-time, ~2 min):
  1. Go to: https://www.consumerfinance.gov/data-research/consumer-complaints/
  2. Click "Download all complaint data" (top-right of the page)
  3. Save the file to your Downloads folder
  4. The file will be named something like: complaints.csv  or  cfpb_complaints.csv
  5. Run this script — it will find and process it automatically

Requirements:
    pip install requests beautifulsoup4 pandas playwright
    playwright install chromium
"""

import glob
import io
import os
import sys
import time
import random
import zipfile
from datetime import datetime

import requests
import urllib3
import pandas as pd
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("WARNING: playwright not installed — Klarna pages will be skipped.")
    print("  Run: pip install playwright && playwright install chromium\n")

DOWNLOADS      = os.path.expanduser("~/Downloads")
WEBSITE_OUT    = os.path.join(DOWNLOADS, "bnpl_website_copy.csv")
COMPLAINTS_OUT = os.path.join(DOWNLOADS, "cfpb_bnpl_complaints.csv")

# ============================================================
# PART 1 — BNPL WEBSITE SCRAPER
# ============================================================

PAGES = [
    {"provider": "Klarna",   "page_type": "homepage",     "url": "https://www.klarna.com/us/"},
    {"provider": "Klarna",   "page_type": "how_it_works", "url": "https://www.klarna.com/us/what-is-klarna/"},
    {"provider": "Klarna",   "page_type": "faq",          "url": "https://www.klarna.com/us/customer-service/"},
    {"provider": "Affirm",   "page_type": "homepage",     "url": "https://www.affirm.com/"},
    {"provider": "Affirm",   "page_type": "how_it_works", "url": "https://www.affirm.com/how-it-works"},
    {"provider": "Affirm",   "page_type": "faq",          "url": "https://helpcenter.affirm.com/s/"},
    {"provider": "Afterpay", "page_type": "homepage",     "url": "https://www.afterpay.com/en-US"},
    {"provider": "Afterpay", "page_type": "how_it_works", "url": "https://www.afterpay.com/en-US/how-it-works"},
    {"provider": "Afterpay", "page_type": "faq",          "url": "https://help.afterpay.com/hc/en-us"},
    {"provider": "Zip",      "page_type": "homepage",     "url": "https://zip.co/us"},
    {"provider": "Zip",      "page_type": "how_it_works", "url": "https://zip.co/us/how-it-works"},
    {"provider": "Zip",      "page_type": "faq",          "url": "https://help.zip.co/hc/en-us"},
    {"provider": "Sezzle",   "page_type": "homepage",     "url": "https://sezzle.com/"},
    {"provider": "Sezzle",   "page_type": "how_it_works", "url": "https://sezzle.com/how-it-works"},
    {"provider": "Sezzle",   "page_type": "faq",          "url": "https://help.sezzle.com/hc/en-us"},
    {"provider": "PayPal",   "page_type": "homepage",     "url": "https://www.paypal.com/us/digital-wallet/ways-to-pay/buy-now-pay-later"},
    {"provider": "PayPal",   "page_type": "how_it_works", "url": "https://www.paypal.com/us/webapps/mpp/paypal-instalments"},
    {"provider": "PayPal",   "page_type": "faq",          "url": "https://www.paypal.com/us/cshelp/article/what-is-pay-later-and-how-does-it-work-help267"},
]

CONTENT_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "span", "button", "a"]
MIN_LEN      = 25
SKIP_TAGS    = {"script", "style", "noscript", "header", "nav", "footer"}
SKIP_CLASSES = {"nav", "footer", "cookie", "menu", "breadcrumb", "skip-link"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_page(url):
    for verify in [True, False]:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20, verify=verify)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.SSLError:
            if not verify:
                print("    SSL error — skipping")
                return None
            print("    SSL cert error — retrying without verification")
        except requests.exceptions.HTTPError as e:
            print(f"    HTTP {e.response.status_code} — skipping")
            return None
        except Exception as e:
            print(f"    Error: {e} — skipping")
            return None


def fetch_page_js(url):
    if not PLAYWRIGHT_AVAILABLE:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx  = browser.new_context(user_agent=HEADERS["User-Agent"], locale="en-US")
            page = ctx.new_page()
            page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf}", lambda r: r.abort())
            page.goto(url, timeout=30_000, wait_until="networkidle")
            page.wait_for_timeout(2_000)
            html = page.content()
            browser.close()
        return BeautifulSoup(html, "html.parser")
    except Exception as e:
        print(f"    Playwright error: {e}")
        return None


def is_boilerplate(tag):
    if tag.name in SKIP_TAGS:
        return True
    if set(tag.get("class", [])) & SKIP_CLASSES:
        return True
    for parent in tag.parents:
        if parent.name in SKIP_TAGS:
            return True
        if set(parent.get("class", [])) & SKIP_CLASSES:
            return True
    return False


def extract_text_blocks(soup, url, provider, page_type):
    rows, seen, now = [], set(), datetime.now().isoformat()
    for tag_name in CONTENT_TAGS:
        for tag in soup.find_all(tag_name):
            if is_boilerplate(tag):
                continue
            text = " ".join(tag.get_text(separator=" ", strip=True).split())
            if len(text) < MIN_LEN or text in seen:
                continue
            seen.add(text)
            rows.append({
                "provider":   provider,
                "page_type":  page_type,
                "url":        url,
                "html_tag":   tag_name,
                "text":       text,
                "scraped_at": now,
            })
    return rows


def scrape_websites():
    print("\n" + "="*60)
    print("PART 1 OF 2 — Scraping BNPL marketing websites")
    print("="*60)
    all_rows = []

    for page in PAGES:
        provider, page_type, url = page["provider"], page["page_type"], page["url"]
        print(f"  {provider} — {page_type}")
        soup = fetch_page_js(url) if provider == "Klarna" else fetch_page(url)

        if soup is None:
            all_rows.append({
                "provider": provider, "page_type": page_type, "url": url,
                "html_tag": "ERROR", "text": "FETCH_FAILED",
                "scraped_at": datetime.now().isoformat(),
            })
        else:
            rows = extract_text_blocks(soup, url, provider, page_type)
            print(f"    -> {len(rows)} text blocks")
            all_rows.extend(rows)

        time.sleep(random.uniform(1.5, 3.0))

    df = pd.DataFrame(all_rows)
    df.to_csv(WEBSITE_OUT, index=False, encoding="utf-8")
    good = df[df["text"] != "FETCH_FAILED"]
    print(f"\n  Done. {len(good)} text blocks across "
          f"{good['provider'].nunique()}/6 providers -> {WEBSITE_OUT}")
    return df


# ============================================================
# PART 2 — CFPB LOCAL FILE PROCESSING
# ============================================================

BNPL_COMPANIES = ["klarna", "affirm", "afterpay", "zip", "sezzle", "paypal"]
BNPL_KEYWORDS  = [
    "buy now pay later", "buy now, pay later", "bnpl",
    "pay in 4", "pay in four", "pay later",
    "klarna", "affirm", "afterpay", "sezzle",
]
DATE_FROM = "2020-01-01"

# CFPB column name -> our column name
COL_MAP = {
    "Complaint ID":                     "complaint_id",
    "Date received":                    "date_received",
    "Product":                          "product",
    "Sub-product":                      "sub_product",
    "Issue":                            "issue",
    "Sub-issue":                        "sub_issue",
    "Consumer complaint narrative":     "consumer_narrative",
    "Company":                          "company",
    "State":                            "state",
    "ZIP code":                         "zip_code",
    "Tags":                             "tags",
    "Submitted via":                    "submitted_via",
    "Date sent to company":             "date_sent",
    "Company response to consumer":     "company_response",
    "Timely response?":                 "timely",
    "Consumer disputed?":               "consumer_disputed",
}


def find_cfpb_file():
    """
    Look for the CFPB CSV (or zip) in Downloads.
    Returns the path, or None if not found.
    """
    patterns = [
        os.path.join(DOWNLOADS, "complaints.csv"),
        os.path.join(DOWNLOADS, "cfpb_complaints.csv"),
        os.path.join(DOWNLOADS, "complaints.csv.zip"),
        os.path.join(DOWNLOADS, "cfpb_complaints.csv.zip"),
        # wildcard fallback
        os.path.join(DOWNLOADS, "*complaint*.csv"),
        os.path.join(DOWNLOADS, "*complaint*.zip"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # Return the most recently modified match
            return max(matches, key=os.path.getmtime)
    return None


def load_cfpb_csv(path):
    """Read the CFPB file (csv or zip) and return a DataFrame iterator."""
    if path.endswith(".zip"):
        print(f"  Opening zip: {os.path.basename(path)}")
        with zipfile.ZipFile(path) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            data = io.BytesIO(zf.read(csv_name))
    else:
        print(f"  Opening CSV: {os.path.basename(path)}")
        data = path

    return pd.read_csv(
        data,
        chunksize=50_000,
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )


def flag_misleading_language(narrative):
    n = (narrative or "").lower()
    return {
        "flag_believed_free":       any(p in n for p in [
            "thought it was free", "believed it was free", "no interest",
            "interest free", "interest-free", "0%", "zero interest",
            "no fees", "thought there were no fees"]),
        "flag_hidden_fees":         any(p in n for p in [
            "hidden fee", "unexpected fee", "surprise", "didn't know",
            "did not know", "wasn't aware", "was not aware",
            "not disclosed", "undisclosed", "didn't tell me",
            "did not tell me", "fine print", "small print"]),
        "flag_unexpected_late_fee": any(p in n for p in [
            "late fee", "penalty", "charged extra", "additional charge",
            "missed payment", "automatic payment", "autopay"]),
        "flag_debt_accumulation":   any(p in n for p in [
            "multiple loans", "several loans", "debt", "owe more than",
            "balance grew", "accumulated", "stacked", "can't afford"]),
        "flag_credit_impact":       any(p in n for p in [
            "credit score", "credit report", "credit check",
            "hard inquiry", "hard pull", "affected my credit"]),
        "flag_misleading_ad":       any(p in n for p in [
            "misleading", "deceptive", "false advertising", "advertised as",
            "marketed as", "told me", "promised", "guarantee",
            "bait and switch"]),
        "flag_loan_confusion":      any(p in n for p in [
            "didn't realize it was a loan", "did not realize it was a loan",
            "thought it was", "believed it was a", "not a loan",
            "just a payment plan", "not credit"]),
    }


def collect_complaints():
    print("\n" + "="*60)
    print("PART 2 OF 2 — Processing CFPB complaint data")
    print("="*60)

    cfpb_path = find_cfpb_file()

    if cfpb_path is None:
        print("""
  *** CFPB file not found in your Downloads folder ***

  Please download it manually (takes ~2 minutes):
    1. Open this URL in your browser:
       https://www.consumerfinance.gov/data-research/consumer-complaints/
    2. Click "Download all complaint data" (top-right of the complaints table)
    3. Save the file to your Downloads folder
    4. Re-run this script

  The file will be named complaints.csv or similar (~300 MB).
""")
        return pd.DataFrame()

    print(f"  Found CFPB file: {cfpb_path}")
    size_mb = os.path.getsize(cfpb_path) / 1024 / 1024
    print(f"  File size: {size_mb:.0f} MB")

    try:
        reader = load_cfpb_csv(cfpb_path)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return pd.DataFrame()

    print(f"  Filtering for BNPL complaints since {DATE_FROM}...")
    kept, total_rows = [], 0

    for chunk in reader:
        total_rows += len(chunk)

        # Standardise column names
        chunk = chunk.rename(columns={k: v for k, v in COL_MAP.items() if k in chunk.columns})

        # Date filter
        if "date_received" in chunk.columns:
            chunk["date_received"] = pd.to_datetime(chunk["date_received"], errors="coerce")
            chunk = chunk[chunk["date_received"] >= DATE_FROM]

        if chunk.empty:
            continue

        company   = chunk.get("company",            pd.Series([""] * len(chunk))).fillna("").str.lower()
        narrative = chunk.get("consumer_narrative",  pd.Series([""] * len(chunk))).fillna("").str.lower()

        mask = (
            company.apply(  lambda c: any(b in c for b in BNPL_COMPANIES))
            | narrative.apply(lambda n: any(kw in n for kw in BNPL_KEYWORDS))
        )
        kept.append(chunk[mask])
        print(f"\r  Scanned {total_rows:,} rows — kept {sum(len(c) for c in kept):,} BNPL complaints...",
              end="", flush=True)

    print()

    if not kept:
        print("  No BNPL complaints found. "
              "Make sure the file is the full CFPB dataset, not a pre-filtered export.")
        return pd.DataFrame()

    df = pd.concat(kept, ignore_index=True)
    print(f"  Total: {len(df):,} BNPL complaints")

    print("  Adding deception flags...")
    flags_df = pd.DataFrame(
        df["consumer_narrative"].fillna("").apply(flag_misleading_language).tolist()
    )
    df = pd.concat([df, flags_df], axis=1)
    df = df.sort_values("date_received", ascending=False)

    df.to_csv(COMPLAINTS_OUT, index=False, encoding="utf-8")
    narr = df["consumer_narrative"].notna().sum()
    print(f"  Saved {len(df):,} rows ({narr:,} with narrative text) -> {COMPLAINTS_OUT}")
    return df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\nBNPL Data Collection — Single-Run Script")
    print("Outputs go to:", DOWNLOADS)
    print("="*60)

    website_df    = scrape_websites()
    complaints_df = collect_complaints()

    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
    good = website_df[website_df["text"] != "FETCH_FAILED"]
    print(f"  Website copy:  {len(good):,} text blocks -> {WEBSITE_OUT}")
    if not complaints_df.empty:
        print(f"  Complaints:    {len(complaints_df):,} rows -> {COMPLAINTS_OUT}")
    else:
        print(f"  Complaints:    not processed yet (see instructions above)")
