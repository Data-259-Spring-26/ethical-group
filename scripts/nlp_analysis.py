import pandas as pd
from textstat import flesch_kincaid_grade
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_text(df, text_col='body'):
    print("Running NLP analysis pipeline...")
    analyzer = SentimentIntensityAnalyzer()
    
    results = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        
        # Readability
        try:
            fk_grade = flesch_kincaid_grade(text)
        except:
            fk_grade = 0.0
            
        # Sentiment
        vader_scores = analyzer.polarity_scores(text)
        blob = TextBlob(text)
        
        # Keyword matching
        risk_keywords = ['fee', 'credit', 'interest', 'penalty', 'debt', 'risk', 'late']
        urgency_keywords = ['now', 'today', 'instant', 'fast', 'hurry', 'wait']
        
        text_lower = text.lower()
        risk_count = sum(1 for w in risk_keywords if w in text_lower)
        urgency_count = sum(1 for w in urgency_keywords if w in text_lower)
        
        res = row.to_dict()
        res['fk_grade'] = fk_grade
        res['vader_compound'] = vader_scores['compound']
        res['textblob_polarity'] = blob.sentiment.polarity
        res['risk_keyword_count'] = risk_count
        res['urgency_keyword_count'] = urgency_count
        results.append(res)
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test on a dummy df if run directly
    df = pd.DataFrame({'body': ['Buy now, pay later! Instant approval today.']})
    print(analyze_text(df))
