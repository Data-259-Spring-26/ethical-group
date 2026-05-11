import pandas as pd
from textstat import flesch_kincaid_grade
df = pd.DataFrame({'body': ['Buy now, pay later! Instant approval today.']})
for text in df['body']:
    print("FK:", flesch_kincaid_grade(text))
