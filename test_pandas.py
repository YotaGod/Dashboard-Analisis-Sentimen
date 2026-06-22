import pandas as pd
import numpy as np

try:
    df = pd.read_csv('hasil_review_Tangerang_Live.csv')
    print("Columns:", df.columns.tolist())
    
    date_col = 'date' if 'date' in df.columns else 'tanggal'
    df['date_parsed'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce').dt.date
    daily_sentiment = df.groupby('date_parsed').size().reset_index(name='count')
    print("Date parsing success, dates head:")
    print(daily_sentiment.head())
    
    print("\nRating unique values:")
    if 'rating' in df.columns:
        print(df['rating'].unique())
        
except Exception as e:
    print("Error:", e)
