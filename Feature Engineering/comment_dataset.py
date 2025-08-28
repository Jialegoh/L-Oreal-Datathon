import nltk

# Download the VADER lexicon (one-time setup)
nltk.download('vader_lexicon')

# Step 1: Install necessary libraries (if not already installed)
!pip install -q textblob nltk

# Import required libraries
import pandas as pd
import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google.colab import files

# Step 2: Upload the cleaned dataset (uploaded or from Google Drive)
df_comments = pd.read_csv('comments5ver1.csv')  # Use the filename as it appears after upload

# Step 3: Data Preprocessing (Handling Missing Values)
# Handle missing values in 'textOriginal' column by filling NaNs with empty strings
df_comments['textOriginal'] = df_comments['textOriginal'].fillna('')

# Handle missing values in 'likeCount' by replacing NaN with 0
df_comments['likeCount'].fillna(0, inplace=True)

# Step 4: Feature Engineering

# 4.1 Create text-based features: comment length, hashtag count, mention count
df_comments['comment_length'] = df_comments['textOriginal'].apply(lambda x: len(x.split()))  # word count
df_comments['comment_char_length'] = df_comments['textOriginal'].apply(lambda x: len(x))  # character count

# Hashtags count
df_comments['hashtag_count'] = df_comments['textOriginal'].apply(lambda x: len(re.findall(r"#\w+", x)))  # using regex

# Mentions count
df_comments['mention_count'] = df_comments['textOriginal'].apply(lambda x: len(re.findall(r"@\w+", x)))  # using regex

# 4.2 Sentiment analysis (TextBlob for sentiment polarity)
df_comments['sentiment_score'] = df_comments['textOriginal'].apply(lambda x: TextBlob(x).sentiment.polarity)  # Sentiment score from -1 to +1

# 4.3 Sentiment analysis (VADER for compound score)
sid = SentimentIntensityAnalyzer()
df_comments['sentiment_score_vader'] = df_comments['textOriginal'].apply(lambda x: sid.polarity_scores(x)['compound'])  # Sentiment from -1 to +1

# 4.4 Reaction-based engagement (Positive, Negative, Neutral)
df_comments['positive_reaction'] = df_comments['sentiment_score_vader'].apply(lambda x: 1 if x > 0 else 0)
df_comments['negative_reaction'] = df_comments['sentiment_score_vader'].apply(lambda x: 1 if x < 0 else 0)
df_comments['neutral_reaction'] = df_comments['sentiment_score_vader'].apply(lambda x: 1 if x == 0 else 0)

# 4.5 Time of Day Features (hour, day of week, weekend)
df_comments['publishedAt'] = pd.to_datetime(df_comments['publishedAt'], errors='coerce')  # Ensure it's in datetime format
df_comments['hour_of_day'] = df_comments['publishedAt'].dt.hour  # Hour of the day (0-23)
df_comments['day_of_week'] = df_comments['publishedAt'].dt.weekday  # Day of week (0 = Monday, 6 = Sunday)
df_comments['is_weekend'] = df_comments['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Is it the weekend (Saturday or Sunday)?

# 4.6 Author Activity Features
author_activity = df_comments.groupby('authorId').agg({
    'commentId': 'count',  # Total number of comments made by the author
    'sentiment_score': 'mean',  # Average sentiment score for the author
    'sentiment_score_vader': 'mean'  # Average VADER sentiment score for the author
}).reset_index()
author_activity.rename(columns={'commentId': 'author_comment_count'}, inplace=True)

# 4.7 Parent-Child Comment Relationship (is reply)
df_comments['is_reply'] = df_comments['parentCommentId'].apply(lambda x: 0 if x == 'root_comment' else 1)

# 4.8 Lexical Richness (Word diversity)
df_comments['lexical_richness'] = df_comments['textOriginal'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)

# 4.9 Emoji Count
df_comments['emoji_count'] = df_comments['textOriginal'].apply(lambda x: len(re.findall(r'[^\w\s,]', x)))  # Count emojis

# Step 5: Output the processed data

# Save the processed data to a CSV file
df_comments.to_csv('processed_comments5.csv', index=False)

# Optional: Display the first few rows of the processed dataset
df_comments.head()
