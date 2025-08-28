# Step 1: Install necessary libraries (if not already installed)
!pip install -q textblob nltk

# Import required libraries
import pandas as pd
import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google.colab import files
from datetime import timedelta

# Step 2: Upload the video dataset (uploaded or from Google Drive)
df_videos = pd.read_csv('videos_clean.csv')  # Use the filename as it appears after upload

# Step 3: Data Preprocessing (Handling Missing Values)
# Handle missing values in 'description' and 'title' columns by filling NaNs with empty strings
df_videos['description'] = df_videos['description'].fillna('')
df_videos['title'] = df_videos['title'].fillna('')

# Handle missing values in 'viewCount', 'likeCount', 'favouriteCount', 'commentCount' by replacing NaN with 0
df_videos['viewCount'].fillna(0, inplace=True)
df_videos['likeCount'].fillna(0, inplace=True)
df_videos['favouriteCount'].fillna(0, inplace=True)
df_videos['commentCount'].fillna(0, inplace=True)

# Step 4: Feature Engineering

# 4.1 Duration of video (contentDuration) in minutes and seconds
def parse_duration(duration):
    try:
        # Format: PT#H#M#S, PT#M#S, PT#H#M
        t = re.match(r"PT(\d+H)?(\d+M)?(\d+S)?", duration)
        hours = int(t.group(1)[:-1]) if t.group(1) else 0
        minutes = int(t.group(2)[:-1]) if t.group(2) else 0
        seconds = int(t.group(3)[:-1]) if t.group(3) else 0
        return hours * 60 + minutes + seconds / 60
    except:
        return None

df_videos['video_duration_minutes'] = df_videos['contentDuration'].apply(parse_duration)

# Step 1: Convert the 'publishedAt' column to datetime format if it's not already
df_videos['publishedAt'] = pd.to_datetime(df_videos['publishedAt'], errors='coerce')

# Step 2: Create the 'is_weekend' column
df_videos['is_weekend'] = df_videos['publishedAt'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)


# 4.4 Hashtags in title and description (using regex to count occurrences)
df_videos['hashtags_in_title'] = df_videos['title'].apply(lambda x: len(re.findall(r"#\w+", x)))  # Count hashtags
df_videos['hashtags_in_description'] = df_videos['description'].apply(lambda x: len(re.findall(r"#\w+", x)))  # Count hashtags

# 4.7 Total Hashtags (from tags column)
def count_hashtags_in_tags(tags):
    # Check if tags is a string and not NaN
    if isinstance(tags, str):
        # Remove the square brackets and split by commas
        tags_list = tags.strip("[]").replace("'", "").split(", ")

        # Return the total count of tags
        return len(tags_list)

    # If it's not a string or it's NaN, return 0
    return 0

# Apply the function to the 'tags' column
df_videos['total_hashtags_tags'] = df_videos['tags'].apply(count_hashtags_in_tags)

df_videos['total_hashtags'] = df_videos['hashtags_in_title'] + df_videos['hashtags_in_description'] + df_videos['total_hashtags_tags']

df_videos['total_mentions'] = df_videos['description'].apply(lambda x: len(re.findall(r"@\w+", x)))

df_videos['likes_per_view'] = df_videos['likeCount'] / df_videos['viewCount']

df_videos['comments_per_view'] = df_videos['commentCount'] / df_videos['viewCount']

df_videos['favourites_per_view'] = df_videos['favouriteCount'] / df_videos['viewCount']

# 4.2 Sentiment analysis for the title and description (using TextBlob)
df_videos['title_sentiment_score'] = df_videos['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
df_videos['description_sentiment_score'] = df_videos['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

# 4.3 Sentiment analysis (VADER for compound score) for title and description
sid = SentimentIntensityAnalyzer()
df_videos['title_sentiment_score_vader'] = df_videos['title'].apply(lambda x: sid.polarity_scores(x)['compound'])
df_videos['description_sentiment_score_vader'] = df_videos['description'].apply(lambda x: sid.polarity_scores(x)['compound'])

# 4.5 Video Popularity Score (combined measure of views, likes, favourites, and comments)
df_videos['popularity_score'] = (df_videos['viewCount'] + df_videos['likeCount'] + df_videos['favouriteCount'] + df_videos['commentCount']) / 4

# 4.6 Topic Categories (Count of categories)
df_videos['topic_categories_count'] = df_videos['topicCategories'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)

# Step 5: Output the processed data

# Save the processed data to a CSV file
df_videos.to_csv('final_processed_videos.csv', index=False)

# Optional: Display the first few rows of the processed dataset
df_videos.head()
