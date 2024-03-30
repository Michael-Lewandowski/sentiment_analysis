'''
1. Import the necessary libraries:
   - pandas for data manipulation
   - spacy for natural language processing
   - textblob for sentiment analysis
   - matplotlib for visualization

2. Load the English language model from spaCy.

3. Define the path to the dataset containing Amazon product reviews.

4. Load the dataset using pandas, specifying low memory usage.

5. Define a function, preprocess_text, that:
   - Takes a piece of text as input.
   - Processes the text using spaCy to tokenize it and remove stopwords and punctuation.
   - Returns the preprocessed text with each token's lemma joined back into a string.

6. Preprocess the 'reviews.text' column in the dataset by:
   - Dropping rows where 'reviews.text' is NaN (missing).
   - Applying the preprocess_text function to each review.
   - Storing the results in a new column, 'preprocessed_reviews'.

7. Define a function, analyze_sentiment, that:
   - Takes a piece of text as input.
   - Uses TextBlob to perform sentiment analysis on the text.
   - Returns the sentiment polarity score.

8. Apply sentiment analysis to the preprocessed reviews by:
   - Using the analyze_sentiment function on each preprocessed review.
   - Storing the sentiment scores in a new column, 'sentiment_scores'.

9. Define a function, categorize_sentiment, that categorizes the sentiment based on the score:
   - If the score is greater than 0.05, it's considered 'Positive'.
   - If the score is less than -0.05, it's considered 'Negative'.
   - Otherwise, it's considered 'Neutral'.

10. Categorize each review's sentiment by applying the categorize_sentiment function to the sentiment scores.

11. Display the first few rows of the DataFrame, showing the original text, preprocessed text, sentiment scores, and sentiment categories.

12. Count the number of reviews in each sentiment category and display these counts.

13. Create a bar chart to visualize the number of reviews in each sentiment category:
    - Set the figure size.
    - Plot a bar chart using the sentiments as the x-axis and their counts as the y-axis, coloring each bar according to the sentiment category.
    - Set the title, x-label, and y-label of the chart.

14. Display the bar chart.
'''

# Import necessary libraries for preprocessing, sentiment analysis, and handling the dataset
import pandas as pd
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# UProvide a path to location of the Amazon_product_reviews.csv file on the system
dataset_path = '/Data Science/PYTHON EXERCISES/T21 - Capstone Project - NLP Applications/Amazon_product_reviews.csv'
reviews_df = pd.read_csv(dataset_path, low_memory=False)

# Function to preprocess text: remove stopwords, punctuation, and perform lemmatization
def preprocess_text(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

# Preprocess the 'reviews.text' column
reviews_df.dropna(subset=['reviews.text'], inplace=True)  # Remove rows where 'reviews.text' is NaN
reviews_df['preprocessed_reviews'] = reviews_df['reviews.text'].apply(preprocess_text)

# Function for performing sentiment analysis using TextBlob
def analyze_sentiment(review):
    blob = TextBlob(review)
    return blob.sentiment.polarity

# Apply sentiment analysis to preprocessed reviews
reviews_df['sentiment_scores'] = reviews_df['preprocessed_reviews'].apply(analyze_sentiment)

# Function to categorize sentiment based on sentiment scores
def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Categorize sentiment
reviews_df['sentiment_category'] = reviews_df['sentiment_scores'].apply(categorize_sentiment)

# Display the first few rows of the DataFrame with sentiment analysis results
print(reviews_df[['reviews.text', 'preprocessed_reviews', 'sentiment_scores', 'sentiment_category']].head())

# Count the number of reviews in each sentiment category
sentiment_counts = reviews_df['sentiment_category'].value_counts()

# Display the counts
print(sentiment_counts)

# Names of sentiments
sentiments = list(sentiment_counts.keys())

# Corresponding counts
counts = sentiment_counts.values

# Create a bar chart
plt.figure(figsize=(8, 6))
plt.bar(sentiments, counts, color=['green', 'blue', 'red'])
plt.title('Sentiment Analysis of Amazon Product Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()
