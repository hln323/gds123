import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from cleantext import clean
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# 1. Retrieve the data
url = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"
df = pd.read_csv(url)

# 2. Apply your custom cleaning function from Exercise 1
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    
    # Replace emails
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)
    
    # Replace dates (simplified approach)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '<DATE>', text)
    text = re.sub(r'\d{1,2}-\d{1,2}-\d{2,4}', '<DATE>', text)
    
    # Replace numbers
    text = re.sub(r'\b\d+\b', '<NUM>', text)
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# Apply cleaning to content column
df['clean_content'] = df['content'].apply(clean_text)

# 3. NLTK-specific processing

# Tokenization using NLTK
df['tokens'] = df['clean_content'].apply(word_tokenize)

# Calculate vocabulary size before further processing
all_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]
vocab_size_original = len(set(all_tokens))
print(f"Original vocabulary size: {vocab_size_original}")

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['tokens_no_stopwords'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token not in stop_words])

# Calculate vocabulary size after stopword removal
all_tokens_no_stopwords = [token for tokens_list in df['tokens_no_stopwords'] for token in tokens_list]
vocab_size_no_stopwords = len(set(all_tokens_no_stopwords))

# Calculate reduction rate
stopword_reduction_rate = ((vocab_size_original - vocab_size_no_stopwords) / vocab_size_original) * 100
print(f"Vocabulary size after stopword removal: {vocab_size_no_stopwords}")
print(f"Reduction rate after stopword removal: {stopword_reduction_rate:.2f}%")

# Apply stemming using Porter Stemmer
ps = PorterStemmer()
df['tokens_stemmed'] = df['tokens_no_stopwords'].apply(lambda tokens: [ps.stem(token) for token in tokens])

# Calculate vocabulary size after stemming
all_tokens_stemmed = [token for tokens_list in df['tokens_stemmed'] for token in tokens_list]
vocab_size_stemmed = len(set(all_tokens_stemmed))

# Calculate reduction rate after stemming
stemming_reduction_rate = ((vocab_size_no_stopwords - vocab_size_stemmed) / vocab_size_no_stopwords) * 100
print(f"Vocabulary size after stemming: {vocab_size_stemmed}")
print(f"Reduction rate after stemming: {stemming_reduction_rate:.2f}%")

# Explain which procedures and libraries were used and why they're appropriate
print("\nProcedures and libraries used:")
print("1. Regular expressions (re) - For pattern matching and replacing URLs, emails, dates, and numbers")
print("2. NLTK tokenization - For splitting text into word tokens properly, handling punctuation and special cases")
print("3. NLTK stopwords - For removing common words that don't carry significant meaning")
print("4. NLTK Porter Stemmer - For reducing words to their base/root form to decrease vocabulary size")
print("5. cleantext library - For offering comprehensive text cleaning capabilities")