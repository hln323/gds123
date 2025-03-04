import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time
from tqdm import tqdm
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set a sample size to work with
SAMPLE_SIZE = 1000  # Adjust based on your system's capabilities

print(f"Loading and sampling {SAMPLE_SIZE} rows from the dataset...")
start_time = time.time()

# Load a random sample of the dataset
large_df = pd.read_csv("995K_FakeNewsCorpus_subset.csv", 
                        nrows=SAMPLE_SIZE,
                        skiprows=lambda i: i > 0 and np.random.random() > SAMPLE_SIZE/995000)

print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
print(f"Dataset shape: {large_df.shape}")

# Function to clean text (from Task 1)
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

# Initialize preprocessing utilities
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Complete preprocessing pipeline
def preprocess_text(text):
    # 1. Clean the text
    cleaned_text = clean_text(text)
    
    # 2. Tokenize
    tokens = word_tokenize(cleaned_text)
    
    # 3. Remove stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # 4. Apply stemming
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    
    # 5. Join tokens back into text
    processed_text = ' '.join(stemmed_tokens)
    
    return cleaned_text, ' '.join(tokens), ' '.join(filtered_tokens), processed_text

# Apply preprocessing to the dataset
print("Applying preprocessing pipeline...")
start_time = time.time()

# Create a tqdm progress bar
tqdm.pandas(desc="Processing articles")

# Apply preprocessing and create new columns
results = large_df['content'].progress_apply(lambda x: preprocess_text(x) if isinstance(x, str) else ("", "", "", ""))
large_df['cleaned_content'], large_df['tokenized_content'], large_df['no_stopwords_content'], large_df['stemmed_content'] = zip(*results)

print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

# Display info about the preprocessed data
print("\nSample of preprocessed content (first article):")
if len(large_df) > 0:
    print("\nOriginal text (first 100 chars):", large_df['content'].iloc[0][:100] + "...")
    print("\nCleaned text (first 100 chars):", large_df['cleaned_content'].iloc[0][:100] + "...")
    print("\nTokenized text (first 100 chars):", large_df['tokenized_content'].iloc[0][:100] + "...")
    print("\nWithout stopwords (first 100 chars):", large_df['no_stopwords_content'].iloc[0][:100] + "...")
    print("\nStemmed text (first 100 chars):", large_df['stemmed_content'].iloc[0][:100] + "...")

# Calculate vocabulary sizes and reduction rates
def get_vocab_sizes(df):
    # Create lists of all tokens
    all_tokens = ' '.join(df['tokenized_content'].tolist()).split()
    all_no_stopwords = ' '.join(df['no_stopwords_content'].tolist()).split()
    all_stemmed = ' '.join(df['stemmed_content'].tolist()).split()
    
    # Get unique tokens
    vocab_original = len(set(all_tokens))
    vocab_no_stopwords = len(set(all_no_stopwords))
    vocab_stemmed = len(set(all_stemmed))
    
    # Calculate reduction rates
    stopword_reduction = ((vocab_original - vocab_no_stopwords) / vocab_original) * 100
    stemming_reduction = ((vocab_no_stopwords - vocab_stemmed) / vocab_no_stopwords) * 100
    
    return {
        'vocab_original': vocab_original,
        'vocab_no_stopwords': vocab_no_stopwords,
        'vocab_stemmed': vocab_stemmed,
        'stopword_reduction': stopword_reduction,
        'stemming_reduction': stemming_reduction
    }

# Display vocabulary statistics
vocab_stats = get_vocab_sizes(large_df)
print("\nVocabulary Statistics:")
print(f"Original vocabulary size: {vocab_stats['vocab_original']}")
print(f"Vocabulary after stopword removal: {vocab_stats['vocab_no_stopwords']}")
print(f"Vocabulary after stemming: {vocab_stats['vocab_stemmed']}")
print(f"Reduction rate after stopword removal: {vocab_stats['stopword_reduction']:.2f}%")
print(f"Reduction rate after stemming: {vocab_stats['stemming_reduction']:.2f}%")

# Save the preprocessed data
output_file = 'preprocessed_fakenews_data.csv'
print(f"\nSaving preprocessed data to {output_file}...")
large_df.to_csv(output_file, index=False)
print("Preprocessing pipeline complete and data saved.")