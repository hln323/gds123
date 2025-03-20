import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("Loading the split datasets...")
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('validation_data.csv')
test_df = pd.read_csv('test_data.csv')

# Display original distribution of types
print("\nOriginal distribution of news types:")
print(train_df['type'].value_counts())

# Binary classification already completed in Part 1
# Just display the distributions
print("\nBinary class distribution (Training set):")
print(train_df['binary_type'].value_counts())
print("\nBinary class distribution (Validation set):")
print(val_df['binary_type'].value_counts())
print("\nBinary class distribution (Test set):")
print(test_df['binary_type'].value_counts())

# Discussion of the binary grouping
print("\nDiscussion of the binary grouping:")
print("- We've categorized only 'reliable' articles as reliable.")
print("- All other categories including 'political', 'bias', and others are labeled as fake.")
print("- This is a simplification that might not perfectly reflect the nuanced nature of news reliability.")
print("- Categories like 'political' might contain both reliable and unreliable content.")
print("- This simplification could introduce bias in our model's predictions.")

# Task 1: Create simple logistic regression with the 10,000 most frequent words
print("\nTask 1: Training a simple logistic regression classifier...")

# Create feature matrix with the 10,000 most frequent words
count_vectorizer = CountVectorizer(max_features=10000)
X_train_count = count_vectorizer.fit_transform(train_df['cleaned_content'])
X_val_count = count_vectorizer.transform(val_df['cleaned_content'])
X_test_count = count_vectorizer.transform(test_df['cleaned_content'])

# Get target labels
y_train = train_df['binary_type']
y_val = val_df['binary_type']
y_test = test_df['binary_type']

# Calculate class weights to handle imbalance
class_weights = {
    'fake': 1.0,
    'reliable': len(train_df[train_df['binary_type'] == 'fake']) / len(train_df[train_df['binary_type'] == 'reliable'])
}

# Train logistic regression model with class weights
baseline_model = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    class_weight=class_weights
)
baseline_model.fit(X_train_count, y_train)

# Evaluate on validation set
baseline_val_pred = baseline_model.predict(X_val_count)
baseline_val_f1 = f1_score(y_val, baseline_val_pred, pos_label='fake')
print(f"\nBaseline model performance on validation set:")
print(f"F1 Score: {baseline_val_f1:.4f}")
print("\nDetailed classification report:")
print(classification_report(y_val, baseline_val_pred))

# Create confusion matrix
cm_baseline = confusion_matrix(y_val, baseline_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['reliable', 'fake'], 
            yticklabels=['reliable', 'fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Baseline Model')
plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png')
plt.close()

# Task 2: Add metadata features (domain)
print("\nTask 2: Adding metadata features...")

if 'domain' in train_df.columns:
    # One-hot encode domains
    domain_encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
    train_domains = domain_encoder.fit_transform(train_df[['domain']])
    val_domains = domain_encoder.transform(val_df[['domain']])
    
    # Combine text features with domain features
    X_train_with_meta = hstack([X_train_count, train_domains])
    X_val_with_meta = hstack([X_val_count, val_domains])
    
    # Train model with metadata
    meta_model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight=class_weights
    )
    meta_model.fit(X_train_with_meta, y_train)
    
    # Evaluate
    meta_val_pred = meta_model.predict(X_val_with_meta)
    meta_val_f1 = f1_score(y_val, meta_val_pred, pos_label='fake')
    print(f"\nModel with metadata performance on validation set:")
    print(f"F1 Score: {meta_val_f1:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(y_val, meta_val_pred))
    
    # Create confusion matrix
    cm_meta = confusion_matrix(y_val, meta_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_meta, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['reliable', 'fake'], 
                yticklabels=['reliable', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Model with Metadata')
    plt.tight_layout()
    plt.savefig('metadata_confusion_matrix.png')
    plt.close()
else:
    print("Domain column not found in the dataset. Skipping metadata enhancement.")
    meta_val_f1 = baseline_val_f1

# Task 3: Add BBC data
print("\nTask 3: Adding BBC data from Exercise 2...")

try:
    # Load BBC data
    with open('bbc_articles_content.json', 'r', encoding='utf-8') as f:
        bbc_data = json.load(f)
    
    # Convert to DataFrame
    bbc_df = pd.DataFrame(bbc_data)
    
    # Keep only articles with content
    bbc_df = bbc_df[bbc_df['text'].str.len() > 0].copy()
    
    # Clean the text using the same function from Part 1
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs
        text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
        
        # Replace emails
        text = re.sub(r'\S+@\S+', '<EMAIL>', text)
        
        # Replace dates
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '<DATE>', text)
        text = re.sub(r'\d{1,2}-\d{1,2}-\d{2,4}', '<DATE>', text)
        
        # Replace numbers
        text = re.sub(r'\b\d+\b', '<NUM>', text)
        
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    # Clean BBC content
    bbc_df['cleaned_content'] = bbc_df['text'].apply(clean_text)
    
    # Add 'type' and 'binary_type' columns
    bbc_df['type'] = 'reliable'
    bbc_df['binary_type'] = 'reliable'
    
    # Select relevant columns for combining with training data
    bbc_subset = bbc_df[['cleaned_content', 'type', 'binary_type']].copy()
    
    # Combine BBC data with training data
    combined_train_df = pd.concat([train_df, bbc_subset], ignore_index=True)
    
    print(f"Added {len(bbc_subset)} BBC articles to the training data")
    print(f"New training data distribution:")
    print(combined_train_df['binary_type'].value_counts())
    
    # Create features for combined dataset
    X_combined_train = count_vectorizer.transform(combined_train_df['cleaned_content'])
    y_combined_train = combined_train_df['binary_type']
    
    # Recalculate class weights
    combined_class_weights = {
        'fake': 1.0,
        'reliable': len(combined_train_df[combined_train_df['binary_type'] == 'fake']) / 
                    len(combined_train_df[combined_train_df['binary_type'] == 'reliable'])
    }
    
    # Train model with combined data
    combined_model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight=combined_class_weights
    )
    combined_model.fit(X_combined_train, y_combined_train)
    
    # Evaluate on validation set
    combined_val_pred = combined_model.predict(X_val_count)
    combined_val_f1 = f1_score(y_val, combined_val_pred, pos_label='fake')
    print(f"\nModel with BBC data performance on validation set:")
    print(f"F1 Score: {combined_val_f1:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(y_val, combined_val_pred))
    
    # Create confusion matrix
    cm_combined = confusion_matrix(y_val, combined_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['reliable', 'fake'], 
                yticklabels=['reliable', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Model with BBC Data')
    plt.tight_layout()
    plt.savefig('bbc_confusion_matrix.png')
    plt.close()
    
except Exception as e:
    print(f"Could not load BBC data: {e}")
    print("Continuing without BBC data...")
    combined_val_f1 = baseline_val_f1

# Compare all models
print("\nComparison of model performance:")
models = ['Baseline', 'With Metadata', 'With BBC Data']
f1_scores = [baseline_val_f1, meta_val_f1, combined_val_f1]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=f1_scores)
plt.title('F1 Score Comparison')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('model_comparison.png')

# Final discussion
print("\nDiscussion of results:")
print(f"1. Our baseline logistic regression model achieved an F1 score of {baseline_val_f1:.4f}.")

if 'domain' in train_df.columns:
    improvement = "improved" if meta_val_f1 > baseline_val_f1 else "decreased"
    print(f"2. Adding domain information {improvement} performance to an F1 score of {meta_val_f1:.4f}.")
    print(f"   This suggests that domains are {'helpful' if meta_val_f1 > baseline_val_f1 else 'not helpful'} for classification.")

try:
    improvement = "improved" if combined_val_f1 > baseline_val_f1 else "decreased"
    print(f"3. Adding BBC articles as reliable sources {improvement} performance to an F1 score of {combined_val_f1:.4f}.")
    print("   This aligns with our hypothesis from Exercise 2 that adding reliable sources would help balance the dataset.")
except:
    pass

print("\nFor the Advanced Model in Part 3, we will continue using only the content field without metadata")
print("as the assignment specifies to limit ourselves to main-text data only for Part 3 and Part 4.")
