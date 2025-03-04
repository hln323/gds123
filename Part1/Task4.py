import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

print("Loading preprocessed dataset...")
start_time = time.time()

# Load the preprocessed data
df = pd.read_csv('preprocessed_fakenews_data.csv')

print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
print(f"Dataset shape: {df.shape}")

# Display distribution of news types
print("\nDistribution of news types before splitting:")
type_counts = df['type'].value_counts()
print(type_counts)

# Remove rows with missing 'type' values if needed
if df['type'].isnull().sum() > 0:
    print(f"\nRemoving {df['type'].isnull().sum()} rows with missing 'type' values")
    df = df.dropna(subset=['type'])
    print(f"Dataset shape after removing missing types: {df.shape}")

# First split: training vs. the rest (80% / 20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['type'])

# Second split: validation vs. test (50% / 50% of the remaining 20%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['type'])

# Verify the split sizes
print("\nSplit sizes:")
print(f"Training set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

# Check distribution of types in each split
print("\nTraining set type distribution:")
print(train_df['type'].value_counts())

print("\nValidation set type distribution:")
print(val_df['type'].value_counts())

print("\nTest set type distribution:")
print(test_df['type'].value_counts())

# Save the splits to separate CSV files
print("\nSaving data splits...")
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('validation_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("Data splitting complete!")

# Summary
print("\nSummary:")
print("The dataset has been split into three sets with the following specifications:")
print("- Training set (80%): For training your baseline and advanced models")
print("- Validation set (10%): For model selection and hyperparameter tuning")
print("- Test set (10%): For final evaluation in Part 4")
print("\nThe splits maintain the same distribution of news types as the original dataset (stratified sampling).")
print("The preprocessed and split data has been saved to separate CSV files for future use.")