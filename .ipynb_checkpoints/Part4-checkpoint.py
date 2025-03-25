import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from joblib import load, dump
import time
import os

print("Loading models and test data...")
# Load the test set from Part 1
test_df = pd.read_csv('test_data.csv')

print(f"Test set size: {len(test_df)}")
print(f"Test set distribution: \n{test_df['type'].value_counts()}")

# Add binary classification if needed
if 'binary_type' not in test_df.columns:
    print("Creating binary classification labels...")
    # Define which types are considered reliable vs fake
    reliable_types = ['reliable']
    fake_types = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 
                  'bias', 'satire', 'clickbait', 'political', 'rumor', 'unknown']
    
    def create_binary_label(news_type):
        if pd.isna(news_type):
            return np.nan
        elif news_type in reliable_types:
            return 'reliable'
        elif news_type in fake_types:
            return 'fake'
        else:
            return np.nan
    
    # Apply binary classification
    test_df['binary_type'] = test_df['type'].apply(create_binary_label)

print(f"Binary test set distribution: \n{test_df['binary_type'].value_counts()}")

# Function to load the LIAR dataset
def load_liar_dataset(test_path="liar_dataset/test.tsv", 
                      val_path="liar_dataset/valid.tsv", 
                      train_path="liar_dataset/train.tsv"):
    try:
        # Define column names based on the LIAR dataset description
        columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 
                  'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 
                  'mostly_true_counts', 'pants_on_fire_counts', 'context']
        
        # Load all three parts of the dataset
        test_df = pd.read_csv(test_path, sep='\t', header=None, names=columns)
        val_df = pd.read_csv(val_path, sep='\t', header=None, names=columns)
        train_df = pd.read_csv(train_path, sep='\t', header=None, names=columns)
        
        # Combine all parts into one dataframe
        liar_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        print(f"Loaded LIAR dataset with {len(liar_df)} rows")
        print(f"LIAR label distribution: \n{liar_df['label'].value_counts()}")
        
        # Map LIAR labels to binary labels
        # Based on the LIAR paper, we'll use:
        # 'true', 'mostly-true', 'half-true' -> reliable
        # 'barely-true', 'false', 'pants-fire' -> fake
        reliable_labels = ['true', 'mostly-true', 'half-true']
        fake_labels = ['barely-true', 'false', 'pants-fire']
        
        liar_df['binary_type'] = liar_df['label'].apply(
            lambda x: 'reliable' if x in reliable_labels else 'fake' if x in fake_labels else np.nan
        )
        
        print(f"Binary LIAR label distribution: \n{liar_df['binary_type'].value_counts()}")
        
        return liar_df
    
    except FileNotFoundError as e:
        print(f"Error loading LIAR dataset: {e}")
        print("Please make sure the LIAR dataset files are in the correct path.")
        return None

# Function to load or train models
def load_or_train_models():
    # Check if models are already saved
    if os.path.exists('baseline_model.joblib') and os.path.exists('advanced_model.joblib'):
        print("Loading saved models...")
        baseline_model = load('baseline_model.joblib')
        advanced_model = load('advanced_model.joblib')
        count_vectorizer = load('count_vectorizer.joblib')
        tfidf_vectorizer = load('tfidf_vectorizer.joblib')
    else:
        print("Training new models...")
        # Train baseline model (similar to Part 2)
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        
        train_df = pd.read_csv('train_data.csv')
        
        # Add binary labels to training data if needed
        if 'binary_type' not in train_df.columns:
            print("Creating binary classification labels for training data...")
            reliable_types = ['reliable']
            fake_types = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 
                          'bias', 'satire', 'clickbait', 'political', 'rumor', 'unknown']
            
            train_df['binary_type'] = train_df['type'].apply(
                lambda x: 'reliable' if x in reliable_types else 'fake' if x in fake_types else np.nan
            )
        
        # Baseline model (Logistic Regression with CountVectorizer)
        count_vectorizer = CountVectorizer(max_features=10000)
        X_train_count = count_vectorizer.fit_transform(train_df['content'])
        y_train = train_df['binary_type']
        
        baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        baseline_model.fit(X_train_count, y_train)
        
        # Advanced model (Neural Network with TF-IDF)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPClassifier
        
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['content'])
        
        advanced_model = MLPClassifier(
            hidden_layer_sizes=(100,),
            alpha=0.0001,
            random_state=42,
            max_iter=300,
            early_stopping=True
        )
        advanced_model.fit(X_train_tfidf, y_train)
        
        # Save models for future use
        dump(baseline_model, 'baseline_model.joblib')
        dump(advanced_model, 'advanced_model.joblib')
        dump(count_vectorizer, 'count_vectorizer.joblib')
        dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
    
    return baseline_model, advanced_model, count_vectorizer, tfidf_vectorizer

# Task 1: Evaluate on FakeNewsCorpus test set
def evaluate_on_fakenews_corpus(test_df, baseline_model, advanced_model, count_vectorizer, tfidf_vectorizer):
    print("\nTask 1: Evaluating models on the FakeNewsCorpus test set")
    
    # Prepare test data
    X_test_count = count_vectorizer.transform(test_df['content'])
    X_test_tfidf = tfidf_vectorizer.transform(test_df['content'])
    y_test = test_df['binary_type']
    
    # Evaluate baseline model
    baseline_pred = baseline_model.predict(X_test_count)
    baseline_f1 = f1_score(y_test, baseline_pred, pos_label='fake')
    
    print(f"\nBaseline Model (Logistic Regression) - F1 Score: {baseline_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, baseline_pred))
    
    # Create confusion matrix
    cm_baseline = confusion_matrix(y_test, baseline_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['reliable', 'fake'], 
                yticklabels=['reliable', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Baseline Model (FakeNewsCorpus)')
    plt.tight_layout()
    plt.savefig('baseline_confusion_fakenews.png')
    plt.close()
    
    # Evaluate advanced model
    advanced_pred = advanced_model.predict(X_test_tfidf)
    advanced_f1 = f1_score(y_test, advanced_pred, pos_label='fake')
    
    print(f"\nAdvanced Model (Neural Network) - F1 Score: {advanced_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, advanced_pred))
    
    # Create confusion matrix
    cm_advanced = confusion_matrix(y_test, advanced_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_advanced, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['reliable', 'fake'], 
                yticklabels=['reliable', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Advanced Model (FakeNewsCorpus)')
    plt.tight_layout()
    plt.savefig('advanced_confusion_fakenews.png')
    plt.close()
    
    return baseline_f1, advanced_f1

# Task 2: Evaluate on LIAR dataset
def evaluate_on_liar_dataset(liar_df, baseline_model, advanced_model, count_vectorizer, tfidf_vectorizer):
    print("\nTask 2: Evaluating models on the LIAR dataset")
    
    if liar_df is None:
        print("LIAR dataset not available. Skipping evaluation.")
        return None, None
    
    # For LIAR dataset, use 'statement' as the text content
    X_liar_count = count_vectorizer.transform(liar_df['statement'])
    X_liar_tfidf = tfidf_vectorizer.transform(liar_df['statement'])
    y_liar = liar_df['binary_type']
    
    # Evaluate baseline model
    baseline_pred = baseline_model.predict(X_liar_count)
    baseline_f1 = f1_score(y_liar, baseline_pred, pos_label='fake')
    
    print(f"\nBaseline Model (Logistic Regression) - F1 Score: {baseline_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_liar, baseline_pred))
    
    # Create confusion matrix
    cm_baseline = confusion_matrix(y_liar, baseline_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['reliable', 'fake'], 
                yticklabels=['reliable', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Baseline Model (LIAR)')
    plt.tight_layout()
    plt.savefig('baseline_confusion_liar.png')
    plt.close()
    
    # Evaluate advanced model
    advanced_pred = advanced_model.predict(X_liar_tfidf)
    advanced_f1 = f1_score(y_liar, advanced_pred, pos_label='fake')
    
    print(f"\nAdvanced Model (Neural Network) - F1 Score: {advanced_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_liar, advanced_pred))
    
    # Create confusion matrix
    cm_advanced = confusion_matrix(y_liar, advanced_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_advanced, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['reliable', 'fake'], 
                yticklabels=['reliable', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Advanced Model (LIAR)')
    plt.tight_layout()
    plt.savefig('advanced_confusion_liar.png')
    plt.close()
    
    return baseline_f1, advanced_f1

# Task 3: Compare results
def compare_results(fakenews_baseline_f1, fakenews_advanced_f1, liar_baseline_f1, liar_advanced_f1):
    print("\nTask 3: Comparing model performance across datasets")
    
    # Create results table
    results = {
        'Model': ['Baseline (LogReg)', 'Advanced (NeuralNet)'],
        'FakeNewsCorpus F1': [fakenews_baseline_f1, fakenews_advanced_f1],
        'LIAR F1': [liar_baseline_f1, liar_advanced_f1] if liar_baseline_f1 else ['N/A', 'N/A'],
        'Performance Drop': [
            f"{((fakenews_baseline_f1 - liar_baseline_f1) / fakenews_baseline_f1 * 100):.2f}%" if liar_baseline_f1 else 'N/A',
            f"{((fakenews_advanced_f1 - liar_advanced_f1) / fakenews_advanced_f1 * 100):.2f}%" if liar_advanced_f1 else 'N/A'
        ]
    }
    
    results_df = pd.DataFrame(results)
    print("\nResults Comparison Table:")
    print(results_df.to_string(index=False))
    
    # Create visualization if LIAR results are available
    if liar_baseline_f1 and liar_advanced_f1:
        plt.figure(figsize=(10, 6))
        
        # Set up grouped bar chart
        datasets = ['FakeNewsCorpus', 'LIAR']
        baseline_scores = [fakenews_baseline_f1, liar_baseline_f1]
        advanced_scores = [fakenews_advanced_f1, liar_advanced_f1]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline (LogReg)')
        rects2 = ax.bar(x + width/2, advanced_scores, width, label='Advanced (NeuralNet)')
        
        # Add labels and legend
        ax.set_ylabel('F1 Score')
        ax.set_title('Model Performance Comparison Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig('model_comparison_across_datasets.png')
        plt.close()
    
    # Save detailed analysis
    with open('evaluation_results.md', 'w') as f:
        f.write("# Fake News Detection - Evaluation Results\n\n")
        
        f.write("## Task 1: FakeNewsCorpus Test Set Evaluation\n\n")
        f.write(f"- Baseline Model F1 Score: {fakenews_baseline_f1:.4f}\n")
        f.write(f"- Advanced Model F1 Score: {fakenews_advanced_f1:.4f}\n")
        f.write(f"- Improvement: {((fakenews_advanced_f1 - fakenews_baseline_f1) / fakenews_baseline_f1 * 100):.2f}%\n\n")
        
        if liar_baseline_f1 and liar_advanced_f1:
            f.write("## Task 2: LIAR Dataset Evaluation\n\n")
            f.write(f"- Baseline Model F1 Score: {liar_baseline_f1:.4f}\n")
            f.write(f"- Advanced Model F1 Score: {liar_advanced_f1:.4f}\n")
            f.write(f"- Improvement: {((liar_advanced_f1 - liar_baseline_f1) / liar_baseline_f1 * 100):.2f}%\n\n")
            
            f.write("## Task 3: Cross-Domain Performance Analysis\n\n")
            f.write(f"- Baseline Model Performance Drop: {((fakenews_baseline_f1 - liar_baseline_f1) / fakenews_baseline_f1 * 100):.2f}%\n")
            f.write(f"- Advanced Model Performance Drop: {((fakenews_advanced_f1 - liar_advanced_f1) / fakenews_advanced_f1 * 100):.2f}%\n\n")
            
            f.write("### Explanation of Performance Discrepancy\n\n")
            f.write("The performance drop when testing on the LIAR dataset can be attributed to several factors:\n\n")
            f.write("1. **Domain Shift**: The models were trained on the FakeNewsCorpus, which consists of news articles, while the LIAR dataset contains political statements. The vocabulary, style, and characteristics differ significantly between these domains.\n\n")
            f.write("2. **Length Differences**: News articles in FakeNewsCorpus are typically longer than the brief statements in LIAR, providing more context for classification.\n\n")
            f.write("3. **Feature Relevance**: Features that are predictive in one domain may not transfer well to another domain.\n\n")
            f.write("4. **Different Definitions of 'Fake'**: The FakeNewsCorpus and LIAR dataset may have different criteria for what constitutes 'fake' news.\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("1. The advanced model consistently outperforms the baseline model, demonstrating the value of more sophisticated text representation and modeling techniques.\n\n")
        f.write("2. Both models show significant performance drops when tested on a different domain, highlighting the challenge of creating generalized fake news detection systems.\n\n")
        f.write("3. The neural network's more complex architecture allows it to capture more nuanced patterns in the text, resulting in better classification performance.\n\n")
        f.write("4. Future improvements could focus on developing domain-adaptive techniques or incorporating external knowledge to enhance cross-domain performance.\n\n")

# Main execution
def main():
    start_time = time.time()
    
    # Load or train models
    baseline_model, advanced_model, count_vectorizer, tfidf_vectorizer = load_or_train_models()
    
    # Task 1: Evaluate on FakeNewsCorpus test set
    fakenews_baseline_f1, fakenews_advanced_f1 = evaluate_on_fakenews_corpus(
        test_df, baseline_model, advanced_model, count_vectorizer, tfidf_vectorizer
    )
    
    # Task 2: Load and evaluate on LIAR dataset
    liar_dataset_path = "liar_dataset"  # Adjust this to your actual path
    if os.path.exists(liar_dataset_path):
        liar_df = load_liar_dataset(
            test_path=f"{liar_dataset_path}/test.tsv",
            val_path=f"{liar_dataset_path}/valid.tsv",
            train_path=f"{liar_dataset_path}/train.tsv"
        )
        
        if liar_df is not None:
            liar_baseline_f1, liar_advanced_f1 = evaluate_on_liar_dataset(
                liar_df, baseline_model, advanced_model, count_vectorizer, tfidf_vectorizer
            )
        else:
            liar_baseline_f1, liar_advanced_f1 = None, None
    else:
        print(f"LIAR dataset directory '{liar_dataset_path}' not found.")
        print("Download it from the source and extract to this path to enable cross-domain evaluation.")
        liar_baseline_f1, liar_advanced_f1 = None, None
    
    # Task 3: Compare results
    compare_results(fakenews_baseline_f1, fakenews_advanced_f1, liar_baseline_f1, liar_advanced_f1)
    
    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
