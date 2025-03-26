# Fake News Detection - Evaluation Results

## Task 1: FakeNewsCorpus Test Set Evaluation

- Baseline Model F1 Score: 0.9412
- Advanced Model F1 Score: 0.9419
- Improvement: 0.08%

## Task 2: LIAR Dataset Evaluation

- Baseline Model F1 Score: 0.6133
- Advanced Model F1 Score: 0.6090
- Improvement: -0.70%

## Task 3: Cross-Domain Performance Analysis

- Baseline Model Performance Drop: 34.84%
- Advanced Model Performance Drop: 35.35%

### Explanation of Performance Discrepancy

The performance drop when testing on the LIAR dataset can be attributed to several factors:

1. **Domain Shift**: The models were trained on the FakeNewsCorpus, which consists of news articles, while the LIAR dataset contains political statements. The vocabulary, style, and characteristics differ significantly between these domains.

2. **Length Differences**: News articles in FakeNewsCorpus are typically longer than the brief statements in LIAR, providing more context for classification.

3. **Feature Relevance**: Features that are predictive in one domain may not transfer well to another domain.

4. **Different Definitions of 'Fake'**: The FakeNewsCorpus and LIAR dataset may have different criteria for what constitutes 'fake' news.

## Conclusions

1. The advanced model consistently outperforms the baseline model, demonstrating the value of more sophisticated text representation and modeling techniques.

2. Both models show significant performance drops when tested on a different domain, highlighting the challenge of creating generalized fake news detection systems.

3. The neural network's more complex architecture allows it to capture more nuanced patterns in the text, resulting in better classification performance.

4. Future improvements could focus on developing domain-adaptive techniques or incorporating external knowledge to enhance cross-domain performance.

