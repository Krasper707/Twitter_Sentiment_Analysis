# Twitter_Sentiment_Analysis

## Overview

This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) techniques and machine learning models. The goal is to classify tweets into sentiment categories based on textual content.

## Features

- Uses NLTK for text preprocessing

- Implements TF-IDF Vectorization for feature extraction

- Trains models using Multinomial Naive Bayes (MultinomialNB) and Random Forest Classifier (RFC)

- Evaluates models using accuracy score and classification report

## Dataset

The dataset consists of tweets labeled with sentiment categories:

- 0: Negative

- 1: Neutral

- 2: Positive

- 3: Mixed

## Model Training & Evaluation

The project utilizes TF-IDF Vectorizer to convert text into numerical features and trains two models:

- Multinomial Naive Bayes (MultinomialNB)

- Random Forest Classifier (RFC)

The Random Forest Classifier achieved the following performance on a test set of 1000 tweets:
```
Accuracy: 0.964

              precision    recall  f1-score   support

           0       0.99      0.94      0.97       172
           1       0.96      0.97      0.96       266
           2       0.95      0.97      0.96       285
           3       0.96      0.97      0.97       277

    accuracy                           0.96      1000
   macro avg       0.97      0.96      0.96      1000
weighted avg       0.96      0.96      0.96      1000
```

## Future Enhancements

- Experiment with deep learning models (e.g., LSTMs, Transformers)

- Optimize hyperparameters for improved accuracy

- Expand dataset for better generalization

### License

This project is licensed under the MIT License.

