# Sentiment-Analysis-API

## Description
A Sentiment Analysis model is built using HuggingFace and Pytorch. The model is deployed using FastAPI and Swagger Documentation is implemented to test the API endpoint. 

## Models Trained and Tested
#### Twitter-roberta-base-sentiment fine tuned on airline_sentiment_analysis dataset
- Accuracy = 0.95
- Precision = 0.95
- Recall = 0.95
- F1 Score = 0.95
#### Spacy Vectors with Multilayer Perceptron Classifier
- Accuracy = 0.88
- Precision = 0.88
- Recall = 0.88
- F1 Score = 0.88
#### TF-IDF Vectors with Multinomial Naive Bayes Classifier
- Accuracy = 0.85
- Precision = 0.86
- Recall = 0.85
- F1 Score = 0.85
#### Spacy Vectors with Random Forest Classifier
- Accuracy = 0.79
- Precision = 0.79
- Recall = 0.79
- F1 Score = 0.79
#### Spacy Vectors with KNN Classifier
- Accuracy = 0.75
- Precision = 0.75
- Recall = 0.75
- F1 Score = 0.75

## Inference
Fine tuning of the “twitter-roberta-base-sentiment-latest” model from HuggingFace shows the highest metrics with weighted and macro averages of F1 Score at 95%. 

## Installation
Clone git repository
```

Install dependencies
```
pip install -r requirements.txt
```

## Usage 
To Train Model
```
cd ml
python main.py
```

To Run Server, copy saved model to app folder and run,
```
uvicorn server:app --reload
```