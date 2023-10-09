# Sentiment-Analysis-API

## Description
A Sentiment Analysis model is built using HuggingFace and Pytorch. The model is deployed using FastAPI and Swagger Documentation is implemented to test the API endpoint. 

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
