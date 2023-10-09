#### Model Experimentation
from sklearn.model_selection import train_test_split
import other_models
import numpy as np
import pandas as pd
import re
import time

def preprocessing(text):
    # text = re.sub(r':[\WA-Za-z0-9]+','', text)
    # text = re.sub(r';[\WA-Za-z0-9]+','', text)
    text = re.sub(r'XD','', text)
    text = re.sub(r'[\n]', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r'https?:\/\/\S+','', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[ ]+',' ', text)
    return text.strip().lower()

def downsample(df):
    positive = df[df['airline_sentiment']=='positive']
    negative = df[df['airline_sentiment']=='negative']
    downsampled_negative = negative.sample(positive.shape[0])
    df_balanced = pd.concat([positive, downsampled_negative])
    return df_balanced

if __name__ == "__main__":

    data = pd.read_csv("airline_sentiment_analysis.csv")
    data['text'] = data['text'].apply(preprocessing)
    data = data[data['text']!='']
    data = downsample(data)
    data['label'] = data['airline_sentiment'].map({'negative':0,'positive':1})
    
    # TF-IDF - Multinomial Naive Bayes
    # start_time = time.time()
    # X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=24)
    # models.tf_idf(X_train, X_test, y_train, y_test)
    # timer = time.time()-start_time
    # print(f"Time taken for Training = {timer}")

    # SpaCy - MLP
    start_time = time.time()
    vecs = other_models.Vectorizer(data)
    X_train, X_test, y_train, y_test = train_test_split(
        vecs['vector'].values,vecs['label'],
        test_size=0.2,
        random_state=24
        )
    X_train_2d = np.stack(X_train)
    X_test_2d = np.stack(X_test)

    other_models.MLP(X_train_2d, X_test_2d, y_train, y_test)

    timer = time.time()-start_time
    print(f"Time taken for training = {timer}")