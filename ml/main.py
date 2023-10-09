#### Fine Tunes Roberta Model to Airline Sentiment Analysis Dataset
import re
import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

BATCH_SIZE = 4
EPOCH = 4
WARMUP_STEPS = 500
LR = 2e-5
MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
if (torch.backends.mps.is_available()):
    print("Using Apple MPS for GPU acceleration")
    DEVICE = torch.device("mps")
elif (torch.cuda.is_available()):
    print("Using Cuda for GPU acceleration")
    DEVICE = torch.device("cuda")
else:
    print("No GPU found, using CPU")
    DEVICE = torch.device("cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def preprocessing(text):
        text = re.sub(r'[\n]', '', text)
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'RT[\s]', '', text)
        text = re.sub(r'https?:\/\/\S+','', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[ ]+',' ', text)
        return text.strip().lower()

class LoadData:

    def __init__(self, path):
        self.data = pd.read_csv(path)

    def load(self):
        self.data['text'] = self.data['text'].apply(preprocessing)
        self.data = self.data[self.data['text']!='']
        self.data = self.downsample(self.data)
        self.data['label'] = self.data['airline_sentiment'].map(
            {'negative':0,'positive':1}
            )
        X_train, X_test, y_train, y_test = train_test_split(
            self.data['text'],
            self.data['label'], 
            test_size=0.2, 
            random_state=24
            )
        return X_train, X_test, y_train, y_test

    def downsample(self, df):
        positive = df[df['airline_sentiment']=='positive']
        negative = df[df['airline_sentiment']=='negative']
        downsampled_negative = negative.sample(positive.shape[0])
        df_balanced = pd.concat([positive, downsampled_negative])
        return df_balanced

class TweetDataset(Dataset):
    # Converts to Pytorch Tensors
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class TrainModel:

    def __init__(self, model, tokenizer, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.tokenizer = tokenizer
        self.model.to(DEVICE)
        self.train_data = self.create_train_dataset()

    def create_train_dataset(self):
        train_encodings = self.tokenizer(
            self.X_train.tolist(),
            truncation=True, 
            padding=True
            )
        train_data = TweetDataset(train_encodings, self.y_train.tolist())
        return train_data

    def train(self):
        self.model.train()
        train_loader = DataLoader(self.train_data, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=LR)

        num_training_steps = EPOCH * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=WARMUP_STEPS, 
            num_training_steps=num_training_steps
            )
        progress_bar = tqdm(range(num_training_steps))

        for epoch in range(EPOCH):
            for batch in train_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        return self.model, self.tokenizer

class Inference:

    def __init__(self, model, tokenizer, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = self.create_test_dataset()

    def create_test_dataset(self):
        test_encodings = self.tokenizer(self.X_test.tolist(), truncation=True, padding=True)
        test_data = TweetDataset(test_encodings, self.y_test.tolist())
        return test_data

    def evaluate(self):
        self.model.eval()
        predictions = np.array([])
        labels = np.array([])
        test_loader = DataLoader(self.test_data, batch_size=BATCH_SIZE*2, shuffle=True)
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            temp_pred = torch.argmax(logits, dim=-1)
            temp_labels = batch["labels"]
            predictions = np.append(predictions, temp_pred.cpu())
            labels = np.append(labels, temp_labels.cpu())

        print(classification_report(predictions, labels))
        return self.model, self.tokenizer

if __name__ == "__main__":

    loader = LoadData("airline_sentiment_analysis.csv")
    X_train, X_test, y_train, y_test = loader.load()
    print("Data Loaded!")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model =  AutoModelForSequenceClassification.from_pretrained(
        MODEL, 
        num_labels=2, 
        ignore_mismatched_sizes=True
        )
    print("Fine Tuning Model.....")
    trainer = TrainModel(model, tokenizer, X_train, y_train)
    tuned_model, tuned_tokenizer = trainer.train()
    print("Model Trained!")

    evaluator = Inference(tuned_model, tuned_tokenizer, X_test, y_test)
    eval_model, eval_tokenizer = evaluator.evaluate()

    # Save Model
    save_folder = '../saved_models'
    if not os.path.exists(save_folder): 
        os.makedirs(save_folder)
        print(f"{save_folder} folder created")
    else:
        pass
    date_time = datetime.now()
    date = date_time.date()
    eval_model.save_pretrained(f"{save_folder}/roberta_senti_tuned_{date}")
    eval_tokenizer.save_pretrained(f"{save_folder}/roberta_senti_tuned_{date}")
    print("Model Saved!")