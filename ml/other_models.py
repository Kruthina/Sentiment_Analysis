from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from datetime import datetime
import pickle
import spacy
import os

date_time = datetime.now()
date = date_time.date()

def tf_idf(X_train, X_test, y_train, y_test ):
    print("\n==================TF-IDF & MultiNB=======================\n")
    clf = Pipeline([
    ('tfidf', TfidfVectorizer()),    
    ('MultiNBclf', MultinomialNB())])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

def Vectorizer(df):
    vec_model = spacy.load("en_core_web_lg")
    df['vector'] = df['text'].apply(lambda text: vec_model(text).vector)
    return df

def MLP(X_train, X_test, y_train, y_test):
    print("\n===================MLPClassifier========================\n")

    clf = MLPClassifier(hidden_layer_sizes=(256,2), max_iter=1000)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test,y_pred))
    save_folder = '../saved_models'
    if not os.path.exists(save_folder): 
        os.makedirs(save_folder)
        print(f"{save_folder} folder created")
    else:
        pass
    
    pickle.dump(clf, open(f"{save_folder}/mlp_{date}.pkl", 'wb'))
    
def KNN(X_train, X_test, y_train, y_test):
    print("\n=========================KNN============================\n")

    clf = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test,y_pred))

    #pickle.dump(clf, open(f"Models/knn_{date}.pkl", 'wb'))

def RandForest(X_train, X_test, y_train, y_test):
    print("\n===============RandomForestClassifier===================\n")

    clf = RandomForestClassifier()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test,y_pred))
