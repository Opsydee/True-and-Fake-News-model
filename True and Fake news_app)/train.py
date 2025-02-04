# import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay
import joblib
import pickle

# load data
data = pd.read_csv('data/ftnews.csv')
print(data.head())

#seperate features and labels
x = data['text']
y = data['class']

# split data into training and test sets
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=4)

#convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df =0.7)
x_train_v =vectorizer.fit_transform(x_train)
x_test_v = vectorizer.transform(x_test)

#Initialize and train the model
mnb = MultinomialNB()
mnb.fit(x_train_v,y_train)
mnb.score(x_train_v, y_train)

#predict
y_pred = mnb.predict(x_test_v)
print(y_pred)

# model evaluation
a = accuracy_score(y_pred,y_test)
print(f'Model accuracy = {a}')
confusion_matrix(y_pred,y_test)
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred), display_labels = ['Fake', 'True']).plot()
import joblib

#save the trained model and vectorizer
joblib.dump(mnb, filename='model/model.pkl')
joblib.dump(vectorizer,'model/vectorizer.pkl')
                                