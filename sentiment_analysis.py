# NLP - Natural Language Processing

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.txt", sep = ";", names = ["text", "label"], nrows=5000)
print(data.head())
data["label"] = data["label"].replace(
    {"sadness": 1, "anger":1, "fear":1, "joy":0, "love":0, "surprise":0}
)
print(data.head())

x = data["text"]
y = data["label"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=1)

# NLTK - Natural Language Toolkit
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

wne = WordNetLemmatizer()

def transform(data):
    corpus = []
    for i in data:
        newi = re.sub("[^a-zA-z]"," ",i)
        newi = newi.lower()
        newi  = newi.split()
        list1 = [wne.lemmatize(word) for word in newi if word not in stopwords.words("english")]
        corpus.append(" ".join(list1))

    return corpus

XTrainCorpus = transform(x_train)
XTestCorpus = transform(x_test)

print(XTrainCorpus[0:5])
