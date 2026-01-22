import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import nltk
nltk.download ('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("spam.csv",encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
# print(df.head())
# print(df['label'].value_counts())

#Label Encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

#Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.strip()
    return text
df['text']=df['text'].apply(clean_text)
# print(df.head())

#Text Conversion
tfidf = TfidfVectorizer(
    stop_words = 'english',
    max_features = 5000,
    ngram_range = (1,2)
)
X = tfidf.fit_transform(df['text'])
y = df['label']

#Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.20, random_state = 42
)

#Naive Bayes
NB = MultinomialNB()
NB.fit(X_train,y_train)

pred_nb = NB.predict(X_test)
print("Naive Bayes\n",classification_report(y_test, pred_nb))
