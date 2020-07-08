from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop = stopwords.words('english')

from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
from joblib import dump, load
#from scipy.sparse import csr_matrix, hstack


def make_train_test(df_train):
    xtrain, xvalid, ytrain, yvalid = train_test_split(df_train.comment_text.values, df_train.toxic.values, 
    stratify=df_train.toxic.values, random_state=42, test_size=0.2, shuffle=True)
    return xtrain, xvalid, ytrain, yvalid
    

def apply_count_vectorizer(xtrain, xvalid, output_path=None):
    count_vectorizer = CountVectorizer(stop_words='english', max_df=.5, min_df=3)
    count_train = count_vectorizer.fit_transform(xtrain)
    count_valid = count_vectorizer.transform(xvalid)
    if output_path:
        dump(count_vectorizer, output_path + '/count_vectorizer.joblib')
    return count_train, count_valid

def run_naive_bayes_classifier(count_train, ytrain, output_path=None):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(count_train, ytrain)
    if output_path:
        dump(nb_classifier, output_path + '/nb_classifier.joblib')
    return nb_classifier
     
def apply_tfidf_vectorizer(xtrain, xvalid, output_path=None):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    count_train_idf = tfidf_vectorizer.fit_transform(xtrain)
    count_valid_idf = tfidf_vectorizer.transform(xvalid)
    if output_path:
        dump(tfidf_vectorizer, output_path + '/tfidf_vectorizer.joblib')
    return count_train_idf, count_valid_idf
    
'''
class Transformer:
    def fit(self):
        pass

    def transform(self):
        pass

    def fit_and_transform(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class Model:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self.load()
        
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def save(self):
        pass

    def load(self):
        pass
'''