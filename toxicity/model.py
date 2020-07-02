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


def vectorize(vectorizer, params):
    vectorizer.set_params(params)
    #count_train = count_vectorizer.fit_transform(xtrain)
    #count_valid = count_vectorizer.transform(xvalid)
    return vectorizer

xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 
                                                  stratify=train.toxic.values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)



count_vectorizer = CountVectorizer(stop_words='english', max_df=.5, min_df=3)
count_train = count_vectorizer.fit_transform(xtrain)
count_valid = count_vectorizer.transform(xvalid)


nb_classifier = MultinomialNB()

nb_classifier.fit(count_train, ytrain)
pred = nb_classifier.predict(count_valid)
pred_proba = nb_classifier.predict_proba(count_valid)[:,1]



from joblib import dump, load
dump(nb_classifier, './../models/nb_classifier.joblib') 
dump(count_vectorizer, './../models/count_vectorizer.joblib') 

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
count_train_idf = tfidf_vectorizer.fit_transform(xtrain)
count_valid_idf = tfidf_vectorizer.transform(xvalid)


nb_classifier.fit(count_train_idf, ytrain)
pred = nb_classifier.predict(count_valid_idf)
pred_proba = nb_classifier.predict_proba(count_valid_idf)[:,1]

