def vectorize(vectorizer, stop_words, max_df, min_df):
    params = {
        'stop_words': stop_words,
        'max_df': max_df,
        'min_df': min_df
    }
    vectorizer.set_params(params)
    #count_train = count_vectorizer.fit_transform(xtrain)
    #count_valid = count_vectorizer.transform(xvalid)
    return vectorizer
