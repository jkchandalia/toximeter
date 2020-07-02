import pandas as pd

class Loader:
    pass

def load(input_path):
    df_data = pd.read_csv(input_path)
    df_data = df_data.loc[:,df_data.columns.isin(['toxic', 'comment_text'])]
    return df_data

pre_path = 'data/'

train = pd.read_csv(pre_path + 'jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
#The following is a non-English dataset and won't be used presently
validation = pd.read_csv(pre_path + 'jigsaw-multilingual-toxic-comment-classification/validation.csv')
#The following is a non-English dataset and won't be used presently
test = pd.read_csv(pre_path + 'jigsaw-multilingual-toxic-comment-classification/test.csv')


#train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)
train.loc[:,train.columns.isin(['toxic','comment_text'])]
train_full = train.copy()
#train = train.loc[:10000,:]
train.comment_text[train.toxic==1][1:2].values
train.toxic.value_counts()

