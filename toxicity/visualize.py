from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict as ddict, Counter
from itertools import compress

from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from toxicity.constants import *
from toxicity.text_preprocessing import *
from nltk.stem.wordnet import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
import random

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop = stopwords.words('english')

def create_toxicity_barplot(df_train, output_path=None):
    df_comb = df_train.groupby(CATEGORIES).size().sort_values(ascending=False).reset_index().rename(columns={0: 'count'})
    df_comb['label'] ='nontoxic'

    for i in range(len(df_comb)):
        label_index = df_comb.loc[i, CATEGORIES].values.astype(bool)
        label = ', '.join(list(compress(CATEGORIES, label_index)))
        if label:
            df_comb.loc[i, 'label'] = label

    df_comb[(df_comb['count']>20) & (df_comb['count']<100000)].plot.bar(x='label', y='count', figsize=(17,8))
    plt.yscale('log')
    plt.xticks(size=15)
    if output_path:
        plt.savefig(output_path+'/toxicity_category_barplot.png')
    else:
        plt.show()


def create_corr_plot(df_train, output_path=None):
    sns.heatmap(df_train.loc[:, CATEGORIES].corr(), annot=True)
    if output_path:
        plt.savefig(output_path+'\heatmap.png')
    else:
        plt.show()


def create_word_counter(df_train, text_col = 'comment_text'):
    word_counter = {}

    for categ in CATEGORIES:
        d = Counter()
        df_train[df_train[categ] == 1][text_col].apply(lambda t: d.update(lemmatize(clean_text(t).split())))
        word_counter[categ] = pd.DataFrame.from_dict(d, orient='index').rename(columns={0: 'count'}).sort_values('count', ascending=False)

    return word_counter

def angry_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(%d, 100%%, 50%%)" % ((random.randint(-40, 40)+360)%360)

def create_wordcloud(word_counter, output_path=None):
    for w in word_counter:
        wc = word_counter[w]

        wordcloud = WordCloud(
            background_color='black',
            max_words=200,
            max_font_size=100, 
            random_state=461
            ).generate_from_frequencies(wc.to_dict()['count'])

        fig = plt.figure(figsize=(8, 8))
        plt.title(w.upper().replace('_', ' '), size=40)
        plt.imshow(wordcloud.recolor(color_func=angry_color_func, random_state=3),
            interpolation="bilinear")
        plt.axis('off')
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

