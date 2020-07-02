import numpy as np
import pandas as pd 
from collections import defaultdict as ddict, Counter
from itertools import compress
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import re

import random
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem.wordnet import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
pd.options.display.max_rows = 999




