
CATEGORIES = list(train.columns[2:8])
df_comb = train.groupby(CATEGORIES)                    .size()                    .sort_values(ascending=False)                    .reset_index()                    .rename(columns={0: 'count'})

df_comb['label'] ='nontoxic'

for i in range(len(df_comb)):
    label_index = df_comb.iloc[i,0:6].values.astype(bool)
    label = ', '.join(list(compress(CATEGORIES, label_index)))
    if label:
        df_comb.loc[i, 'label'] = label

df_comb[(df_comb['count']>20) & (df_comb['count']<100000)].plot.bar(x='label', y='count', figsize=(17,8))
plt.yscale('log')
plt.xticks(size=15)

sns.heatmap(train.iloc[:,2:8].corr(), annot=True)


word_counter = {}

for categ in CATEGORIES:
    d = Counter()
    train[train[categ] == 1]['comment_text'].apply(lambda t: d.update(lemmatize(clean_text(t).split())))
    word_counter[categ] = pd.DataFrame.from_dict(d, orient='index')                                        .rename(columns={0: 'count'})                                        .sort_values('count', ascending=False)

def angry_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(%d, 100%%, 50%%)" % ((random.randint(-40, 40)+360)%360)

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

    plt.show()

