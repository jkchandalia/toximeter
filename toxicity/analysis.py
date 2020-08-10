#Quantify how much misspellings and long tail might be affecting results
def explore_comment_length(x, y):
    texts=x.astype(str)
    tokenizer=fast_tokenizer
    chunk_size=256
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    lens = []
    for j in range(len(all_ids)):
        lens.append(len(all_ids[j]))

    plt.hist(lens, 50)
    plt.yscale('log')

    long_index = (np.array(lens)>500)
    long_index = (np.array(lens)>500)
    print('Number of comments: ' + str(len(long_index)))
    print('Number of toxic comments: ' + str(sum(y)))
    print('Number of comments longer than 500 tokens: ' + str(sum(long_index)))
    print('Number of toxic comments longer than 500 tokens: ' + str(sum(y[long_index])))

encoded = tokenizer.encode_batch(['man walks down the street happily don''t you think @fire aslkfd291o'])

print(encoded[0].ids)
for id_item in encoded[0].ids:
    print(tokenizer.id_to_token(id_item))

#Testcase
fast_tokenizer.token_to_id('[UNK]')
print(fast_tokenizer.token_to_id('Man'))
print(fast_tokenizer.token_to_id('man'))
fast_tokenizer.id_to_token(28995)