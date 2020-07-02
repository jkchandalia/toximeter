
# Include below feature engineering:
# 
# df_train['total_length'] = df_train['comment_text'].str.len()
# df_train['new_line'] = df_train['comment_text'].str.count('\n'* 1)
# df_train['new_small_space'] = df_train['comment_text'].str.count('\n'* 2)
# df_train['new_medium_space'] = df_train['comment_text'].str.count('\n'* 3)
# df_train['new_big_space'] = df_train['comment_text'].str.count('\n'* 4)
# 
# df_train['new_big_space'] = df_train['comment_text'].str.count('\n'* 4)
# df_train['uppercase_words'] = df_train['comment_text'].apply(lambda l: sum(map(str.isupper, list(l))))
# df_train['question_mark'] = df_train['comment_text'].str.count('\?')
# df_train['exclamation_mark'] = df_train['comment_text'].str.count('!')
# 
# FEATURES = ['total_length', 
#             'new_line', 
#             'new_small_space', 
#             'new_medium_space', 
#             'new_big_space', 
#             'uppercase_words',
#             'question_mark',
#             'exclamation_mark']
# COLUMNS += FEATURES
