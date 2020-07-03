import pandas as pd 

def create_features(df_train, text_col='comment_text'):
    # Include below feature engineering:
    df_train['total_length'] = df_train[text_col].str.len()
    df_train['new_line'] = df_train[text_col].str.count('\n'* 1)
    df_train['new_small_space'] = df_train[text_col].str.count('\n'* 2)
    df_train['new_medium_space'] = df_train[text_col].str.count('\n'* 3)
    df_train['new_big_space'] = df_train[text_col].str.count('\n'* 4)

    df_train['new_big_space'] = df_train[text_col].str.count('\n'* 4)
    df_train['uppercase_words'] = df_train[text_col].apply(lambda l: sum(map(str.isupper, list(l))))
    df_train['question_mark'] = df_train[text_col].str.count('\?')
    df_train['exclamation_mark'] = df_train[text_col].str.count('!')

    return df_train