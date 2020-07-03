import pandas as pd

def load(input_path, filter=True):
    df_data = pd.read_csv(input_path)
    if filter:
        df_data = df_data.loc[:,df_data.columns.isin(['toxic', 'comment_text'])]
    return df_data

'''
class Loader:
    pass
'''