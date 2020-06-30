import pandas as pd

def load(input_path):
    df_data = pd.read_csv(input_path)
    df_data = df_data.loc[:,df_data.columns.isin(['toxic', 'comment_text'])]
    return df_data
