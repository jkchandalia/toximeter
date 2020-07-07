import pandas as pd
from os.path import dirname, exists, expanduser, isdir, join, splitext

def load(input_path, filter=True, return_X_y=False):
    """Load and return the Kaggle toxicity dataset (jigsaw-toxic-comment-train.csv) 
    for classification. 
    ==============   ==============
    Samples total            223549
    ==============   ==============
    Parameters
    ----------
    filter : bool, default=True
        If True, returns only the toxic label and the comment_text in the output
        and drops the labels corresponding to differenty types of toxicity.
    return_X_y : bool, default=False
        If True, returns ``(pd.DataFrame(data), pd.Series(target))`` instead of 
        a single dataframe that includes both comment_text and the toxic label.
    Returns
    -------
    data : pd.DataFrame
    (data, target) : tuple if ``return_X_y`` is True
    """
    df_data = pd.read_csv(input_path)
    if filter:
        df_data = df_data.loc[:,df_data.columns.isin(['toxic', 'comment_text'])]
        if return_X_y:
            return (df_data.drop('toxic', axis=1), df_data['toxic'])
    elif return_X_y:
        raise ValueError("return_X_y cannot be True if filter is False")
    return df_data


'''
class Loader:
    pass
'''