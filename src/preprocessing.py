import pandas as pd


def clean_data(df):
    """Clean and preprocess the taxpayer data."""
    df = df.copy()
    df["income_log"] = df["declared_income"].apply(lambda x: 0 if x <= 0 else x)
    return df
