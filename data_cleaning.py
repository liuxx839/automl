# data_cleaning.py
import pandas as pd
import numpy as np

def remove_na_rows(df, columns):
    return df.dropna(subset=columns)

def impute_mean(df, columns):
    for col in columns:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def impute_median(df, columns):
    for col in columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df

DATA_CLEANING_METHODS = {
    'Remove NA rows': remove_na_rows,
    'Impute mean': impute_mean,
    'Impute median': impute_median
}