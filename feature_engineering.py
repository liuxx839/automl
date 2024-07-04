import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer, OneHotEncoder

def standardize(df, columns):
    
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def normalize(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def binarize(df, column, threshold):
    binarizer = Binarizer(threshold=threshold)
    df[column] = binarizer.fit_transform(df[[column]])
    return df

def one_hot_encode(df, columns):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names(columns))
    df = df.drop(columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def log_transform(df, columns):
    for col in columns:
        df[f'{col}_log'] = np.log1p(df[col])
    return df

FEATURE_ENGINEERING_METHODS = {
    'Standardize': standardize,
    'Normalize': normalize,
    'Binarize': binarize,
    'One-Hot Encode': one_hot_encode,
    'Log Transform': log_transform
}

def apply_feature_engineering(df, input_cols, output_col, methods):
    processed_df = df.copy()
    new_input_cols = input_cols.copy()
    
    for method, columns in methods.items():
        if method == 'Binarize':
            for col in columns:
                # 这里不再使用 st.number_input，而是使用一个固定的阈值或者从外部传入
                threshold = 0.5  # 可以改为从参数传入
                processed_df = FEATURE_ENGINEERING_METHODS[method](processed_df, col, threshold)
        elif method == 'One-Hot Encode':
            processed_df = FEATURE_ENGINEERING_METHODS[method](processed_df, columns)
            new_input_cols = [col for col in processed_df.columns if col != output_col]
        else:
            processed_df = FEATURE_ENGINEERING_METHODS[method](processed_df, columns)
            if method == 'Log Transform':
                new_input_cols.extend([f'{col}_log' for col in columns])
    
    return processed_df, new_input_cols