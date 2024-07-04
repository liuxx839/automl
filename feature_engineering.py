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
        # 对数变换前先将非正值替换为一个很小的正数
        min_positive = df[df[col] > 0][col].min()
        df[col] = df[col].replace({0: min_positive/2})
        df[f'{col}_log'] = np.log1p(df[col])
        
        # 处理无穷大和无穷小的情况
        df[f'{col}_log'] = df[f'{col}_log'].replace([np.inf, -np.inf], np.nan)
        
        # 用该列的均值填充 NaN 值
        df[f'{col}_log'] = df[f'{col}_log'].fillna(df[f'{col}_log'].mean())
    
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
        if output_col in columns:
            columns.remove(output_col)  # 暂时移除输出列
        
        if method == 'Binarize':
            for col in columns:
                threshold = methods['Binarize_threshold'].get(col, 0.5)  # 获取每列的阈值
                processed_df = FEATURE_ENGINEERING_METHODS[method](processed_df, col, threshold)
        elif method == 'One-Hot Encode':
            processed_df = FEATURE_ENGINEERING_METHODS[method](processed_df, columns)
            new_input_cols = [col for col in processed_df.columns if col != output_col and col not in columns]
            new_input_cols.extend([col for col in processed_df.columns if col.startswith(tuple(columns))])
        else:
            processed_df = FEATURE_ENGINEERING_METHODS[method](processed_df, columns)
            if method == 'Log Transform':
                new_input_cols.extend([f'{col}_log' for col in columns])
        
        if output_col in methods[method]:
            # 对输出列应用相同的转换
            if method != 'One-Hot Encode':  # One-Hot编码不适用于输出列
                processed_df = FEATURE_ENGINEERING_METHODS[method](processed_df, [output_col])
                if method == 'Log Transform':
                    output_col = f'{output_col}_log'
    
    return processed_df, new_input_cols, output_col
