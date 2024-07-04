import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from econml.dml import LinearDML
from lightgbm import LGBMRegressor
from scipy import stats
import time

def run_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X)

def run_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model.predict(X)

# ... 其他模型函数 ...

def run_causal_inference(X, y):
    df = pd.concat([X, y], axis=1)
    Y_column = y.name
    S_columns = X.columns.tolist()
    
    results, effects_dict = mean_confidence_interval(df, S_columns, Y_column)
    
    # 为了符合当前格式，我们返回一个预测结果
    # 这里我们使用第一个处理变量的效应作为示例
    first_effect = effects_dict[S_columns[0]]
    return pd.Series(first_effect, index=X.index)

def mean_confidence_interval(df, S_columns, Y_column, confidence_level=0.95):
    start_time = time.time()
    results = []
    effects_dict = {}
    for i, T_column in enumerate(S_columns):
        print(f"\n{'='*40}")
        print(f"Processing column: {T_column}")
        print(f"{'='*40}")
        T = df[T_column]
        X_columns = S_columns[:i] + S_columns[i+1:]
        print(X_columns)
        X = df[X_columns]
        Y = df[Y_column]
        est = LinearDML(model_y=LGBMRegressor(verbose=-1),
                        model_t=LGBMRegressor(verbose=-1),
                        random_state=2321, cv=5, discrete_treatment=False)
        est.fit(Y, T, X=X)
        effect = est.effect(X)
        effects_dict[T_column] = effect
        effect_mean = np.mean(effect)
        effect_se = np.std(effect, ddof=1) / np.sqrt(len(effect))
        alpha = 1 - confidence_level
        degrees_freedom = len(effect) - 1
        t_critical = stats.t.ppf(1 - alpha / 2, degrees_freedom)
        margin_of_error = t_critical * effect_se
        lower_bound = effect_mean - margin_of_error
        upper_bound = effect_mean + margin_of_error
        effect_mean = (np.exp(effect_mean) - 1)
        lower_bound = (np.exp(lower_bound) - 1)
        upper_bound = (np.exp(upper_bound) - 1)
        results.append({
            "T_column": T_column,
            "Effect Mean": effect_mean,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound
        })
        print(f"Effect Mean for {T_column}: {effect_mean}")
        print(f"Confidence Interval for {T_column}: [{lower_bound}, {upper_bound}]")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    results_df = pd.DataFrame(results)
    return results_df, effects_dict

MODEL_LIST = {
    'Linear Regression': run_linear_regression,
    'Logistic Regression': run_logistic_regression,
    # ... 其他模型 ...
    'Causal Inference': run_causal_inference
}
