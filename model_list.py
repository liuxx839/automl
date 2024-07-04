# model_list.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X)

def decision_tree(X, y):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    return model.predict(X)

# 添加新模型时，只需在此处添加新的函数
# def new_model(X, y):
#     ...

# 模型字典
MODEL_LIST = {
    "线性回归": linear_regression,
    "决策树回归": decision_tree,
    # "新模型名称": new_model,
}