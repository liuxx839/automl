import streamlit as st
import pandas as pd
import numpy as np
from model_list import MODEL_LIST
from feature_engineering import FEATURE_ENGINEERING_METHODS, apply_feature_engineering
from data_cleaning import DATA_CLEANING_METHODS
from config_utils import add_config_section

st.set_page_config(layout="wide")

# 初始化 session state
if 'page' not in st.session_state:
    st.session_state.page = 'Data Loading'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'input_cols' not in st.session_state:
    st.session_state.input_cols = None
if 'output_col' not in st.session_state:
    st.session_state.output_col = None
if 'cleaning_method' not in st.session_state:
    st.session_state.cleaning_method = None
if 'feature_engineering_methods' not in st.session_state:
    st.session_state.feature_engineering_methods = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'original_row_count' not in st.session_state:
    st.session_state.original_row_count = None
if 'processed_row_count' not in st.session_state:
    st.session_state.processed_row_count = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# 添加配置部分到侧边栏
add_config_section()

# 侧边栏导航
st.sidebar.title('导航')
page = st.sidebar.radio('选择页面', ['Data Loading', 'Data Preprocessing', 'Model Running'])

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        raise ValueError("不支持的文件格式。请上传 CSV 或 Excel 文件。")

def display_stats(df, columns):
    stats = df[columns].describe()
    na_count = df[columns].isna().sum()
    stats.loc['na_count'] = na_count
    return stats

def apply_data_cleaning(df, columns, method):
    return DATA_CLEANING_METHODS[method](df, columns)

def run_model(df, input_cols, output_col, groupby_cols, model_func):
    if groupby_cols:
        results = []
        for name, group in df.groupby(groupby_cols):
            X = group[input_cols]
            y = group[output_col]
            
            try:
                prediction = model_func(X, y)
            except Exception as e:
                st.error(f"模型运行错误：{str(e)}")
                return None

            result = pd.DataFrame({
                'Actual': y,
                'Predicted': prediction
            })
            for col, val in zip(groupby_cols, name):
                result[col] = val
            
            results.append(result)
        
        return pd.concat(results, ignore_index=True)
    else:
        X = df[input_cols]
        y = df[output_col]
        try:
            prediction = model_func(X, y)
        except Exception as e:
            st.error(f"模型运行错误：{str(e)}")
            return None
        
        return pd.DataFrame({'Actual': y, 'Predicted': prediction})

# Data Loading 页面
if page == 'Data Loading':
    st.title('数据加载和选择')
    
    uploaded_file = st.file_uploader("选择CSV或Excel文件", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            st.session_state.data = load_data(uploaded_file)
            st.session_state.original_row_count = len(st.session_state.data)
            st.write("数据预览：")
            st.dataframe(st.session_state.data.head())
            
            columns = st.session_state.data.columns.tolist()
            
            st.write("选择输入列：")
            col1, col2 = st.columns(2)
            selected_input_cols = []
            
            for i, col in enumerate(columns):
                if i % 2 == 0:
                    with col1:
                        if st.checkbox(col, value=col in st.session_state.input_cols):
                            selected_input_cols.append(col)
                else:
                    with col2:
                        if st.checkbox(col, value=col in st.session_state.input_cols):
                            selected_input_cols.append(col)
            
            st.session_state.input_cols = selected_input_cols
            
            st.session_state.output_col = st.selectbox("选择输出列", columns, index=columns.index(st.session_state.output_col) if st.session_state.output_col in columns else 0)
            
            if st.session_state.input_cols and st.session_state.output_col:
                stats = display_stats(st.session_state.data, st.session_state.input_cols + [st.session_state.output_col])
                st.write("选定列的统计信息：")
                st.dataframe(stats)
        
        except Exception as e:
            st.error(f"文件加载错误：{str(e)}")

# Data Preprocessing 页面
elif page == 'Data Preprocessing':
    st.title('数据预处理')
    
    if st.session_state.data is not None:
        # 数据清洗部分
        st.subheader("数据清洗")
        st.session_state.cleaning_method = st.selectbox("选择处理缺失值的方法", list(DATA_CLEANING_METHODS.keys()), index=list(DATA_CLEANING_METHODS.keys()).index(st.session_state.cleaning_method) if st.session_state.cleaning_method else 0)
        
        # 特征工程部分
        st.subheader("特征工程")
        apply_fe = st.checkbox("应用特征工程", value=bool(st.session_state.feature_engineering_methods))
        if apply_fe:
            feature_engineering_methods = {}
            binarize_thresholds = {}
            for method in FEATURE_ENGINEERING_METHODS.keys():
                all_cols = st.session_state.input_cols + [st.session_state.output_col]
                cols = st.multiselect(f"选择要应用 {method} 的列", all_cols, default=st.session_state.feature_engineering_methods.get(method, []))
                if cols:
                    feature_engineering_methods[method] = cols
                    if method == 'Binarize':
                        binarize_thresholds = {}
                        for col in cols:
                            threshold = st.number_input(f"输入 {col} 的二值化阈值", value=0.5)
                            binarize_thresholds[col] = threshold
            feature_engineering_methods['Binarize_threshold'] = binarize_thresholds
            st.session_state.feature_engineering_methods = feature_engineering_methods
        else:
            st.session_state.feature_engineering_methods = {}
        
        if st.button("应用预处理"):
            with st.spinner("正在进行数据清洗和特征工程..."):
                # 应用数据清洗
                cleaned_df = apply_data_cleaning(st.session_state.data, 
                                                 st.session_state.input_cols + [st.session_state.output_col], 
                                                 st.session_state.cleaning_method)
                
                # 应用特征工程
                if apply_fe:
                    processed_df, new_input_cols, new_output_col = apply_feature_engineering(cleaned_df, 
                                                                             st.session_state.input_cols, 
                                                                             st.session_state.output_col, 
                                                                             st.session_state.feature_engineering_methods)
                    st.session_state.output_col = new_output_col
                else:
                    processed_df, new_input_cols = cleaned_df, st.session_state.input_cols
                
                st.session_state.processed_data = processed_df
                st.session_state.input_cols = new_input_cols
                st.session_state.processed_row_count = len(processed_df)
            
            # ... (其余代码保持不变)
            
            st.success("预处理完成！")
            st.write("处理后的数据预览：")
            st.dataframe(st.session_state.processed_data.head())
            
            st.write("处理后的列统计信息：")
            processed_stats = display_stats(st.session_state.processed_data, st.session_state.input_cols + [st.session_state.output_col])
            st.dataframe(processed_stats)
            
            st.write(f"原始数据行数: {st.session_state.original_row_count}")
            st.write(f"处理后数据行数: {st.session_state.processed_row_count}")
    else:
        st.warning("请先在 'Data Loading' 页面加载数据。")

# Model Running 页面
elif page == 'Model Running':
    st.title('模型运行')
    
    if st.session_state.processed_data is not None:
        model_name = st.selectbox("选择模型", list(MODEL_LIST.keys()), 
                                  index=list(MODEL_LIST.keys()).index(st.session_state.model_name) 
                                  if st.session_state.model_name in MODEL_LIST.keys() 
                                  else 0)
        st.session_state.model_name = model_name
        
        groupby_cols = st.multiselect("选择分组列（可选）", 
                                      st.session_state.processed_data.columns.tolist(), 
                                      default=st.session_state.get('groupby_cols', []))
        st.session_state.groupby_cols = groupby_cols
        
        if st.button("运行模型"):
            with st.spinner("正在运行模型..."):
                model_func = MODEL_LIST[model_name]
                if model_name == 'Causal Inference':
                    final_result = model_func(st.session_state.processed_data[st.session_state.input_cols], 
                                              st.session_state.processed_data[st.session_state.output_col])
                else:
                    final_result = run_model(st.session_state.processed_data, 
                                             st.session_state.input_cols, 
                                             st.session_state.output_col, 
                                             groupby_cols, 
                                             model_func)
        
            if final_result is not None:
                st.write("模型结果：")
                st.dataframe(final_result)
        
                if model_name == 'Causal Inference':
                    st.write("因果效应估计：")
                    for col in final_result.columns:
                        if col != 'T_column':
                            st.write(f"{col}: {final_result[col].mean():.4f}")
                    
                    # 可视化因果效应
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    for _, row in final_result.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['T_column'], row['T_column']],
                            y=[row['Lower Bound'], row['Upper Bound']],
                            mode='lines',
                            name=row['T_column']
                        ))
                        fig.add_trace(go.Scatter(
                            x=[row['T_column']],
                            y=[row['Effect Mean']],
                            mode='markers',
                            name=row['T_column'] + ' mean'
                        ))
                    
                    fig.update_layout(title='因果效应估计及置信区间',
                                      xaxis_title='处理变量',
                                      yaxis_title='效应大小')
                    st.plotly_chart(fig)
                else:
                    if isinstance(final_result, pd.DataFrame) and 'Actual' in final_result.columns and 'Predicted' in final_result.columns:
                        mse = np.mean((final_result['Actual'] - final_result['Predicted'])**2)
                        st.write(f"总体均方误差 (MSE): {mse:.4f}")
                    else:
                        st.write("模型结果统计：")
                        st.write(final_result.describe())
        
                # 显示处理后的数据行数
                st.write(f"原始数据行数: {st.session_state.original_row_count}")
                st.write(f"处理后数据行数: {st.session_state.processed_row_count}")
            else:
                st.error("模型运行失败，没有返回结果。")
    else:
        st.warning("请先完成数据加载和预处理步骤。")
