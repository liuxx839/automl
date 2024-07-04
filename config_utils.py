# config_utils.py

import json
import streamlit as st

def save_configuration():
    config = {
        'input_cols': st.session_state.input_cols,
        'output_col': st.session_state.output_col,
        'cleaning_method': st.session_state.cleaning_method,
        'feature_engineering_methods': st.session_state.feature_engineering_methods,
        'model_name': st.session_state.get('model_name', ''),
        'groupby_cols': st.session_state.get('groupby_cols', [])
    }
    
    config_str = json.dumps(config, indent=2)
    st.download_button(
        label="下载配置文件",
        data=config_str,
        file_name="automl_config.json",
        mime="application/json"
    )

def load_configuration(config_file):
    config = json.load(config_file)
    st.session_state.input_cols = config.get('input_cols', [])
    st.session_state.output_col = config.get('output_col', None)
    st.session_state.cleaning_method = config.get('cleaning_method', None)
    st.session_state.feature_engineering_methods = config.get('feature_engineering_methods', {})
    st.session_state.model_name = config.get('model_name', None)
    st.session_state.groupby_cols = config.get('groupby_cols', [])

def add_config_section():
    st.sidebar.header("配置")
    config_file = st.sidebar.file_uploader("上传配置文件（可选）", type=["json"])
    if config_file is not None:
        try:
            load_configuration(config_file)
            st.sidebar.success("配置已加载")
        except json.JSONDecodeError:
            st.sidebar.error("无效的 JSON 文件")
        except KeyError as e:
            st.sidebar.error(f"配置文件缺少必要的键: {str(e)}")
        except Exception as e:
            st.sidebar.error(f"加载配置时出错: {str(e)}")
    
    if st.sidebar.button("保存当前配置"):
        save_configuration()
