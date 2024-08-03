import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
import tempfile
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI

# Fill ".env" with OPENAI_API_KEY Before Proceeding 
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


lida = Manager(text_gen=llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo", use_cache=True)


def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def identify_columns(df):
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numeric_cols, categorical_cols
    except Exception as e:
        st.error(f"Error identifying columns: {e}")
        return [], []

def identify_primary_key_columns(df):
    try:
        primary_keys = [col for col in df.columns if df[col].is_unique]
        return primary_keys
    except Exception as e:
        st.error(f"Error identifying primary key columns: {e}")
        return []

def identify_time_stamp_column(df):
    try:
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                return col
            except (ValueError, TypeError):
                continue
        return None
    except Exception as e:
        st.error(f"Error identifying time stamp column: {e}")
        return None

def numerical_analysis(df, numeric_cols):
    analysis = {}
    try:
        for col in numeric_cols:
            col_data = df[col]
            analysis[col] = {
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Mode': col_data.mode()[0] if not col_data.mode().empty else None,
                'Standard Deviation': col_data.std()
            }
    except Exception as e:
        st.error(f"Error performing numerical analysis: {e}")
    return analysis

def calculate_correlation(df, numeric_cols):
    try:
        return df[numeric_cols].corr()
    except Exception as e:
        st.error(f"Error calculating correlation: {e}")
        return pd.DataFrame()

def plot_bar_charts(df, categorical_cols):
    try:
        num_charts = len(categorical_cols)
        if (num_charts == 0):
            return None
        
        # Using Plotly for better interactivity
        rows = (num_charts // 2) + (num_charts % 2)
        cols = 2
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=categorical_cols)

        for idx, col in enumerate(categorical_cols):
            value_counts = df[col].value_counts().head(10)
            row = (idx // 2) + 1
            col_num = (idx % 2) + 1
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=col),
                row=row, col=col_num
            )
            fig.update_xaxes(title_text=col, row=row, col=col_num)
            fig.update_yaxes(title_text='Count', row=row, col=col_num)
        
        fig.update_layout(title_text="Bar Charts of Categorical Columns", showlegend=False, height=1000)
        return fig
    except Exception as e:
        st.error(f"Error creating bar charts: {e}")
        return None

def plot_time_series(df, time_stamp_col):
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        df.set_index(time_stamp_col, inplace=True)
        df.sort_index(inplace=True)
        fig = px.line(df, x=df.index, y=numeric_cols, title='Time Series Plot')
        fig.update_layout(height=500)
        return fig
    except Exception as e:
        st.error(f"Error plotting time series for {time_stamp_col}: {e}")
        return None

def decompose_time_series(df, time_stamp_col):
    try:
        df = df.copy()
        df.set_index(time_stamp_col, inplace=True)
        df.sort_index(inplace=True)
        if df.select_dtypes(include=['number']).shape[1] > 0:
            numeric_col = df.select_dtypes(include=['number']).columns[0]
            series = df[numeric_col]
            decomposition = seasonal_decompose(series, model='additive', period=12)
            fig = decomposition.plot()
            fig.set_size_inches(14, 8)  # Adjusted size
            return fig
        else:
            st.write("No numeric columns available for time series decomposition.")
            return None
    except Exception as e:
        st.error(f"Error decomposing time series for {time_stamp_col}: {e}")
        return None

def plot_correlation_heatmap(correlation_matrix):
    try:
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
        fig.update_layout(height=500)
        return fig
    except Exception as e:
        st.error(f"Error plotting correlation heatmap: {e}")
        return None

def create_summary_df(df):
    summary_data = {
        "Column Name": [],
        "Unique Values": [],
        "Column Type": []
    }
    
    for col in df.columns:
        summary_data["Column Name"].append(col)
        summary_data["Unique Values"].append(df[col].nunique())
        if pd.api.types.is_numeric_dtype(df[col]):
            summary_data["Column Type"].append("Numeric")
        else:
            summary_data["Column Type"].append("Categorical")
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Streamlit App
st.set_page_config(layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Unified Data Analysis Dashboard')

# Sidebar Menu
menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Question based Graph", "Data Dashboard", "Ask CSV"])

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    if df is not None:
        numeric_cols, categorical_cols = identify_columns(df)
        primary_keys = identify_primary_key_columns(df)
        time_stamp_col = identify_time_stamp_column(df)
        if menu == "Summarize":
            st.subheader("Summarization of your Data")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Summary:")
                st.write(df.describe())
            with col2:
                st.subheader('Uploaded Data Contains')
                summary_df = create_summary_df(df)
                st.write(summary_df)
            num_goals = st.sidebar.slider("Number of Goals", min_value=0, max_value=5, value=0)

            # Save the DataFrame to a temporary CSV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                df.to_csv(temp_csv.name, index=False)
                temp_csv_path = temp_csv.name
            if(num_goals==0):
                st.write("Set Goal to at-least 1")
            else :
                summary = lida.summarize(temp_csv_path, summary_method="default", textgen_config=textgen_config)

                goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
                for i, goal in enumerate(goals):
                    st.write(f"Goal {i + 1}: {goal}")

                    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                    charts = lida.visualize(summary=summary, goal=goal, textgen_config=textgen_config, library="seaborn")
                
                    img_base64_string = charts[0].raster
                    img = base64_to_image(img_base64_string)
                    st.image(img)

        elif menu == "Question based Graph":
            st.subheader("Query your Data to Generate Graph")
            summary_df = create_summary_df(df)
            st.write(summary_df)
            text_area = st.text_area("Query your Data to Generate Graph", height=100)
            if st.button("Generate Graph"):
                if len(text_area) > 0:
                    st.info("Your Query: " + text_area)
                    
                    # Save the DataFrame to a temporary CSV file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                        df.to_csv(temp_csv.name, index=False)
                        temp_csv_path = temp_csv.name

                    summary = lida.summarize(temp_csv_path, summary_method="default", textgen_config=textgen_config)
                    user_query = text_area
                    
                    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)
                    image_base64 = charts[0].raster
                    img = base64_to_image(image_base64)
                    st.image(img)

        elif menu == "Data Dashboard":
            # Layout for correlation and numerical analysis
            col1, col2 = st.columns([6,4])

            with col1:
                st.subheader('Correlation Plot')
                if numeric_cols:
                    correlation_matrix = calculate_correlation(df, numeric_cols)
                    corr_fig = plot_correlation_heatmap(correlation_matrix)
                    if corr_fig:
                        st.plotly_chart(corr_fig)
                else:
                    st.write("No numeric columns for correlation matrix.")

            with col2:
                st.subheader("Data Summary:")
                st.write(df.describe())

            # Layout for bar plots and time series plot
            col1, col2 = st.columns([6,5])

            with col1:
                st.header('Bar Plots')
                if categorical_cols:
                    bar_fig = plot_bar_charts(df, categorical_cols)
                    if bar_fig:
                        st.plotly_chart(bar_fig)
                else:
                    st.write("No categorical columns to analyze.")

            with col2:
                st.header('Time Series Analysis')
                if time_stamp_col:
                    ts_fig = plot_time_series(df, time_stamp_col)
                    if ts_fig:
                        st.plotly_chart(ts_fig)

                    st.subheader('Time Series Decomposition')
                    ts_decomp_fig = decompose_time_series(df, time_stamp_col)
                    if ts_decomp_fig:
                        st.pyplot(ts_decomp_fig)
        
        elif menu == "Ask CSV":
            st.header("Ask CSV 📅 using OpenAI ֎")
            user_question = st.text_input("Shoot your Qns on the CSV file")
            llm = OpenAI(temperature=0.3)

            if uploaded_file is not None:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                        df.to_csv(temp_csv.name, index=False)
                        temp_csv_path = temp_csv.name

                    # Set allow_dangerous_code=True to enable the agent to run Python code
                    agent = create_csv_agent(llm, temp_csv_path, verbose=True, allow_dangerous_code=True)
                    
                    if user_question:
                        response = agent.run(user_question)
                        st.write(response)
                except Exception as e:
                    st.error(f"Error creating CSV agent: {e}")

else:
    st.write("Please upload a CSV file to proceed.")
