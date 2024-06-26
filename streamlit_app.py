import streamlit as st
import pandas as pd
import os
from pandasai import SmartDataframe
from pandasai.llm import BambooLLM
from pandasai.llm import GooglePalm
import matplotlib

matplotlib.use('TkAgg')

os.environ["PANDASAI_API_KEY"] = "$2a$10$b7Q3yDy7YeFeW.VIWj4b8OmWzKN19C16GxAiTsh57MQ/timN8MLvm"

API_KEY = st.secrets["GOOGLE_PALM2"]
llm =BambooLLM(api_key="$2a$10$b7Q3yDy7YeFeW.VIWj4b8OmWzKN19C16GxAiTsh57MQ/timN8MLvm")


st.title('Upload Your CSV File!!')
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    prompt = st.text_area("Enter Your Prompt")
    prompt=f"The given csv is a classroom attendance csv with columns ['Enrollment','Name(name is in ['Name'] format) ] and rest of the columns are dates , each row has a students name enrollment no. and 1s and 0s for if they were present or absent on that day keeping that in mind answer this : {prompt}"
    df = SmartDataframe(df, config={"llm": llm})
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating Response..."):
                # st.write("It IS Generating")
                st.write(df.chat(prompt))
        else:
            st.warning("Enter Prompt")
