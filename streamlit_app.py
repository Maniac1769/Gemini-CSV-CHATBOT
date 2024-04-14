import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import GooglePalm
import matplotlib

matplotlib.use('TkAgg')


API_KEY = st.secrets["GOOGLE_PALM2"]
llm = GooglePalm(api_key=API_KEY)


st.title('Upload Your CSV File!!')
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    prompt = st.text_area("Enter Your Prompt")
    prompt=f"
    The given csv is a classroom attendance csv with columns ['Enrollment','Name(name is in ['Name'] format) ] and rest of the columns are dates , 
    each row has a students name enrollment no. and 1s and 0s for if they were present or absent on that day keeping that in mind answer this : {prompt}"
    df = SmartDataframe(df, config={"llm": llm})
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating Response..."):
                # st.write("It IS Generating")
                st.write(df.chat(prompt))
        else:
            st.warning("Enter Prompt")
