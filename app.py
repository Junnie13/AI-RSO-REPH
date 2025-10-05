from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd
import streamlit as st
 
# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Employee Retention Dashboard",
    layout="wide"
)
# Import your data
df = pd.read_csv("dataset.csv")

overview_tab, drilldown_tab = st.tabs(["📊 Overview", "🔍 Data Drilldown"])

with drilldown_tab: 
    pyg_app = StreamlitRenderer(df)
    pyg_app.explorer()