import streamlit as st
import pandas as pd

# Set page layout
st.set_page_config(layout="wide")
st.title("Machine Learning Pipeline")

# File upload section
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension == "xlsx":
        df = pd.read_excel(uploaded_file)
    
    # Save dataframe to session state for later steps
    st.session_state['df'] = df

    # Display Data Preview
    st.subheader("Data Preview")
    st.write(df.head())
    
    st.success("✅ Dataset uploaded successfully! Proceed to the next step.")

    # Ask for domain selection
    st.header("Select Domain of Data")
    domain = st.selectbox("Choose the domain related to your data:",
                          ["Healthcare", "Finance", "Retail", "Manufacturing", "Other"])
    st.session_state['domain'] = domain

    # State key deliverables with green checkmarks
    st.header("Key Deliverables")
    st.markdown("""
    - ✅ **Model Download**
    - ✅ **Automated Report Generation**
    - ✅ **Guided Step Recommendations (Suggestions Based on Data & Domain**
    - ✅ **Model Interpretability & Explainability (Explainable AI**
    """)

