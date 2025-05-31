import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import asyncio
from google import genai

# Set page to open in wide mode
st.set_page_config(layout="wide")

# Function to calculate and plot feature importance
def plot_feature_importance(X, Y, feature_names):
    clf = ExtraTreesClassifier(n_estimators=250, random_state=42)
    clf.fit(X, Y)

    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)  
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(10, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.xlim(0, 100)
    plt.tight_layout()

    return plt, sorted_idx, feature_importance

# AI-based reasoning function for feature selection
def reasoning(columns, domain):
    asyncio.set_event_loop(asyncio.new_event_loop())  # Ensure an event loop exists
    
    client = genai.Client(api_key="AIzaSyBEvZK3VC-3Kw-MKN-UJPEMX3pXPnnxKac")

    prompt = (f"My dataset consists of the following features: {columns}. "
              f"The data belongs to the {domain} domain. "
              f"I need to select the most relevant features for building an ML model. "
              f"Considering domain knowledge, feature relevance, correlations, and statistical importance, "
              f"which features should I prioritize and why? Provide an explanation for the top suggested features.")
    
    with st.spinner("Generating suggestion..."):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

    return response.text

if 'df_final' in st.session_state:
    df_final = st.session_state['df_final']
    st.title("Feature Engineering")

    st.subheader("Select Target Variable")
    target_column = st.selectbox("Select the target variable:", options=df_final.columns.tolist(), index=len(df_final.columns) - 1)

    independent_columns = [col for col in df_final.columns if col != target_column]
    st.write(f"Independent features: {', '.join(independent_columns)}")

    if independent_columns and target_column:
        if 'df_final' in st.session_state and 'selected_features' in st.session_state and 'target_column' in st.session_state:
            df_final = st.session_state['df_final']
        X_c = df_final[independent_columns]
        Y_c = df_final[target_column]

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, Y_c, test_size=0.20, random_state=0, stratify=Y_c)

        fig, sorted_idx, feature_importance = plot_feature_importance(X_train_c, y_train_c, X_c.columns)
        st.pyplot(fig)

        top_3_idx = sorted_idx[-3:]
        top_3_features = X_c.columns[top_3_idx]

        # Fetch actual column names dynamically
        actual_columns = df_final.columns.tolist()
        st.write("Dataset Columns:", actual_columns)
        
        # Generate reasoning based on dataset columns
        domain = "Healthcare" if "Pregnancies" in actual_columns else "Sales"
        suggestion = reasoning(actual_columns, domain)
    
        st.subheader("Suggestion")
        st.write(suggestion)
    
        st.subheader("Feature Selection")
        selected_features = st.multiselect("Select the features you think are important for model building", options=X_c.columns[sorted_idx], default=list(top_3_features))

        # Store both selected features and target column in session state
        if selected_features:
            st.session_state['selected_features'] = selected_features
            st.session_state['target_column'] = target_column  # Store target
            st.write(f"Chosen features: {', '.join(selected_features)}")
        else:
            st.warning("Please choose features for model building.")

    else:
        st.error("Please select a target variable.")
else:
    st.error("No cleaned DataFrame found in session state.")
