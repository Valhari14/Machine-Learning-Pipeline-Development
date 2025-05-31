import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from google import genai
import asyncio

# Set page to open in wide mode
st.set_page_config(layout="wide")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    return fig

# General model evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC AUC": auc
    }
    cm_plot = plot_confusion_matrix(cm, ['Not Diabetic', 'Diabetic'])
    
    return metrics, cm_plot

# AI-based reasoning function
def reasoning(columns, domain, evals):
    asyncio.set_event_loop(asyncio.new_event_loop())  # Ensure an event loop exists
    
    client = genai.Client(api_key="AIzaSyBEvZK3VC-3Kw-MKN-UJPEMX3pXPnnxKac")

    prompt = (f"I have a dataset with columns {columns}, and the data belongs to the {domain} domain. "
            f"I am developing an intelligent ML pipeline to select the most suitable model for specific tasks. "
            f"Currently, the focus is on binary classification in {domain}. I have a list of models including "
            f"Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, AdaBoost, "
            f"Gradient Boosting, Random Forest, and Extra Trees. The evaluation metrics are Accuracy, Precision, "
            f"Recall, F1-score, and ROC AUC. The base model evaluation results are {evals}. "
            f"I want to rank these models based on their performance while incorporating {domain} domain knowledge "
            f"to determine the best model for my task, Do not provide code")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    return response.text

# Streamlit UI
st.title('Model Selection & Baseline Algorithm Evaluation')

# Check session state for required data
if 'df_final' in st.session_state and 'selected_features' in st.session_state and 'target_column' in st.session_state:
    df_final = st.session_state['df_final']
    selected_features = st.session_state['selected_features']
    target_column = st.session_state['target_column']

    st.write(f"Model will be trained on features: {', '.join(selected_features)}")
    st.write(f"The target variable is: {target_column}")

    X = df_final[selected_features]
    Y = df_final[target_column]

    st.subheader("Suggested Models for Binary Classification")
    suggested_models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Extra Trees": ExtraTreesClassifier()
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0, stratify=Y)
    
    model_results = {}
    comparison_data = []

    for model_name, model in suggested_models.items():
        model.fit(X_train, y_train)
        metrics, cm_plot = evaluate_model(model, X_test, y_test)
        model_results[model_name] = (metrics, cm_plot, {})

    comparison_df = pd.DataFrame([{**{"Model": model}, **metrics} for model, (metrics, _, _) in model_results.items()])
    comparison_df.set_index('Model', inplace=True)
    
    st.header("Metric Comparison Across Models")
    st.dataframe(comparison_df)
    
    # Fetch actual column names dynamically
    actual_columns = df_final.columns.tolist()
    st.write("Dataset Columns:", actual_columns)
    
    # Generate reasoning based on dataset columns
    domain = "Healthcare" if "Pregnancies" in actual_columns else "Sales"
    suggestion = reasoning(actual_columns, domain, comparison_df.to_dict())
    
    st.subheader("Suggestion")
    st.write(suggestion)
    
    # Plot Model Comparison
    st.header("Comparison Plot of Model Performance")
    comparison_df_long = comparison_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    comparison_plot = px.bar(comparison_df_long, x='Metric', y='Score', color='Model', barmode='group',
                             title='Model Performance Comparison')
    st.plotly_chart(comparison_plot)
    
    # Model selection for ensembling
    st.header("Select Models for Ensembling")
    selected_models = st.multiselect("Choose models for further evaluation:", list(suggested_models.keys()))
    
    if selected_models:
        st.session_state['selected_models'] = selected_models
        st.success(f"Selected models saved: {', '.join(selected_models)}")
    else:
        st.warning("Please select at least one model for further evaluation.")
    
    if 'selected_models' in st.session_state:
        st.write("You have selected the following models for further evaluation:")
        st.write(', '.join(st.session_state['selected_models']))
