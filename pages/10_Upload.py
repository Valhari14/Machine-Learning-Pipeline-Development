import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import MinMaxScaler
import io
import base64

# Set page to open in wide mode
st.set_page_config(layout="wide", page_title="Model Evaluator")

# Define component builders for tooltips and info icons (reused from your original code)
def build_title_with_info(title, info_text, heading_level=3):
    """Create a heading with an info tooltip"""
    heading_tag = f"h{heading_level}"
    st.markdown(
        f"""
        <style>
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: help;
        }}
        .info-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #0066cc;
            background-color: #e6f2ff;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 5px;
            border: 1px solid #0066cc;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 250px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 14px;
            line-height: 1.4;
        }}
        .tooltip .tooltiptext::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        </style>
        <{heading_tag}>{title} <span class="tooltip"><span class="info-icon">i</span><span class="tooltiptext">{info_text}</span></span></{heading_tag}>
        """,
        unsafe_allow_html=True
    )

def build_label_with_info(label, info_text):
    """Create a label with an info tooltip"""
    st.markdown(
        f"""
        <div style="margin-bottom: 5px; font-weight: 500;">{label} <span class="tooltip"><span class="info-icon">i</span><span class="tooltiptext">{info_text}</span></span></div>
        """,
        unsafe_allow_html=True
    )

def build_regular_heading(title, heading_level=3):
    """Create a heading without an info tooltip"""
    st.markdown(f"<h{heading_level}>{title}</h{heading_level}>", unsafe_allow_html=True)

def build_regular_label(label):
    """Create a label without an info tooltip"""
    st.markdown(f"<div style='margin-bottom: 5px; font-weight: 500;'>{label}</div>", unsafe_allow_html=True)

# Custom CSS for UI elements
st.markdown("""
    <style>
        /* Style the file uploader */
        div[data-testid="stFileUploader"] {
            border: 1px solid #0066cc;
            border-radius: 5px;
            padding: 10px;
            background-color: #f8f9fa;
            margin-bottom: 20px;
        }
        
        /* Style the checkbox input */
        div[data-testid="stCheckbox"] > div {
            display: flex;
            align-items: center;
        }

        /* Add some padding and margin around the checkbox */
        div[data-testid="stCheckbox"] {
            margin-bottom: 10px;
            padding: 5px;
            border: 1px solid #0066cc;
            border-radius: 5px;
            background-color: #f8f9fa;
        }

        /* Style the button */
        div[data-testid="stButton"] button {
            background-color: #0066cc;
            color: white;
            border-radius: 5px;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s;
        }
        
        div[data-testid="stButton"] button:hover {
            background-color: #004c99;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Style for the number inputs */
        div[data-testid="stNumberInput"] input {
            border: 1px solid #0066cc;
            border-radius: 5px;
            padding: 8px;
        }
        
        /* Card-like containers for inputs */
        .input-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    st.pyplot(plt)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(plt)

# Function to load a model
def load_model(uploaded_file):
    if uploaded_file is not None:
        try:
            # Load the model
            bytes_data = uploaded_file.getvalue()
            model = joblib.load(io.BytesIO(bytes_data))
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# Function to load test data
def load_test_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Load the CSV file
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

# Function to preprocess test data for prediction
def preprocess_data(df, feature_columns, target_column=None):
    if target_column and target_column in df.columns:
        features = df[feature_columns]
        target = df[target_column]
    else:
        features = df[feature_columns]
        target = None
    
    # Apply the same scaling as in the original code
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    if target is not None:
        target = target.values
    
    return features_scaled, target

# Function to make predictions with the loaded model
def make_predictions(model, X):
    try:
        # Handle different model types
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            # Some models return a single column for binary classification
            if y_pred_proba.shape[1] == 1:
                y_pred_proba = np.column_stack((1 - y_pred_proba, y_pred_proba))
            y_pred_proba = y_pred_proba[:, 1]  # Get probabilities for the positive class
        else:
            y_pred_proba = model.predict(X)
            
        y_pred = (y_pred_proba > 0.5).astype(int)
        return y_pred, y_pred_proba
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None, None

def create_download_link(df):
    """Generate a link to download the predictions as a CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" style="background-color: #0066cc; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; font-weight: bold;">Download Predictions</a>'
    return href

def main():
    build_title_with_info(
        "Machine Learning Model Uploader and Evaluator", 
        "This application allows you to upload trained models, evaluate them on test data, and make new predictions.",
        heading_level=1
    )
    
    # Sidebar for model upload and configuration
    st.sidebar.title("Model Configuration")
    
    # Upload model file
    build_label_with_info(
        "Upload Model", 
        "Upload a trained model file (*.pkl) that was exported from the ensemble learning application."
    )
    
    uploaded_model = st.sidebar.file_uploader("Upload your model (.pkl file)", type=["pkl"])
    
    # Upload test data file
    build_label_with_info(
        "Upload Data", 
        "Upload a CSV file with test data to evaluate the model or make new predictions."
    )
    
    uploaded_data = st.sidebar.file_uploader("Upload your test data (.csv file)", type=["csv"])
    
    # Define feature columns (allowing customization)
    if uploaded_data is not None:
        df = load_test_data(uploaded_data)
        if df is not None:
            st.sidebar.subheader("Configure Features")
            
            # Display column selection
            all_columns = df.columns.tolist()
            default_feature_columns = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]
            
            # Filter default columns to only include those that actually exist in the data
            default_feature_columns = [col for col in default_feature_columns if col in all_columns]
            
            feature_columns = st.sidebar.multiselect(
                "Select feature columns",
                options=all_columns,
                default=default_feature_columns
            )
            
            # Target column selection (optional, for evaluation)
            has_target = st.sidebar.checkbox("Dataset contains target column for evaluation", value=True)
            target_column = None
            
            if has_target:
                target_options = [col for col in all_columns if col not in feature_columns]
                if target_options:
                    target_column = st.sidebar.selectbox(
                        "Select target column",
                        options=target_options,
                        index=0 if "Outcome" in target_options else 0
                    )
    
    # Main content
    tab1, tab2 = st.tabs(["Model Evaluation", "Make Predictions"])
    
    with tab1:
        build_regular_heading("Model Evaluation", heading_level=2)
        
        if uploaded_model is not None and uploaded_data is not None:
            model = load_model(uploaded_model)
            df = load_test_data(uploaded_data)
            
            if model is not None and df is not None and 'feature_columns' in locals() and len(feature_columns) > 0:
                if has_target and target_column:
                    st.info(f"Evaluating model using {len(feature_columns)} features and target column '{target_column}'")
                    
                    X_test, y_test = preprocess_data(df, feature_columns, target_column)
                    
                    if X_test is not None and y_test is not None:
                        y_pred, y_pred_proba = make_predictions(model, X_test)
                        
                        if y_pred is not None and y_pred_proba is not None:
                            # Display classification report
                            st.subheader("Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            st.table(pd.DataFrame(report).transpose())
                            
                            # Key metrics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                build_label_with_info(
                                    "Accuracy", 
                                    "The proportion of correctly classified instances out of all instances. Higher is better."
                                )
                                st.write(f"{(y_test == y_pred).mean():.4f}")
                                
                                build_label_with_info(
                                    "Precision", 
                                    "The ability of the model to avoid false positives. Higher values indicate fewer false positives."
                                )
                                st.write(f"{precision_score(y_test, y_pred):.4f}")
                                
                                build_label_with_info(
                                    "Recall", 
                                    "The ability of the model to find all positive instances. Higher values indicate fewer false negatives."
                                )
                                st.write(f"{recall_score(y_test, y_pred):.4f}")
                            
                            with col2:
                                build_label_with_info(
                                    "F1-score", 
                                    "The harmonic mean of precision and recall. Provides a balance between the two metrics."
                                )
                                st.write(f"{f1_score(y_test, y_pred):.4f}")
                                
                                build_label_with_info(
                                    "ROC AUC", 
                                    "Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between classes."
                                )
                                st.write(f"{roc_auc_score(y_test, y_pred_proba):.4f}")
                            
                            # Evaluation Plots
                            st.subheader("Evaluation Plots")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("ROC Curve")
                                plot_roc_curve(y_test, y_pred_proba)
                            
                            with col2:
                                st.write("Precision-Recall Curve")
                                plot_precision_recall_curve(y_test, y_pred_proba)
                            
                            with col3:
                                st.write("Confusion Matrix")
                                plot_confusion_matrix(y_test, y_pred)
                        else:
                            st.warning("Unable to make predictions with the uploaded model.")
                    else:
                        st.warning("Unable to preprocess the data.")
                else:
                    st.warning("Please select a target column for evaluation.")
            else:
                st.warning("Please check your model, data, and feature selections.")
        else:
            st.info("Please upload a model file and test data to evaluate.")
    
    with tab2:
        build_regular_heading("Make Predictions", heading_level=2)
        
        if uploaded_model is not None:
            model = load_model(uploaded_model)
            
            if model is not None:
                # Input method selection
                input_method = st.radio("Select input method", ["Input Form", "Upload CSV"])
                
                if input_method == "Upload CSV":
                    if uploaded_data is not None:
                        df = load_test_data(uploaded_data)
                        
                        if df is not None and 'feature_columns' in locals() and len(feature_columns) > 0:
                            # Display the first few rows of the data
                            st.subheader("Preview of Data")
                            st.dataframe(df[feature_columns].head())
                            
                            # Process the data for prediction
                            X_test, _ = preprocess_data(df, feature_columns)
                            
                            if X_test is not None:
                                # Make predictions button
                                if st.button("Generate Predictions"):
                                    y_pred, y_pred_proba = make_predictions(model, X_test)
                                    
                                    if y_pred is not None and y_pred_proba is not None:
                                        # Create a dataframe with predictions
                                        results_df = df.copy()
                                        results_df['Prediction'] = y_pred
                                        results_df['Prediction_Probability'] = y_pred_proba
                                        
                                        # Display the results
                                        st.subheader("Prediction Results")
                                        st.dataframe(results_df)
                                        
                                        # Add download button for predictions
                                        st.markdown(create_download_link(results_df), unsafe_allow_html=True)
                                    else:
                                        st.error("Failed to generate predictions.")
                            else:
                                st.warning("Unable to preprocess the data.")
                        else:
                            st.warning("Please check your data and feature selections.")
                    else:
                        st.info("Please upload test data to make predictions.")
                
                elif input_method == "Input Form":
                    st.subheader("Enter Feature Values")
                    
                    # Default features for diabetes prediction
                    default_features = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]
                    
                    # Create input fields for each feature
                    input_values = {}
                    
                    for feature in default_features:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            build_label_with_info(
                                feature, 
                                f"Enter the value for {feature}"
                            )
                        with col2:
                            input_values[feature] = st.number_input(
                                f"{feature}", 
                                min_value=0.0, 
                                value=100.0 if feature == "Glucose" else 25.0,
                                key=f"input_{feature}",
                                label_visibility="collapsed"
                            )
                    
                    # Make prediction button
                    if st.button("Predict"):
                        # Create a dataframe from input values
                        input_df = pd.DataFrame([input_values])
                        
                        # Preprocess the input
                        X_input, _ = preprocess_data(input_df, list(input_values.keys()))
                        
                        # Generate prediction
                        y_pred, y_pred_proba = make_predictions(model, X_input)
                        
                        if y_pred is not None and y_pred_proba is not None:
                            # Display prediction
                            st.subheader("Prediction Result")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    label="Predicted Class", 
                                    value="Positive (Diabetic)" if y_pred[0] == 1 else "Negative (Non-Diabetic)"
                                )
                            
                            with col2:
                                st.metric(
                                    label="Probability", 
                                    value=f"{y_pred_proba[0]:.4f}"
                                )
                            
                            # Visualization of the probability
                            fig, ax = plt.subplots(figsize=(10, 2))
                            ax.barh([0], [y_pred_proba[0]], color='#0066cc', height=0.5)
                            ax.barh([0], [1-y_pred_proba[0]], left=[y_pred_proba[0]], color='#cccccc', height=0.5)
                            ax.set_xlim(0, 1)
                            ax.set_yticks([])
                            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                            ax.set_xlabel('Probability of Diabetes')
                            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                        else:
                            st.error("Failed to generate prediction.")
            else:
                st.warning("Failed to load the model.")
        else:
            st.info("Please upload a model file to make predictions.")

if __name__ == "__main__":
    main()