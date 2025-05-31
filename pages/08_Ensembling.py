import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                     cross_val_score)
from sklearn.ensemble import (VotingClassifier, GradientBoostingClassifier, 
                              RandomForestClassifier, ExtraTreesClassifier, 
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import MinMaxScaler
from mlens.ensemble import SuperLearner

# Set page to open in wide mode
st.set_page_config(layout="wide")

# Ignore Warnings
warnings.filterwarnings('ignore')

# Set a seed for reproducibility
SEED = 7
np.random.seed(SEED)

# Define component builders for tooltips and info icons
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

# Function to create a regular heading without info icon
def build_regular_heading(title, heading_level=3):
    """Create a heading without an info tooltip"""
    st.markdown(f"<h{heading_level}>{title}</h{heading_level}>", unsafe_allow_html=True)

# Function to create a regular label without info icon
def build_regular_label(label):
    """Create a label without an info tooltip"""
    st.markdown(f"<div style='margin-bottom: 5px; font-weight: 500;'>{label}</div>", unsafe_allow_html=True)

# Custom CSS for button and checkbox styling
st.markdown("""
    <style>
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

        /* Style the download button */
        div[data-testid="stDownloadButton"] button {
            background-color: #0066cc;
            color: white;
            border-radius: 5px;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s;
        }
        
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #004c99;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_models():
    """Generate a library of base learners."""
    models = {
        'Logistic Regression': LogisticRegression(C=0.7678243129497218, penalty='l2', solver='saga'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=15),
        'Support Vector Machine': SVC(C=1.7, kernel='linear', probability=True),  # Ensure probability=True
        'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=3, max_features=2, min_samples_leaf=3),
        'AdaBoost': AdaBoostClassifier(learning_rate=0.05, n_estimators=150),
        'Gradient Boosting': GradientBoostingClassifier(learning_rate=0.01, n_estimators=100),
        'Gaussian Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(),
        'Extra Trees': ExtraTreesClassifier()
    }
    return models

@st.cache_data
def load_and_preprocess_data():
    df_clean = pd.read_csv('df_clean_output.csv')
    df_unscaled = df_clean[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Outcome']]
    df_imp_scaled = MinMaxScaler().fit_transform(df_unscaled)
    
    X = df_imp_scaled[:, 0:4]
    Y = df_imp_scaled[:, 4]
    
    X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(
        X, Y, test_size=0.1, random_state=0, stratify=df_imp_scaled[:, 4]
    )
    
    return X_train_sc, X_test_sc, y_train_sc, y_test_sc

@st.cache_resource
def create_and_train_ensemble(_X_train_sc, _y_train_sc, _X_test_sc, _y_test_sc):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    ensemble = VotingClassifier(estimators=list(get_models().items()), voting='soft')  # Use soft voting for probabilities
    results = cross_val_score(ensemble, _X_train_sc, _y_train_sc, cv=kfold)
    
    ensemble_model = ensemble.fit(_X_train_sc, _y_train_sc)
    pred = ensemble_model.predict(_X_test_sc)
    pred_proba = ensemble_model.predict_proba(_X_test_sc)[:, 1]  # Now this should work correctly
    
    return ensemble_model, results.mean(), pred, pred_proba

@st.cache_data
def train_predict(_model_list, _xtrain, _xtest, _ytrain):
    """Fit models in list on training set and return predictions."""
    P = pd.DataFrame(np.zeros((_xtest.shape[0], len(_model_list))), columns=_model_list.keys())

    for name, model in _model_list.items():
        model.fit(_xtrain, _ytrain)
        P[name] = model.predict_proba(_xtest)[:, 1]

    return P

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

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(plt)

# Save the trained Super Learner model
def save_model(model, filename="super_learner_model.pkl"):
    joblib.dump(model, filename)
    st.success("Model saved successfully!")

# Provide a download button for the saved model
def download_model(filename="super_learner_model.pkl"):
    with open(filename, "rb") as f:
        st.download_button(
            label="📥 Download Trained Model",
            data=f,
            file_name=filename,
            mime="application/octet-stream"
        )

def main():
    build_title_with_info(
        "Ensemble Learning with Comprehensive Evaluation", 
        "This application demonstrates how to combine multiple machine learning models to create a stronger ensemble model. It includes detailed evaluation metrics and visualizations.",
        heading_level=1
    )

    # Load and preprocess data
    with st.spinner("Loading and preprocessing data..."):
        X_train_sc, X_test_sc, y_train_sc, y_test_sc = load_and_preprocess_data()

    st.session_state.base_model_created = True
    
    build_label_with_info(
        "Creating Base Ensemble Model", 
        "Building a Voting Classifier that combines predictions from multiple base models. This forms the foundation for our ensemble approach."
    )
    
    with st.spinner("Creating and training base model..."):
        ensemble_model, cv_accuracy, y_pred, y_pred_proba = create_and_train_ensemble(X_train_sc, y_train_sc, X_test_sc, y_test_sc)
    
    st.success("Base model created successfully!")
    st.write('Cross-validation accuracy: ', cv_accuracy)

    # Evaluate the ensemble model - MODIFIED TO USE REGULAR HEADING
    build_regular_heading(
        "Ensemble Model Evaluation",
        heading_level=2
    )
    
    # Classification report - MODIFIED TO USE REGULAR LABEL
    build_regular_label("Classification Report")
    
    report = classification_report(y_test_sc, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

    # Additional metrics - MODIFIED TO USE REGULAR HEADING
    build_regular_heading(
        "Key Performance Metrics",
        heading_level=4
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        build_label_with_info(
            "Accuracy", 
            "The proportion of correctly classified instances out of all instances. Higher is better."
        )
        st.write(f"{(y_test_sc == y_pred).mean():.4f}")
        
        build_label_with_info(
            "Precision", 
            "The ability of the model to avoid false positives. Higher values indicate fewer false positives."
        )
        st.write(f"{precision_score(y_test_sc, y_pred):.4f}")
        
        build_label_with_info(
            "Recall", 
            "The ability of the model to find all positive instances. Higher values indicate fewer false negatives."
        )
        st.write(f"{recall_score(y_test_sc, y_pred):.4f}")
    
    with col2:
        build_label_with_info(
            "F1-score", 
            "The harmonic mean of precision and recall. Provides a balance between the two metrics."
        )
        st.write(f"{f1_score(y_test_sc, y_pred):.4f}")
        
        build_label_with_info(
            "ROC AUC", 
            "Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between classes. Higher values indicate better performance."
        )
        st.write(f"{roc_auc_score(y_test_sc, y_pred_proba):.4f}")

    # Plots - MODIFIED TO USE REGULAR HEADING
    build_regular_heading(
        "Evaluation Plots",
        heading_level=2
    )
    
    # Create three columns for the plots
    col1, col2, col3 = st.columns(3)
    
    # ROC Curve
    with col1:
        build_title_with_info(
            "ROC Curve", 
            "Shows the tradeoff between true positive rate and false positive rate at various threshold settings. The area under the curve (AUC) quantifies the model's ability to discriminate between classes.",
            heading_level=4
        )
        plot_roc_curve(y_test_sc, y_pred_proba)

    # Precision-Recall Curve
    with col2:
        build_title_with_info(
            "Precision-Recall Curve", 
            "Shows the tradeoff between precision and recall at various threshold settings. Particularly useful for imbalanced datasets.",
            heading_level=4
        )
        plot_precision_recall_curve(y_test_sc, y_pred_proba)

    # Confusion Matrix
    with col3:
        build_title_with_info(
            "Confusion Matrix", 
            "Shows the counts of true positives, false positives, true negatives, and false negatives. Helps identify where the model is making mistakes.",
            heading_level=4
        )
        plot_confusion_matrix(y_test_sc, y_pred)

    # Generate predictions and display correlation matrix
    models = get_models()
    P = train_predict(models, X_train_sc, X_test_sc, y_train_sc)

    # Super Learner section - MODIFIED TO USE REGULAR HEADING
    build_regular_heading(
        "Super Learner Training",
        heading_level=2
    )
    
    # Access the selected models from session state
    if 'selected_models' in st.session_state:
        # Checkbox to select all models - MODIFIED TO USE REGULAR LABEL
        build_regular_label("Model Selection")
        select_all = st.checkbox("Select All Models", value=True)
        
        if select_all:
            selected_model_names = st.session_state['selected_models']  # Automatically select all models
        else:
            selected_model_names = []

        st.write("Selected models:", selected_model_names)

        # Train Super Learner - MODIFIED TO USE REGULAR LABEL
        build_regular_label("Train Super Learner")
        if st.button("Train Super Learner"):
            if not st.session_state.get('base_model_created', False):
                st.warning("Please create the base model first before training the Super Learner.")
            elif selected_model_names:
                with st.spinner("Training Super Learner..."):
                    super_learner = SuperLearner(scorer=roc_auc_score, random_state=SEED)
                    
                    # Add models without the name argument
                    base_learners = {name: models[name] for name in selected_model_names}
                    super_learner.add(list(base_learners.values()))

                    # Add the meta learner
                    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
                    super_learner.add(meta_learner)

                    super_learner.fit(X_train_sc, y_train_sc)
                    pred = super_learner.predict(X_test_sc)
                    pred_proba = super_learner.predict_proba(X_test_sc)

                    # Handle the case where pred_proba is one-dimensional
                    if pred_proba.ndim == 1:  # Only one class
                        pred_proba = np.column_stack((1 - pred_proba, pred_proba))

                    pred_proba = pred_proba[:, 1]  # Get probabilities for the positive class

                    st.success("Super Learner trained successfully!")

                    # Display metrics
                    build_title_with_info(
                        "Super Learner Performance Metrics", 
                        "Evaluation metrics showing how well the Super Learner performs compared to individual models.",
                        heading_level=4
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        build_label_with_info(
                            "Accuracy", 
                            "Proportion of correct predictions (both true positives and true negatives)."
                        )
                        st.write(f"{(y_test_sc == pred).mean():.4f}")
                        
                        build_label_with_info(
                            "Precision", 
                            "Proportion of true positives among all positive predictions. Shows how reliable positive predictions are."
                        )
                        st.write(f"{precision_score(y_test_sc, pred):.4f}")
                        
                        build_label_with_info(
                            "Recall", 
                            "Proportion of true positives identified among all actual positives. Shows the model's ability to find all positive cases."
                        )
                        st.write(f"{recall_score(y_test_sc, pred):.4f}")
                    
                    with col2:
                        build_label_with_info(
                            "F1-score", 
                            "Harmonic mean of precision and recall, providing a single metric that balances both concerns."
                        )
                        st.write(f"{f1_score(y_test_sc, pred):.4f}")
                        
                        build_label_with_info(
                            "ROC AUC", 
                            "Area Under the ROC Curve. Measures the model's ability to distinguish between classes across all possible classification thresholds."
                        )
                        st.write(f"{roc_auc_score(y_test_sc, pred_proba):.4f}")

                    # Super Learner Evaluation Plots
                    build_title_with_info(
                        "Super Learner Evaluation Plots", 
                        "Visual evaluation of the Super Learner's performance using different metrics and visualizations.",
                        heading_level=3
                    )
                    
                    # Create three columns for the plots
                    col1, col2, col3 = st.columns(3)

                    # ROC Curve
                    with col1:
                        build_title_with_info(
                            "ROC Curve", 
                            "Shows the Super Learner's ability to discriminate between positive and negative classes at various threshold settings.",
                            heading_level=4
                        )
                        plot_roc_curve(y_test_sc, pred_proba)

                    # Precision-Recall Curve
                    with col2:
                        build_title_with_info(
                            "Precision-Recall Curve", 
                            "Shows the tradeoff between precision and recall for the Super Learner at different classification thresholds.",
                            heading_level=4
                        )
                        plot_precision_recall_curve(y_test_sc, pred_proba)

                    # Confusion Matrix
                    with col3:
                        build_title_with_info(
                            "Confusion Matrix", 
                            "Visual representation of the Super Learner's prediction errors and correct classifications.",
                            heading_level=4
                        )
                        plot_confusion_matrix(y_test_sc, pred)
                        
                    # Save and Download Model
                    build_title_with_info(
                        "Save and Download Model", 
                        "Save the trained Super Learner model to disk and download it for future use in other applications.",
                        heading_level=3
                    )
                    save_model(super_learner)
                    download_model()

# Define test_model function separately from main
# def test_model():
#     """Test the trained model with user input."""
#     build_regular_heading(
#         "Test Your Model",
#         heading_level=2
#     )

#     st.write("Use this section to test the model with your own input values.")
    
#     # Create input form
#     with st.form(key="model_test_form"):
#         st.write("Enter values for the features used in training:")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             build_label_with_info(
#                 "Glucose", 
#                 "Plasma glucose concentration (mg/dL). Normal range is typically 70-99 mg/dL fasting."
#             )
#             glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=120.0, step=1.0)
            
#             build_label_with_info(
#                 "BMI", 
#                 "Body Mass Index (weight in kg/(height in m)²). Normal range is typically 18.5-24.9."
#             )
#             bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0, step=0.1)
            
#         with col2:
#             build_label_with_info(
#                 "Age", 
#                 "Age in years."
#             )
#             age = st.number_input("Age (years)", min_value=0, max_value=120, value=35, step=1)
            
#             build_label_with_info(
#                 "Diabetes Pedigree Function", 
#                 "A function that scores likelihood of diabetes based on family history. Higher values indicate greater risk."
#             )
#             diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        
#         submit_button = st.form_submit_button(label="Predict Diabetes Risk")
    
#     # Process the form submission
#     if submit_button:
#         # Try to load the model
#         try:
#             model_path = "super_learner_model.pkl"
            
#             if os.path.exists(model_path):
#                 # Load the trained model
#                 loaded_model = joblib.load(model_path)
                
#                 # Check if the model is a SuperLearner or standard sklearn model
#                 is_super_learner = hasattr(loaded_model, 'meta_learner_')
                
#                 # Prepare input data
#                 input_data = np.array([[glucose, bmi, age, diabetes_pedigree]])
                
#                 # Scale the input data using the same scaler used for training
#                 # First, check if df_clean_output.csv exists
#                 if os.path.exists('df_clean_output.csv'):
#                     scaler = MinMaxScaler()
#                     scaler.fit(pd.read_csv('df_clean_output.csv')[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']])
#                     input_data_scaled = scaler.transform(input_data)
#                 else:
#                     # If the file doesn't exist, assume the input is already in the right range (0-1)
#                     # or create a simple scaling based on typical ranges
#                     st.warning("Training data file not found. Using approximate scaling.")
                    
#                     # Approximate scaling based on typical ranges
#                     glucose_scaled = glucose / 200.0  # Assuming max glucose is around 200
#                     bmi_scaled = bmi / 50.0  # Assuming max BMI is around 50
#                     age_scaled = age / 100.0  # Assuming max age is around 100
#                     pedigree_scaled = diabetes_pedigree / 2.0  # Assuming max pedigree is around 2
                    
#                     input_data_scaled = np.array([[glucose_scaled, bmi_scaled, age_scaled, pedigree_scaled]])
                
#                 # Make prediction
#                 with st.spinner("Making prediction..."):
#                     # Try direct prediction first
#                     diabetes_probability = None
#                     try:
#                         if hasattr(loaded_model, 'predict_proba'):
#                             prediction_proba = loaded_model.predict_proba(input_data_scaled)
                            
#                             # Handle the case where prediction_proba is one-dimensional
#                             if prediction_proba.ndim == 1:
#                                 prediction_proba = np.column_stack((1 - prediction_proba, prediction_proba))
                            
#                             diabetes_probability = prediction_proba[0, 1] * 100  # Convert to percentage
#                         else:
#                             # If the model can't predict probabilities, use binary prediction
#                             prediction = loaded_model.predict(input_data_scaled)
#                             diabetes_probability = 100.0 if prediction[0] > 0.5 else 0.0
#                             st.warning("This model doesn't provide probability estimates. Using binary prediction.")
#                     except Exception as e:
#                         st.info(f"Standard prediction failed: {str(e)}. Trying alternative method for SuperLearner...")
                        
#                         # If direct prediction failed, try alternative approaches
#                         if diabetes_probability is None:
#                             # Extract base learners from the SuperLearner if possible
#                             if is_super_learner and hasattr(loaded_model, 'learners_'):
#                                 # Get predictions from base learners
#                                 base_preds = []
#                                 for learner in loaded_model.learners_:
#                                     try:
#                                         # Try to get probability prediction
#                                         if hasattr(learner, 'predict_proba'):
#                                             pred = learner.predict_proba(input_data_scaled)
#                                             if pred.ndim > 1 and pred.shape[1] > 1:
#                                                 base_preds.append(pred[:, 1])
#                                             else:
#                                                 base_preds.append(pred)
#                                         else:
#                                             # Fall back to binary prediction
#                                             pred = learner.predict(input_data_scaled)
#                                             base_preds.append(pred)
#                                     except:
#                                         # Skip learners that fail
#                                         continue
                                
#                                 # If we got any predictions from base learners, average them
#                                 if base_preds:
#                                     avg_pred = np.mean(base_preds)
#                                     diabetes_probability = float(avg_pred) * 100
#                                     st.warning("Using average of base learner predictions as SuperLearner couldn't process directly.")
                    
#                     # If we still don't have a probability, use fallback estimation
#                     if diabetes_probability is None:
#                         # Last resort - make a guess based on the feature values
#                         st.error("Could not get predictions from the model. Using feature-based estimation instead.")
                        
#                         # Simple heuristic based on feature values - THIS IS NOT A REAL MODEL!
#                         glucose_factor = (glucose - 70) / (180 - 70) if glucose > 70 else 0
#                         bmi_factor = (bmi - 18.5) / (35 - 18.5) if bmi > 18.5 else 0
#                         age_factor = age / 100
#                         pedigree_factor = diabetes_pedigree / 2
                        
#                         # Weighted combination
#                         diabetes_probability = (glucose_factor * 0.4 + bmi_factor * 0.3 + 
#                                                age_factor * 0.1 + pedigree_factor * 0.2) * 100
                        
#                         st.warning("⚠️ This is a fallback estimate, not the actual model prediction!")
                
#                 # Display results
#                 st.subheader("Prediction Results")
                
#                 # Create a gauge-like visualization
#                 fig, ax = plt.subplots(figsize=(8, 3))
                
#                 # Create a horizontal bar representing the risk spectrum
#                 ax.barh(['Risk'], [100], color='lightgray', height=0.6)
#                 ax.barh(['Risk'], [diabetes_probability], color='#ff9999' if diabetes_probability > 50 else '#66b3ff', height=0.6)
                
#                 # Add a vertical line for the threshold
#                 ax.axvline(x=50, color='red', linestyle='--', alpha=0.7)
                
#                 # Add text showing the exact probability
#                 ax.text(diabetes_probability, 0, f' {diabetes_probability:.1f}%', 
#                         va='center', ha='left' if diabetes_probability < 50 else 'right',
#                         fontweight='bold', fontsize=14, color='black')
                
#                 # Customize the plot
#                 ax.set_xlim(0, 100)
#                 ax.set_xlabel('Diabetes Risk (%)')
#                 ax.set_yticks([])
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.spines['left'].set_visible(False)
                
#                 st.pyplot(fig)
                
#                 # Risk interpretation
#                 if diabetes_probability < 20:
#                     risk_level = "Low"
#                     color = "green"
#                     message = "Based on the provided features, the model predicts a low risk of diabetes."
#                 elif diabetes_probability < 50:
#                     risk_level = "Moderate"
#                     color = "orange"
#                     message = "Based on the provided features, the model predicts a moderate risk of diabetes."
#                 else:
#                     risk_level = "High"
#                     color = "red"
#                     message = "Based on the provided features, the model predicts a high risk of diabetes."
                
#                 st.markdown(f"<h3 style='color: {color}'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
#                 st.write(message)
                
#                 st.warning("Remember that this is just a prediction based on limited data. Always consult with healthcare professionals for proper medical advice.")
                
#                 # Feature importance visualization (relative to the input)
#                 st.subheader("Feature Analysis")
#                 st.write("How each feature contributes to the prediction:")
                
#                 # Compare with population averages (these are fictional - replace with actual values)
#                 avg_glucose = 110.0
#                 avg_bmi = 26.5
#                 avg_age = 40.0
#                 avg_pedigree = 0.4
                
#                 fig, ax = plt.subplots(figsize=(10, 6))
                
#                 features = ['Glucose', 'BMI', 'Age', 'Diabetes Pedigree']
#                 user_values = [glucose, bmi, age, diabetes_pedigree]
#                 avg_values = [avg_glucose, avg_bmi, avg_age, avg_pedigree]
                
#                 x = np.arange(len(features))
#                 width = 0.35
                
#                 ax.bar(x - width/2, user_values, width, label='Your Values', color='#66b3ff')
#                 ax.bar(x + width/2, avg_values, width, label='Population Average', color='#c2c2d6')
                
#                 ax.set_ylabel('Value')
#                 ax.set_title('Your Values vs. Population Average')
#                 ax.set_xticks(x)
#                 ax.set_xticklabels(features)
#                 ax.legend()
                
#                 # Add percentage differences
#                 for i, (user, avg) in enumerate(zip(user_values, avg_values)):
#                     diff_pct = ((user - avg) / avg) * 100
#                     if abs(diff_pct) >= 5:  # Only show significant differences
#                         ax.text(i, max(user, avg) + 1, f"{diff_pct:+.1f}%", 
#                                 ha='center', va='bottom', 
#                                 color='green' if diff_pct < 0 and features[i] != 'Age' else 'red' if diff_pct > 0 and features[i] != 'Age' else 'black')
                
#                 st.pyplot(fig)
                
#             else:
#                 st.error("No model file found. Please train and save the model first.")
        
#         except Exception as e:
#             st.error(f"Error loading or using the model: {str(e)}")
#             st.info("Make sure you've trained and saved the model before testing it.")

# # End of code to be added

# def test_model():
#     """Test the trained model with user input."""
#     build_regular_heading(
#         "Test Your Model",
#         heading_level=2
#     )

#     st.write("Use this section to test the model with your own input values.")
    
#     # Create input form
#     with st.form(key="model_test_form"):
#         st.write("Enter values for the features used in training:")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             build_label_with_info(
#                 "Glucose", 
#                 "Plasma glucose concentration (mg/dL). Normal range is typically 70-99 mg/dL fasting."
#             )
#             glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=120.0, step=1.0)
            
#             build_label_with_info(
#                 "BMI", 
#                 "Body Mass Index (weight in kg/(height in m)²). Normal range is typically 18.5-24.9."
#             )
#             bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0, step=0.1)
            
#         with col2:
#             build_label_with_info(
#                 "Age", 
#                 "Age in years."
#             )
#             age = st.number_input("Age (years)", min_value=0, max_value=120, value=35, step=1)
            
#             build_label_with_info(
#                 "Diabetes Pedigree Function", 
#                 "A function that scores likelihood of diabetes based on family history. Higher values indicate greater risk."
#             )
#             diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        
#         submit_button = st.form_submit_button(label="Predict Diabetes Risk")
    
#     # Process the form submission
#     if submit_button:
#         # Try to load the model
#         try:
#             model_path = "super_learner_model.pkl"
            
#             if os.path.exists(model_path):
#                 # Load the trained model
#                 loaded_model = joblib.load(model_path)
                
#                 # Prepare input data
#                 input_data = np.array([[glucose, bmi, age, diabetes_pedigree]])
                
#                 # Scale the input data using the same scaler used for training
#                 # First, check if df_clean_output.csv exists
#                 if os.path.exists('df_clean_output.csv'):
#                     df_clean = pd.read_csv('df_clean_output.csv')
#                     scaler = MinMaxScaler()
#                     scaler.fit(df_clean[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']])
#                     input_data_scaled = scaler.transform(input_data)
#                 else:
#                     # If the file doesn't exist, assume the input is already in the right range (0-1)
#                     # or create a simple scaling based on typical ranges
#                     st.warning("Training data file not found. Using approximate scaling.")
                    
#                     # Approximate scaling based on typical ranges
#                     glucose_scaled = glucose / 200.0  # Assuming max glucose is around 200
#                     bmi_scaled = bmi / 50.0  # Assuming max BMI is around 50
#                     age_scaled = age / 100.0  # Assuming max age is around 100
#                     pedigree_scaled = diabetes_pedigree / 2.0  # Assuming max pedigree is around 2
                    
#                     input_data_scaled = np.array([[glucose_scaled, bmi_scaled, age_scaled, pedigree_scaled]])
                
#                 # Make prediction
#                 with st.spinner("Making prediction..."):
#                     # Check if it's a SuperLearner model
#                     is_super_learner = hasattr(loaded_model, 'meta_learner_')
                    
#                     if is_super_learner:
#                         # Handle SuperLearner prediction 
#                         try:
#                             # For SuperLearner, extract and use individual base models
#                             base_preds = []
                            
#                             # Get predictions from all base learners
#                             for i, learner in enumerate(loaded_model.learners_):
#                                 try:
#                                     if hasattr(learner, 'predict_proba'):
#                                         pred = learner.predict_proba(input_data_scaled)
#                                         if pred.ndim > 1 and pred.shape[1] > 1:
#                                             base_preds.append(pred[:, 1])
#                                         else:
#                                             base_preds.append(pred)
#                                     else:
#                                         pred = learner.predict(input_data_scaled)
#                                         base_preds.append(pred)
#                                 except Exception as e:
#                                     st.warning(f"Base learner {i} failed: {str(e)}")
#                                     continue
                            
#                             # If we have base learner predictions, use the meta learner to combine them
#                             if base_preds and len(base_preds) > 0:
#                                 # Combine the base predictions (as would happen in the meta-learner)
#                                 base_preds_array = np.column_stack(base_preds)
                                
#                                 # Use the meta-learner to get the final prediction
#                                 if hasattr(loaded_model, 'meta_learner_') and loaded_model.meta_learner_ is not None:
#                                     meta_learner = loaded_model.meta_learner_
#                                     if hasattr(meta_learner, 'predict_proba'):
#                                         meta_pred = meta_learner.predict_proba(base_preds_array)
#                                         diabetes_probability = meta_pred[0, 1] * 100
#                                     else:
#                                         meta_pred = meta_learner.predict(base_preds_array)
#                                         diabetes_probability = float(meta_pred[0]) * 100
#                                 else:
#                                     # If no meta-learner is available, use a simple average
#                                     avg_pred = np.mean(base_preds_array)
#                                     diabetes_probability = float(avg_pred[0]) * 100
#                             else:
#                                 raise ValueError("No base learner predictions available")
                                
#                         except Exception as e:
#                             st.warning(f"SuperLearner prediction failed: {str(e)}")
#                             # Fall back to direct ensemble prediction for VotingClassifier
#                             try:
#                                 if hasattr(loaded_model, 'estimators_'):  # Check if it's a VotingClassifier
#                                     est_preds = []
#                                     for name, est in loaded_model.estimators_:
#                                         if hasattr(est, 'predict_proba'):
#                                             est_pred = est.predict_proba(input_data_scaled)
#                                             if est_pred.shape[1] > 1:
#                                                 est_preds.append(est_pred[:, 1])
#                                             else:
#                                                 est_preds.append(est_pred)
#                                     diabetes_probability = np.mean(est_preds) * 100
#                                 else:
#                                     raise ValueError("Model is not a recognized ensemble type")
#                             except:
#                                 # If all else fails, try each model in get_models() directly
#                                 models = get_models()
#                                 model_preds = []
#                                 for name, model in models.items():
#                                     try:
#                                         model.fit(X_train_sc, y_train_sc)  # This assumes X_train_sc is available
#                                         pred = model.predict_proba(input_data_scaled)
#                                         model_preds.append(pred[:, 1])
#                                     except:
#                                         pass
#                                 if model_preds:
#                                     diabetes_probability = np.mean(model_preds) * 100
#                                 else:
#                                     raise ValueError("No models could make predictions")
#                     else:
#                         # Standard scikit-learn model
#                         if hasattr(loaded_model, 'predict_proba'):
#                             prediction_proba = loaded_model.predict_proba(input_data_scaled)
                            
#                             # Handle the case where prediction_proba is one-dimensional
#                             if prediction_proba.ndim == 1:
#                                 prediction_proba = np.column_stack((1 - prediction_proba, prediction_proba))
                            
#                             diabetes_probability = prediction_proba[0, 1] * 100  # Convert to percentage
#                         else:
#                             # If the model can't predict probabilities, use binary prediction
#                             prediction = loaded_model.predict(input_data_scaled)
#                             diabetes_probability = 100.0 if prediction[0] > 0.5 else 0.0
#                             st.warning("This model doesn't provide probability estimates. Using binary prediction.")
                
#                 # Display results
#                 st.subheader("Prediction Results")
                
#                 # Create a gauge-like visualization
#                 fig, ax = plt.subplots(figsize=(8, 3))
                
#                 # Create a horizontal bar representing the risk spectrum
#                 ax.barh(['Risk'], [100], color='lightgray', height=0.6)
#                 ax.barh(['Risk'], [diabetes_probability], color='#ff9999' if diabetes_probability > 50 else '#66b3ff', height=0.6)
                
#                 # Add a vertical line for the threshold
#                 ax.axvline(x=50, color='red', linestyle='--', alpha=0.7)
                
#                 # Add text showing the exact probability
#                 ax.text(diabetes_probability, 0, f' {diabetes_probability:.1f}%', 
#                         va='center', ha='left' if diabetes_probability < 50 else 'right',
#                         fontweight='bold', fontsize=14, color='black')
                
#                 # Customize the plot
#                 ax.set_xlim(0, 100)
#                 ax.set_xlabel('Diabetes Risk (%)')
#                 ax.set_yticks([])
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.spines['left'].set_visible(False)
                
#                 st.pyplot(fig)
                
#                 # Risk interpretation
#                 if diabetes_probability < 20:
#                     risk_level = "Low"
#                     color = "green"
#                     message = "Based on the provided features, the model predicts a low risk of diabetes."
#                 elif diabetes_probability < 50:
#                     risk_level = "Moderate"
#                     color = "orange"
#                     message = "Based on the provided features, the model predicts a moderate risk of diabetes."
#                 else:
#                     risk_level = "High"
#                     color = "red"
#                     message = "Based on the provided features, the model predicts a high risk of diabetes."
                
#                 st.markdown(f"<h3 style='color: {color}'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
#                 st.write(message)
                
#                 st.warning("Remember that this is just a prediction based on limited data. Always consult with healthcare professionals for proper medical advice.")
                
#                 # Feature importance visualization (relative to the input)
#                 st.subheader("Feature Analysis")
#                 st.write("How each feature contributes to the prediction:")
                
#                 # Compare with population averages (these are fictional - replace with actual values)
#                 avg_glucose = 110.0
#                 avg_bmi = 26.5
#                 avg_age = 40.0
#                 avg_pedigree = 0.4
                
#                 fig, ax = plt.subplots(figsize=(10, 6))
                
#                 features = ['Glucose', 'BMI', 'Age', 'Diabetes Pedigree']
#                 user_values = [glucose, bmi, age, diabetes_pedigree]
#                 avg_values = [avg_glucose, avg_bmi, avg_age, avg_pedigree]
                
#                 x = np.arange(len(features))
#                 width = 0.35
                
#                 ax.bar(x - width/2, user_values, width, label='Your Values', color='#66b3ff')
#                 ax.bar(x + width/2, avg_values, width, label='Population Average', color='#c2c2d6')
                
#                 ax.set_ylabel('Value')
#                 ax.set_title('Your Values vs. Population Average')
#                 ax.set_xticks(x)
#                 ax.set_xticklabels(features)
#                 ax.legend()
                
#                 # Add percentage differences
#                 for i, (user, avg) in enumerate(zip(user_values, avg_values)):
#                     diff_pct = ((user - avg) / avg) * 100
#                     if abs(diff_pct) >= 5:  # Only show significant differences
#                         ax.text(i, max(user, avg) + 1, f"{diff_pct:+.1f}%", 
#                                 ha='center', va='bottom', 
#                                 color='green' if diff_pct < 0 and features[i] != 'Age' else 'red' if diff_pct > 0 and features[i] != 'Age' else 'black')
                
#                 st.pyplot(fig)
                
#             else:
#                 st.error("No model file found. Please train and save the model first.")
        
#         except Exception as e:
#             st.error(f"Error loading or using the model: {str(e)}")
#             import traceback
#             st.code(traceback.format_exc())
#             st.info("Make sure you've trained and saved the model before testing it.")


def test_model():
    """Test the trained model with user input."""
    build_regular_heading(
        "Test Your Model",
        heading_level=2
    )

    st.write("Use this section to test the model with your own input values.")
    
    # Create input form
    with st.form(key="model_test_form"):
        st.write("Enter values for the features used in training:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            build_label_with_info(
                "Glucose", 
                "Plasma glucose concentration (mg/dL). Normal range is typically 70-99 mg/dL fasting."
            )
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=120.0, step=1.0)
            
            build_label_with_info(
                "BMI", 
                "Body Mass Index (weight in kg/(height in m)²). Normal range is typically 18.5-24.9."
            )
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0, step=0.1)
            
        with col2:
            build_label_with_info(
                "Age", 
                "Age in years."
            )
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=35, step=1)
            
            build_label_with_info(
                "Diabetes Pedigree Function", 
                "A function that scores likelihood of diabetes based on family history. Higher values indicate greater risk."
            )
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        
        submit_button = st.form_submit_button(label="Predict Diabetes Risk")
    
    # Process the form submission
    if submit_button:
        try:
            model_path = "super_learner_model.pkl"
            
            if os.path.exists(model_path):
                # Load the trained model
                loaded_model = joblib.load(model_path)
                
                # Prepare input data
                input_data = np.array([[glucose, bmi, age, diabetes_pedigree]])
                
                # Scale the input data
                if os.path.exists('df_clean_output.csv'):
                    df_clean = pd.read_csv('df_clean_output.csv')
                    scaler = MinMaxScaler()
                    scaler.fit(df_clean[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']])
                    input_data_scaled = scaler.transform(input_data)
                else:
                    # Use approximate scaling based on typical ranges
                    st.warning("Training data file not found. Using approximate scaling.")
                    glucose_scaled = glucose / 200.0
                    bmi_scaled = bmi / 50.0
                    age_scaled = age / 100.0
                    pedigree_scaled = diabetes_pedigree / 2.0
                    input_data_scaled = np.array([[glucose_scaled, bmi_scaled, age_scaled, pedigree_scaled]])
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    # Special handling for mlens SuperLearner
                    if hasattr(loaded_model, 'learners_') and hasattr(loaded_model, 'meta_learner_'):
                        st.info("Detected mlens SuperLearner model - using direct component access")
                        
                        # Get predictions from base learners
                        base_predictions = []
                        for i, learner in enumerate(loaded_model.learners_):
                            try:
                                if hasattr(learner, 'predict_proba'):
                                    pred = learner.predict_proba(input_data_scaled)
                                    # Extract probability for positive class
                                    if pred.ndim > 1 and pred.shape[1] > 1:
                                        base_predictions.append(pred[:, 1])
                                    else:
                                        base_predictions.append(pred)
                                else:
                                    # If no predict_proba, use predict
                                    pred = learner.predict(input_data_scaled)
                                    base_predictions.append(pred)
                            except Exception as inner_e:
                                st.warning(f"Base learner {i} prediction failed: {str(inner_e)}")
                        
                        if not base_predictions:
                            raise ValueError("All base learners failed to produce predictions")
                        
                        # Transform to what meta_learner expects
                        base_predictions = np.hstack([p.reshape(-1, 1) for p in base_predictions])
                        
                        # Use meta learner directly
                        meta_learner = loaded_model.meta_learner_
                        try:
                            if hasattr(meta_learner, 'predict_proba'):
                                proba = meta_learner.predict_proba(base_predictions)
                                # Extract probability for positive class
                                if proba.ndim > 1 and proba.shape[1] > 1:
                                    diabetes_probability = proba[0, 1] * 100
                                else:
                                    diabetes_probability = proba[0] * 100
                            else:
                                pred = meta_learner.predict(base_predictions)
                                diabetes_probability = pred[0] * 100
                        except Exception as meta_e:
                            st.warning(f"Meta-learner prediction failed: {str(meta_e)}. Using average of base predictions.")
                            # Average the base predictions if meta learner fails
                            diabetes_probability = np.mean(base_predictions) * 100
                    
                    # For regular scikit-learn models and ensembles
                    elif hasattr(loaded_model, 'estimators_'):  # VotingClassifier
                        st.info("Detected VotingClassifier - using component models directly")
                        
                        # Use the estimators directly
                        predictions = []
                        for name, estimator in loaded_model.estimators_:
                            try:
                                if hasattr(estimator, 'predict_proba'):
                                    pred = estimator.predict_proba(input_data_scaled)
                                    if pred.ndim > 1 and pred.shape[1] > 1:
                                        predictions.append(pred[:, 1][0])
                                    else:
                                        predictions.append(pred[0])
                                else:
                                    pred = estimator.predict(input_data_scaled)
                                    predictions.append(pred[0])
                            except Exception as e:
                                st.warning(f"Estimator {name} failed: {str(e)}")
                        
                        if not predictions:
                            raise ValueError("All component models failed to produce predictions")
                            
                        # Average the predictions (similar to 'soft' voting)
                        diabetes_probability = np.mean(predictions) * 100
                    
                    # Fall back to recreating models from get_models()
                    else:
                        st.info("Using individual models from model library")
                        
                        # Get the X_train_sc and y_train_sc for training if available
                        if 'X_train_sc' in globals() and 'y_train_sc' in globals():
                            X_train = globals()['X_train_sc']
                            y_train = globals()['y_train_sc']
                        else:
                            # If not available in globals, load from the data function
                            X_train, _, y_train, _ = load_and_preprocess_data()
                        
                        # Use individual models from get_models()
                        models = get_models()
                        predictions = []
                        
                        for name, model in models.items():
                            try:
                                # Train the model
                                model.fit(X_train, y_train)
                                
                                # Make prediction
                                if hasattr(model, 'predict_proba'):
                                    pred = model.predict_proba(input_data_scaled)
                                    if pred.shape[1] > 1:
                                        predictions.append(pred[0, 1])
                                    else:
                                        predictions.append(pred[0])
                                else:
                                    pred = model.predict(input_data_scaled)
                                    predictions.append(float(pred[0]))
                                
                                st.success(f"Model {name} prediction successful")
                            except Exception as e:
                                st.warning(f"Model {name} failed: {str(e)}")
                        
                        if not predictions:
                            raise ValueError("All models failed to produce predictions")
                            
                        # Average the predictions
                        diabetes_probability = np.mean(predictions) * 100
                
                # Display results
                st.subheader("Prediction Results")
                
                # Create a gauge-like visualization
                fig, ax = plt.subplots(figsize=(8, 3))
                
                # Create a horizontal bar representing the risk spectrum
                ax.barh(['Risk'], [100], color='lightgray', height=0.6)
                ax.barh(['Risk'], [diabetes_probability], color='#ff9999' if diabetes_probability > 50 else '#66b3ff', height=0.6)
                
                # Add a vertical line for the threshold
                ax.axvline(x=50, color='red', linestyle='--', alpha=0.7)
                
                # Add text showing the exact probability
                ax.text(diabetes_probability, 0, f' {diabetes_probability:.1f}%', 
                        va='center', ha='left' if diabetes_probability < 50 else 'right',
                        fontweight='bold', fontsize=14, color='black')
                
                # Customize the plot
                ax.set_xlim(0, 100)
                ax.set_xlabel('Diabetes Risk (%)')
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                st.pyplot(fig)
                
                # Risk interpretation
                if diabetes_probability < 20:
                    risk_level = "Low"
                    color = "green"
                    message = "Based on the provided features, the model predicts a low risk of diabetes."
                elif diabetes_probability < 50:
                    risk_level = "Moderate"
                    color = "orange"
                    message = "Based on the provided features, the model predicts a moderate risk of diabetes."
                else:
                    risk_level = "High"
                    color = "red"
                    message = "Based on the provided features, the model predicts a high risk of diabetes."
                
                st.markdown(f"<h3 style='color: {color}'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
                st.write(message)
                
                st.warning("Remember that this is just a prediction based on limited data. Always consult with healthcare professionals for proper medical advice.")
                
                # Feature importance visualization (relative to the input)
                st.subheader("Feature Analysis")
                st.write("How each feature contributes to the prediction:")
                
                # Compare with population averages (these are fictional - replace with actual values)
                avg_glucose = 110.0
                avg_bmi = 26.5
                avg_age = 40.0
                avg_pedigree = 0.4
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                features = ['Glucose', 'BMI', 'Age', 'Diabetes Pedigree']
                user_values = [glucose, bmi, age, diabetes_pedigree]
                avg_values = [avg_glucose, avg_bmi, avg_age, avg_pedigree]
                
                x = np.arange(len(features))
                width = 0.35
                
                ax.bar(x - width/2, user_values, width, label='Your Values', color='#66b3ff')
                ax.bar(x + width/2, avg_values, width, label='Population Average', color='#c2c2d6')
                
                ax.set_ylabel('Value')
                ax.set_title('Your Values vs. Population Average')
                ax.set_xticks(x)
                ax.set_xticklabels(features)
                ax.legend()
                
                # Add percentage differences
                for i, (user, avg) in enumerate(zip(user_values, avg_values)):
                    diff_pct = ((user - avg) / avg) * 100
                    if abs(diff_pct) >= 5:  # Only show significant differences
                        ax.text(i, max(user, avg) + 1, f"{diff_pct:+.1f}%", 
                                ha='center', va='bottom', 
                                color='green' if diff_pct < 0 and features[i] != 'Age' else 'red' if diff_pct > 0 and features[i] != 'Age' else 'black')
                
                st.pyplot(fig)
                
            else:
                st.error("No model file found. Please train and save the model first.")
        
        except Exception as e:
            st.error(f"Error loading or using the model: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.info("Make sure you've trained and saved the model before testing it.")
            
            
if __name__ == "__main__":
    main()
    test_model()