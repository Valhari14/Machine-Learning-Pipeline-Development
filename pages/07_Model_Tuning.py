import random
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define component builders - this approach properly renders HTML
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

# Custom CSS for checkbox styling
st.markdown("""
    <style>
        /* Style the checkbox input */
        div[data-testid="stCheckbox"] > div {
            display: flex;
            align-items: center;
        }

        /* Add some padding and margin around the checkbox */
        div[data-testid="stCheckbox"] {
            margin-bottom: 15px;
            padding: 5px;
            border: 2px solid #007BFF;
            border-radius: 5px;
            background-color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

# Check if required session states are available
if 'df_final' in st.session_state and 'selected_models' in st.session_state and 'selected_features' in st.session_state and 'target_column' in st.session_state:

    # Retrieve session state variables
    df = st.session_state['df_final']
    selected_models = st.session_state['selected_models']
    selected_features = st.session_state['selected_features']
    target_column = st.session_state['target_column']
    
    # Display the selected models with tooltip
    build_title_with_info(
        "Selected Models for Hyperparameter Tuning", 
        "These are the machine learning models you have chosen to optimize with different parameter settings."
    )
    st.write(selected_models)

    # Provide user the choice for Auto-tuning or Manual-tuning using checkboxes with tooltips
    build_title_with_info(
        "Hyperparameter Tuning Methods", 
        "Different approaches to find the best parameters for your machine learning models."
    )
    
    build_label_with_info(
        "Auto Tuning Hyperparameters", 
        "Automatically searches for the best parameters using grid search or randomized search algorithms."
    )
    auto_tuning = st.checkbox("Auto Tuning")
    
    build_label_with_info(
        "Manual Tuning Hyperparameters", 
        "Allows you to manually set and test different parameter values for each model."
    )
    manual_tuning = st.checkbox("Manual Tuning")

    # Ensure only one can be selected
    if auto_tuning and manual_tuning:
        st.warning("Please select only one tuning method: Auto-tuning or Manual-tuning.")
        auto_tuning = False  # Reset auto_tuning if both are selected
        manual_tuning = True  # Keep manual_tuning

    model_dict = {
        "Logistic Regression": LogisticRegression(),  
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Extra Trees": ExtraTreesClassifier()
    }

    # Dictionary to store evaluation metrics
    evaluation_metrics = {}

    if auto_tuning:
        build_title_with_info(
            "Auto-Tuning Hyperparameters", 
            "The system will automatically search for optimal parameters using cross-validation techniques."
        )

        # Setting the random seed for reproducibility
        SEED = 42
        
        # Dictionary to store the best parameters for each model
        best_params = {}

        # Iterate over selected models and perform tuning
        for model_name in selected_models:
            st.write(f"Tuning: {model_name}")
            model = model_dict[model_name]

            if model_name == "Logistic Regression":
                param_grid = {'C': [0.1, 1.0, 10], 'penalty': ['l2']}
                randomized_search = RandomizedSearchCV(model, param_grid, n_iter=100, cv=5, scoring='roc_auc', random_state=SEED)
                randomized_search.fit(df[selected_features], df[target_column])
                model = randomized_search.best_estimator_
                best_params[model_name] = randomized_search.best_params_

            elif model_name == "Decision Tree":
                param_grid = {'max_depth': [3, 5, 7, None], 'criterion': ['gini', 'entropy']}
                randomized_search = RandomizedSearchCV(model, param_grid, n_iter=100, cv=5, scoring='roc_auc', random_state=SEED)
                randomized_search.fit(df[selected_features], df[target_column])
                model = randomized_search.best_estimator_
                best_params[model_name] = randomized_search.best_params_

            else:
                param_grid = {}
                if model_name == "K-Nearest Neighbors":
                    param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
                elif model_name == "Support Vector Machine":
                    param_grid = {'C': [0.1, 1.0, 10], 'kernel': ['linear', 'rbf', 'poly']}
                elif model_name == "AdaBoost":
                    param_grid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1.0]}
                elif model_name == "Gradient Boosting":
                    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
                elif model_name == "Random Forest":
                    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
                elif model_name == "Extra Trees":
                    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}

                # Perform hyperparameter tuning using GridSearchCV
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
                grid_search.fit(df[selected_features], df[target_column])
                model = grid_search.best_estimator_
                best_params[model_name] = grid_search.best_params_

        # Display all best parameters at once
        build_title_with_info(
            "Best Hyperparameters for Each Model", 
            "The optimal parameter values found during auto-tuning process for each selected model."
        )
        for model_name, params in best_params.items():
            st.write(f"{model_name}: {params}")

        # Evaluate models with best parameters
        build_title_with_info(
            "Evaluation Metrics for Tuned Models", 
            "Performance measurements showing how well each model performs with its optimal parameters."
        )
        cols = st.columns(4)  # Create 4 columns for layout
        plot_index = 0  # Track which column to place the plot in

        # Create a list to store the classification reports for later display
        reports = {}

        for model_name, params in best_params.items():
            model = model_dict[model_name].set_params(**params)

            # Fit the model
            X = df[selected_features]
            y = df[target_column]
            model.fit(X, y)

            # Predictions
            y_pred = model.predict(X)

            # Evaluation
            report = classification_report(y, y_pred, output_dict=True)
            evaluation_metrics[model_name] = report
            reports[model_name] = pd.DataFrame(report).transpose()  # Store the report for later

            # Plot confusion matrix
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(3, 4))  # Adjust the figure size here
            sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', square=True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            # Display the plot in the appropriate column
            with cols[plot_index]:
                build_title_with_info(
                    f"{model_name} Confusion Matrix", 
                    "Visual representation of predicted vs. actual values showing true positives, false positives, true negatives, and false negatives.",
                    heading_level=4
                )
                st.pyplot(fig)

            plot_index += 1  # Move to the next column
            if plot_index >= 4:  # Reset after three plots
                plot_index = 0
                cols = st.columns(4)  # Reset columns for new row

        # Now display all classification reports in a single section
        build_title_with_info(
            "Classification Reports", 
            "Detailed metrics including precision, recall, F1-score, and support for each class in the prediction task."
        )
        for model_name, report_df in reports.items():
            build_title_with_info(
                f"Classification Report for {model_name}", 
                "Breakdown of model performance metrics for each class and overall averages.",
                heading_level=4
            )
            st.table(report_df)

    if manual_tuning:
        build_title_with_info(
            "Manual-Tuning Hyperparameters", 
            "Allows you to manually set specific parameter values for each model to test different configurations."
        )

        # Create a list to store the confusion matrices for later display
        confusion_matrices = []
        reports = {}

        # Iterate over selected models for manual tuning
        for model_name in selected_models:
            build_title_with_info(
                f"Tune Hyperparameters for {model_name}", 
                "Adjust the parameters below to customize how this model will learn from your data.",
                heading_level=4
            )

            # Define manual hyperparameter widgets based on model
            if model_name == "Logistic Regression":
                build_label_with_info(
                    "C (Regularization Strength)", 
                    "Controls the strength of regularization. Lower values mean stronger regularization which helps prevent overfitting."
                )
                C = st.number_input("C", min_value=0.01, max_value=10.0, value=1.0, key=f"log_reg_C")
                
                build_label_with_info(
                    "Penalty", 
                    "Type of regularization to apply. L2 is standard ridge regression, while L1 is lasso regression which can lead to sparse models."
                )
                penalty = st.selectbox("Penalty", options=['l2', 'l1'], key=f"log_reg_penalty")
                
                tuned_model = LogisticRegression(C=C, penalty=penalty)
            
            elif model_name == "K-Nearest Neighbors":
                build_label_with_info(
                    "Number of Neighbors", 
                    "The number of nearest neighbors to consider when classifying a new data point. Higher values reduce noise but may miss local patterns."
                )
                n_neighbors = st.slider("Neighbors", min_value=1, max_value=20, value=5, key=f"knn_neighbors")
                
                build_label_with_info(
                    "Weights", 
                    "How to weight the votes from neighbors. Uniform gives equal weight, while distance weights closer neighbors more heavily."
                )
                weights = st.selectbox("Weights", ['uniform', 'distance'], key=f"knn_weights")
                
                tuned_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

            elif model_name == "Decision Tree":
                build_label_with_info(
                    "Max Depth", 
                    "Maximum depth of the decision tree. Deeper trees can model more complex patterns but risk overfitting."
                )
                max_depth = st.slider("Depth", min_value=1, max_value=10, value=3, key=f"dt_max_depth")
                
                build_label_with_info(
                    "Criterion", 
                    "The function used to measure the quality of a split. Gini measures impurity, while entropy uses information gain."
                )
                criterion = st.selectbox("Criterion", ['gini', 'entropy'], key=f"dt_criterion")
                
                tuned_model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

            elif model_name == "Gaussian Naive Bayes":
                st.info("Gaussian Naive Bayes has no hyperparameters to tune.")
                tuned_model = GaussianNB()  # No hyperparameters to tune

            elif model_name == "Support Vector Machine":
                build_label_with_info(
                    "C (Regularization Strength)", 
                    "Controls the trade-off between having a smooth decision boundary and classifying training points correctly. Lower values create a smoother decision boundary."
                )
                C = st.number_input("C", min_value=0.01, max_value=10.0, value=1.0, key=f"svc_C")
                
                build_label_with_info(
                    "Kernel", 
                    "The kernel function used to transform the data into a higher-dimensional space. Different kernels work better for different types of data relationships."
                )
                kernel = st.selectbox("Kernel", options=['linear', 'rbf', 'poly'], key=f"svc_kernel")
                
                tuned_model = SVC(C=C, kernel=kernel, probability=True)

            elif model_name == "AdaBoost":
                build_label_with_info(
                    "Number of Estimators", 
                    "The number of weak learners (decision trees) to train. More estimators can improve performance but increase computation time."
                )
                n_estimators = st.slider("Estimators", min_value=50, max_value=500, value=100, key=f"ada_n_estimators")
                
                build_label_with_info(
                    "Learning Rate", 
                    "Controls how much each weak learner contributes to the final model. Lower values require more estimators but often yield better performance."
                )
                learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, key=f"ada_learning_rate")
                
                tuned_model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

            elif model_name == "Gradient Boosting":
                build_label_with_info(
                    "Number of Estimators", 
                    "The number of boosting stages to perform. More trees help reduce errors but can lead to overfitting and longer training times."
                )
                n_estimators = st.slider("Estimators", min_value=100, max_value=1000, value=200, key=f"gb_n_estimators")
                
                build_label_with_info(
                    "Learning Rate", 
                    "Shrinks the contribution of each tree, helping to prevent overfitting. Lower rates require more estimators but often result in better accuracy."
                )
                learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, key=f"gb_learning_rate")
                
                tuned_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

            elif model_name == "Random Forest":
                build_label_with_info(
                    "Number of Estimators", 
                    "The number of trees in the forest. More trees generally increase performance but also increase computation time."
                )
                n_estimators = st.slider("Estimators", min_value=100, max_value=500, value=200, key=f"rf_n_estimators")
                
                build_label_with_info(
                    "Max Depth", 
                    "Maximum depth of each tree in the forest. Deeper trees can model more complex patterns but may overfit."
                )
                max_depth = st.slider("Depth", min_value=1, max_value=20, value=10, key=f"rf_max_depth")
                
                tuned_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

            elif model_name == "Extra Trees":
                build_label_with_info(
                    "Number of Estimators", 
                    "The number of trees in the forest. Similar to Random Forest but with additional randomization in the tree building process."
                )
                n_estimators = st.slider("Estimators", min_value=100, max_value=500, value=200, key=f"et_n_estimators")
                
                build_label_with_info(
                    "Max Depth", 
                    "Maximum depth of each tree. Controls how detailed the decision rules become in each tree."
                )
                max_depth = st.slider("Depth", min_value=1, max_value=20, value=10, key=f"et_max_depth")
                
                tuned_model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth)

            # Fit the tuned model
            X = df[selected_features]
            y = df[target_column]
            tuned_model.fit(X, y)

            # Predictions
            y_pred = tuned_model.predict(X)

            # Evaluation
            report = classification_report(y, y_pred, output_dict=True)
            evaluation_metrics[model_name] = report
            reports[model_name] = pd.DataFrame(report).transpose()  # Store the report for later

            # Collect confusion matrix data
            cm = confusion_matrix(y, y_pred)
            confusion_matrices.append((model_name, cm))  # Store the model name and confusion matrix for later

        # Display the confusion matrices in a grid format
        build_title_with_info(
            "Confusion Matrices", 
            "Visual representations showing the count of correct and incorrect predictions for each class. Helps identify where models are making mistakes."
        )
        cols = st.columns(4)  # Create 4 columns for layout
        plot_index = 0  # Track which column to place the plot in

        for model_name, cm in confusion_matrices:
            fig, ax = plt.subplots(figsize=(3, 4))  # Adjust the figure size here
            sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', square=True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            # Display the plot in the appropriate column
            with cols[plot_index]:
                build_title_with_info(
                    f"{model_name}", 
                    "Confusion matrix showing prediction accuracy for this model with your selected parameters.",
                    heading_level=4
                )
                st.pyplot(fig)

            plot_index += 1  # Move to the next column
            if plot_index >= 4:  # Reset after four plots
                plot_index = 0
                cols = st.columns(4)  # Reset columns for new row

        # Now display all classification reports in a single section
        build_title_with_info(
            "Classification Reports", 
            "Detailed performance metrics for each model showing precision, recall, F1-score and other important evaluation metrics."
        )
        for model_name, report_df in reports.items():
            build_title_with_info(
                f"Classification Report for {model_name}", 
                "Breakdown of model accuracy by class with metrics including precision (avoids false positives), recall (avoids false negatives), and F1-score (balance of both).",
                heading_level=4
            )
            st.table(report_df)

else:
    st.error("Required session state variables are not found.")