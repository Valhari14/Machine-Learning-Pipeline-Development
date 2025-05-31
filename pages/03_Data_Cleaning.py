from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np

# Set page to open in wide mode
st.set_page_config(layout="wide")

# Custom CSS to style the download button and checkbox
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
    div[data-testid="stDownloadButton"] > button {
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #007BFF;
        border: 2px solid #007BFF;
        border-radius: 5px;
        cursor: pointer;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #0056b3;
    }
    div[data-testid="stDownloadButton"] {
        margin-bottom: 15px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to drop selected columns
def drop_columns(df, columns_to_drop):
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

# Function to detect and remove outliers using Tukey's method
def TukeyOutliers(df_out, nameOfFeature):
    valueOfFeature = df_out[nameOfFeature]
    Q1 = np.percentile(valueOfFeature, 25.)
    Q3 = np.percentile(valueOfFeature, 75.)
    step = (Q3 - Q1) * 1.5
    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    return outliers, len(outliers)

# Function to detect outliers using Z-score method
def ZScoreOutliers(df_out, nameOfFeature):
    valueOfFeature = df_out[nameOfFeature]
    mean = np.mean(valueOfFeature)
    std = np.std(valueOfFeature)
    threshold = 3  # Z-score threshold (3 is common, but can be adjusted)
    z_scores = (valueOfFeature - mean) / std
    outliers = valueOfFeature[abs(z_scores) > threshold].index.tolist()
    return outliers, len(outliers)

# Random Sample Imputation Function
def random_sample_imputation(df, feature):
    df_missing = df[df[feature].isnull()]
    df_non_missing = df[df[feature].notnull()]
    df_missing[feature] = df_missing[feature].apply(lambda x: np.random.choice(df_non_missing[feature]))
    df.loc[df[feature].isnull(), feature] = df_missing[feature]
    return df

# Mean Imputation Function
def mean_imputation(df, feature):
    mean_value = df[feature].mean()
    df[feature].fillna(mean_value, inplace=True)
    return df

# Median Imputation Function
def median_imputation(df, feature):
    median_value = df[feature].median()
    df[feature].fillna(median_value, inplace=True)
    return df

# Remove Missing Values Function
def remove_missing_values(df, feature):
    df = df[df[feature].notnull()]
    return df

# Replace with 0 Function
def replace_with_zero(df, feature):
    df[feature].fillna(0, inplace=True)
    return df

# Check if the dataset is available in the session state
if 'df' in st.session_state:
    df = st.session_state['df']
    df_modified = False  # Flag to track if the data is modified
    st.title("Data Preparation")
    st.write("Data preview:", df.head())
    
    # Drop columns feature
    if st.checkbox("Drop Columns"):
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        if columns_to_drop:
            df = drop_columns(df, columns_to_drop)
            st.session_state['df'] = df  # Update session state
            st.session_state['df_final'] = df  # Store updated dataframe only after button click
            df_modified = True
            st.write("Selected columns successfully dropped.")

        # Refresh available columns after drop operation
        available_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_features = df.columns[df.isnull().any()].tolist()

        # Ensure dropped columns are removed from all selections
        numeric_cols = [col for col in numeric_cols if col in available_columns]
        missing_features = [col for col in missing_features if col in available_columns]

        # Store updated dataframe in session state
        st.session_state['df'] = df
        st.session_state['df_final'] = df  # Store updated dataframe only after button click


    # Outlier removal with method selection
    if st.checkbox("Remove Outliers"):
        df_t = df.copy()
        st.subheader("Select features")
        
        # Ensure only numerical columns are selectable
        numeric_cols = df_t.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect("Select numeric features to remove outliers from", numeric_cols)

        # Outlier Method Explanation
        st.subheader("Outlier Detection Methods")
        st.write("Outlier detection helps identify extreme values in the dataset. Below are two common methods:")

        st.markdown("""
        1. **Tukey's Method**: Uses the interquartile range (IQR) to detect outliers. It is robust to extreme values and identifies outliers outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        2. **Z-score Method**: Identifies outliers based on how many standard deviations a data point is from the mean. A common threshold is 3, meaning points more than 3 standard deviations away from the mean are flagged as outliers.
        """)
            # Fetch actual column names dynamically
        actual_columns = df_t.columns.tolist()
        
        # Generate reasoning based on dataset columns
        domain = "Healthcare" if "Pregnancies" in actual_columns else "Sales"
        # Suggest button to recommend an outlier detection method
        if st.button("Suggest"):
            if selected_features:
                # Determine and display the recommended method
                if domain == "Healthcare":
                    recommended_method = "Tukey's Method"
                    reasoning = """
                    Recommended Outlier Detection Method: Tukey's Method
                    
                    **Why?** The *Tukey Method* (based on the *Interquartile Range (IQR)*) is preferred for this dataset due to the following reasons:

                    #### ✅ **1. Robust to Skewed Distributions**
                    - Many medical datasets, including this one, often have **skewed distributions** rather than a normal (Gaussian) distribution.
                    - The **Z-score method assumes normality** and may **incorrectly classify values** as outliers if the data is skewed.
                    - The **IQR method does not assume normality**, making it more reliable for medical datasets.

                    #### ✅ **2. Effectively Identifies Extreme Values**
                    - The IQR method detects extreme values by measuring the **spread of the middle 50% of the data**.
                    - This ensures that **true outliers** are removed while **preserving valid variations**.

                    #### ✅ **3. Works Well for High-Variability Features**
                    - Features like **BMI, Insulin, and Blood Pressure** often exhibit **wide variations** in medical datasets.
                    - The IQR method effectively removes **only extreme deviations** while keeping meaningful data.

                    ❌ **Why NOT Z-score?**
                    - **Z-score assumes normality**, which is **not always valid** for medical datasets.
                    - It may classify **valid extreme values as outliers**, leading to data loss.
                    """
                else:
                    recommended_method = "Z-score Method"
                    reasoning = """
                    Recommended Outlier Detection Method: Z-score Method
                    
                    **Why?** The *Z-score method* is preferred for this dataset due to the following reasons:

                    #### ✅ **1. Assumes Normality & Works Well with Continuous Variables**
                    - The dataset contains numerical features such as **Income, Age, Children, and Cars**.
                    - Features like **Income and Age** generally follow a **near-normal distribution** in demographic datasets.
                    - The **Z-score method effectively detects outliers** in such distributions by measuring how far a data point deviates from the mean.

                    #### ✅ **2. Effectively Identifies Extreme Values Without Over-Filtering**
                    - The Z-score method detects outliers by considering **values that are more than 3 standard deviations from the mean**.
                    - This ensures that **only truly extreme values are removed**, rather than normal variations in data.
                    - The **IQR method can be too aggressive** in long-tailed distributions like Income, leading to excessive data loss.

                    #### ✅ **3. Best Suited for Income and Age Features**
                    - **Income distributions often have long tails**, meaning that some individuals may have significantly higher earnings.
                    - **IQR can mistakenly classify high-income individuals as outliers**, while the **Z-score method is better at distinguishing between true outliers and natural variations**.
                    - **Age typically follows a near-normal distribution**, making it a good candidate for Z-score detection.

                    ❌ **Why NOT Tukey's Method?**
                    - **Tukey's method can be too aggressive** in long-tailed distributions like **Income**, **removing valid high-income values**.
                    - It is **not ideal for normally distributed features** like **Age**, where Z-score provides a more statistically sound approach.
                    """

                # Display the recommendation in a visually appealing card
                st.markdown(
                    f"""
                    <div style="border-radius: 10px; padding: 15px; background-color: #f8f9fa; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);">
                        <h4 style="color: #2c3e50;">Recommended Outlier Removal Method: <b>{recommended_method}</b></h4>
                        <p style="font-size: 14px; color: #34495e;">{reasoning}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(f"**Recommended method: {recommended_method}**")


            else:
                st.write("Please select at least one feature to get a suggestion.")
                
                
        # Dropdown to select the outlier removal method
        outlier_method = st.selectbox("Select outlier removal method", ["None", "Tukey's Method", "Z-score Method"])

        if outlier_method != "None" and selected_features:
            outliers_indices = set()
            outlier_summary = []

            for feature in selected_features:
                if outlier_method == "Tukey's Method":
                    outliers, num_outliers = TukeyOutliers(df_t, feature)
                elif outlier_method == "Z-score Method":
                    outliers, num_outliers = ZScoreOutliers(df_t, feature)

                outliers_indices.update(outliers)
                outlier_summary.append(f"Feature '{feature}': {num_outliers} outliers found using {outlier_method}.")

            if outliers_indices:
                df_cleaned = df_t.drop(outliers_indices).reset_index(drop=True)
                st.write(f"Total number of outliers removed: {len(outliers_indices)}")
                st.write(f"New dataset has {df_cleaned.shape[0]} samples and {df_cleaned.shape[1]} features.")

                for summary in outlier_summary:
                    st.write(summary)

                st.session_state['df'] = df_cleaned
                df = df_cleaned
                df_modified = True
            
            else:
                st.write("No outliers detected across the selected features.")
            # Store updated dataframe in session state
            st.session_state['df'] = df
        elif outlier_method == "None":
            st.write("No outlier removal method selected. Please choose 'Tukey's Method' or 'Z-score Method' to proceed.")


    # Missing value handling
    if st.checkbox("Handle Missing Values"):
        
        if 'df' in st.session_state:
            df = st.session_state['df']
        missing_features = df.columns[df.isnull().any()].tolist()
        if missing_features:
            st.write(f"Missing values found in: {missing_features}")
            # **Handle Numerical Features**
            numerical_features = [col for col in missing_features if df[col].dtype.kind in 'if']
            if numerical_features:
                st.subheader(" Handle Missing Values in Numerical Features")

                # Multi-selection for numerical features
                selected_numerical = st.multiselect(
                    "Select numerical features",
                    numerical_features,
                    default=[]  # Ensure no default selection
                )

                if selected_numerical:
                    # Dropdown with default "Select Method"
                    method_num = st.selectbox(
                        "Choose method for numerical missing values",
                        ["Select Method", "Mean Imputation", "Median Imputation", "Random Sample Imputation", 
                         "Remove Missing Values", "Replace with 0"]
                    )

                    # Only apply the selected method if it's not "Select Method"
                    if method_num != "Select Method":
                        if method_num == "Mean Imputation":
                            for feature in selected_numerical:
                                df[feature].fillna(df[feature].mean(), inplace=True)
            

                        elif method_num == "Median Imputation":
                            for feature in selected_numerical:
                                df[feature].fillna(df[feature].median(), inplace=True)
                    

                        elif method_num == "Random Sample Imputation":
                            for feature in selected_numerical:
                                df_missing = df[df[feature].isnull()]
                                df_non_missing = df[df[feature].notnull()]
                                df.loc[df[feature].isnull(), feature] = np.random.choice(df_non_missing[feature])
                               

                        elif method_num == "Remove Missing Values":
                            df.dropna(subset=selected_numerical, inplace=True)
                            

                        elif method_num == "Replace with 0":
                            df[selected_numerical] = df[selected_numerical].fillna(0)
                          

                        # Confirmation message
                        if method_num != "Select Method":
                            st.success(f"✅ Missing values in selected numerical features handled using '{method_num}'!")

            # Store updated dataframe in session state
            st.session_state['df'] = df


           # **Handle Categorical Features**
            categorical_features = [col for col in missing_features if df[col].dtype == 'object']
            if categorical_features:
                st.subheader("Handle Missing Values in Categorical Features")

                # Multi-selection for categorical features
                selected_categorical = st.multiselect("Select categorical features", categorical_features)

                if selected_categorical:
                    # Choose method (default text instead of pre-selection)
                    method_cat = st.selectbox(
                        "Choose method for categorical missing values",
                        ["Select Method", "Random Sample Imputation", "Replace with Mode", "Replace with NaN", "Remove Missing Values"]
                    )

                    # Apply the selected method
                    if method_cat == "Random Sample Imputation":
                        for feature in selected_categorical:
                            df_missing = df[df[feature].isnull()]
                            df_non_missing = df[df[feature].notnull()]
                            df.loc[df[feature].isnull(), feature] = np.random.choice(df_non_missing[feature])

                    elif method_cat == "Replace with Mode":
                        for feature in selected_categorical:
                            mode_value = df[feature].mode()[0]
                            df[feature].fillna(mode_value, inplace=True)

                    elif method_cat == "Replace with NaN":
                        for feature in selected_categorical:
                            df[feature] = df[feature].astype("object")  # Ensure it stays categorical
                            df[feature].fillna(np.nan, inplace=True)  # Explicitly assign NaN

                    elif method_cat == "Remove Missing Values":
                        df.dropna(subset=selected_categorical, inplace=True)

                    # Confirmation message
                    if method_cat != "Select Method":
                        st.success(f"✅ Missing values in selected categorical features handled using '{method_cat}'!")

                # Store updated dataframe in session state
                st.session_state['df'] = df
                st.session_state['df_final'] = df  # Store updated dataframe only after button click

        else:
            st.info("✅ No missing values found in the dataset.")


    # Label Encoding
    if st.checkbox("Enable Label Encoding"):
        categorical_features = [col for col in df.columns if df[col].dtype == 'object']
        # Track if encoding was applied
        encoding_applied = False
        
        if categorical_features:
            st.info("Categorical variables detected! Choose features for Label Encoding.")
            selected_encode_features = st.multiselect("Select categorical features to encode", categorical_features)

            if selected_encode_features:
                # Button to apply encoding
                if st.button("Encode"):
                    label_encoders = {}  # Store label encoders for future use
                    for feature in selected_encode_features:
                        le = LabelEncoder()
                        df[feature] = le.fit_transform(df[feature])  # Perform Label Encoding
                        label_encoders[feature] = le  # Save the encoder for reference

                    st.session_state['df'] = df  # Store updated dataframe in session state
                    st.session_state['df_final'] = df  # Store final dataframe
                    st.success(f"✅ Label Encoding applied to selected features: {', '.join(selected_encode_features)}")
                    encoding_applied = True
                    
                    # Add a download button for the encoded data
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Encoded Data",
                        data=csv,
                        file_name="encoded_data.csv",
                        mime="text/csv",
                    )
                    st.info("Click the button above to download your encoded dataset.")
            else:
                st.warning("⚠ Please select at least one categorical feature for encoding.")
        else:
            st.warning("No categorical variables found. Encoding is not required.")
    
    # Add a general download button at the bottom for the final dataset
    if 'df_final' in st.session_state:
        st.subheader("Download Final Processed Dataset")
        final_csv = st.session_state['df_final'].to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Dataset",
            data=final_csv,
            file_name="processed_data.csv",
            mime="text/csv",
        )