import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import io

# Set page to open in wide mode
st.set_page_config(layout="wide")

# Custom CSS to style both the checkbox 
st.markdown("""
    <style>
    div[data-testid="stCheckbox"] > div {
        display: flex;
        align-items: center;
    }

    div[data-testid="stCheckbox"] {
        margin-bottom: 15px;
        padding: 5px;
        border: 2px solid #007BFF;
        border-radius: 5px;
        background-color: #f0f0f0;
    }

    .correlation-section {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 30px;
    }
    
    .box-plot {
        width: 60%;
    }

    </style>
""", unsafe_allow_html=True)

# Function to create a heatmap
def HeatMap(df, x=True):
    correlations = df.corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heatmap_fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f', square=True, 
                linewidths=.5, annot=x, cbar_kws={"shrink": .75})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    return heatmap_fig
# Custom function for plotting outliers
def OutLiersBox(df, nameOfFeature):
    trace0 = go.Box(
        y=df[nameOfFeature],
        name="All Points",
        jitter=0.3,
        pointpos=-1.8,
        boxpoints='all',
        marker=dict(color='rgb(7,40,89)'),
        line=dict(color='rgb(7,40,89)')
    )

    trace1 = go.Box(
        y=df[nameOfFeature],
        name="Only Whiskers",
        boxpoints=False,
        marker=dict(color='rgb(9,56,125)'),
        line=dict(color='rgb(9,56,125)')
    )

    trace2 = go.Box(
        y=df[nameOfFeature],
        name="Suspected Outliers",
        boxpoints='suspectedoutliers',
        marker=dict(color='rgb(8,81,156)', outliercolor='rgba(219, 64, 82, 0.6)',
                    line=dict(outliercolor='rgba(219, 64, 82, 0.6)', outlierwidth=2)),
        line=dict(color='rgb(8,81,156)')
    )

    trace3 = go.Box(
        y=df[nameOfFeature],
        name="Whiskers and Outliers",
        boxpoints='outliers',
        marker=dict(color='rgb(107,174,214)'),
        line=dict(color='rgb(107,174,214)')
    )

    data = [trace0, trace1, trace2, trace3]

    layout = go.Layout(
        margin=dict(t=40),  # Adjust the top margin for better spacing
        legend=dict(
            orientation="h",  # Set the legend orientation to horizontal
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

st.title("Exploratory Data Analysis")

# Retrieve the dataframe from session state
if 'df' in st.session_state:
    df = st.session_state['df']

# Display Data Types and Null Values
# Display Data Description and Data Types & Null Values Side by Side
if st.checkbox("Show Descriptive Statistics"):
    st.subheader("Data Description")

    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)  # Ensuring proper fit

    with col2:
        st.subheader("Data Types and Null Values")
        null_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Null Values': df.isnull().sum(),
            'Non-null Count': df.notnull().sum()
        })
        null_info['Null Values'] = null_info['Null Values'].apply(lambda x: f"No null values" if x == 0 else f"{x} null values")
        # Wrap inside an expander to prevent React errors
        with st.expander("View Null Values Table"):
            st.dataframe(null_info, use_container_width=True)


if st.checkbox("Show Data Visualization"):
   
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    # Select only numerical columns for correlation analysis
    numerical_df = df.select_dtypes(include=['number'])

    if not numerical_df.empty:
        numerical_features = numerical_df.columns.tolist()
        st.write(f"The following numerical features are used for correlation analysis: {', '.join(numerical_features)}")
        
        heatmap_fig = HeatMap(numerical_df, x=True)
        st.pyplot(heatmap_fig)

        col1, col2 = st.columns([3, 4], gap="large")

        with col1:
            st.markdown("### Correlation Table")  # Ensure same heading level
            correlations = numerical_df.corr().abs().unstack().sort_values(ascending=False)
            correlations = correlations[correlations < 1].drop_duplicates()
            correlations = correlations.reset_index()
            correlations.columns = ['Feature 1', 'Feature 2', 'Correlation']
            st.write(correlations.head(10))

        with col2:
            st.subheader("Interpretation of Correlations")
            st.markdown("""
        **Understanding Correlation:**
        
        Correlation values range from **-1** to **1**:
        
        - **Positive Correlation** (closer to 1): As one feature increases, the other feature tends to increase.
        Example: Higher study hours leading to better grades.
        
        - **Negative Correlation** (closer to -1): As one feature increases, the other feature tends to decrease.
        Example: More exercise might result in lower body fat percentage.
        
        - **No Correlation** (closer to 0): Minimal or no linear relationship between features.
        Example: Shoe size vs. exam scores.

        **Importance for Predictive Modeling:**
        
        ✔ Strong correlations (values near ±1) indicate a significant relationship and are often key features for prediction.
        
        ✔ High correlation between independent features can lead to multicollinearity, which may require addressing by removing or combining features to avoid redundancy and overfitting.
        """)
       # Adding CSS to ensure proper alignment
        st.markdown("""
        <style>
            .stColumn {
                display: inline-block;
                vertical-align: top;
            }
        </style>
        """, unsafe_allow_html=True)     
    
    else:
        st.warning("No numerical features found for correlation analysis.")
    
   # Cardinality Analysis for Categorical Variables
    st.subheader("Cardinality Analysis")
    categorical_df = df.select_dtypes(include=['object'])
    
    if not categorical_df.empty:
        categorical_features = categorical_df.columns.tolist()
        st.write(f"**The following categorical features are used for cardinality analysis:** {', '.join(categorical_features)}")
    
        # Creating two columns for better layout
        col1, col2 = st.columns(2)
    
        # Unique Category Count per Feature
        with col1:
            st.subheader("Unique Category Count")
            cardinality_info = categorical_df.nunique().reset_index()
            cardinality_info.columns = ["Feature", "Unique Categories"]
            st.dataframe(cardinality_info, use_container_width=True)
    
        # Category Frequency Distribution
        with col2:
            st.subheader("Category Frequency Distribution")
            cardinality_details = []
    
            for var in categorical_features:
                value_counts = df[var].value_counts()
                relative_counts = (value_counts / len(df)).round(2)  
    
                for category, count in value_counts.items():
                    cardinality_details.append({
                        "Feature": var,
                        "Category": category,
                        "Count": count,
                        "Proportion": f"{relative_counts[category] * 100:.2f}%"  
                    })
    
            # Convert to DataFrame
            cardinality_df = pd.DataFrame(cardinality_details)
    
            # Fix React Error: Use Expander & Limit Displayed Rows
            with st.expander("View Category Frequency Details"):
                st.dataframe(cardinality_df.head(100), use_container_width=True) 
    
            
        st.subheader("Interpretation of Cardinality")
        st.markdown('''
 
Cardinality refers to the number of **unique categories** in a categorical feature, influencing data encoding, model performance, and interpretability.  

#### **Unique Category Count per Feature**  
This table shows the number of **distinct categories** in each categorical feature.  

- **Low Cardinality (Few Unique Categories):** Useful for **one-hot encoding** (e.g., "Gender" → Male, Female, Other).  
- **High Cardinality (Many Unique Categories):** Features like **Customer ID** require alternative encodings (e.g., target encoding, hashing) to prevent complexity and overfitting.  

#### **Category Frequency Distribution**  
This table presents how often **each category appears** within its feature.  

- **Balanced Distribution:** Categories appear in **similar proportions**, preventing bias.  
- **Imbalanced Distribution:** Some categories dominate, which may require **handling rare values** (e.g., grouping as "Other").  

#### **Importance in Predictive Modelling**  
 ✔ **Prevents Overfitting:** High-cardinality features may need specialized encodings.  
 ✔ **Enhances Feature Engineering:** Helps decide between **one-hot, label, frequency, or target encoding**.  
 ✔ **Detects Bias:** Identifies dominant categories that could skew model predictions.  
 ✔ **Improves Sampling Strategies:** Guides **upsampling/downsampling** in classification tasks.  
''')

        
    else:
        st.warning("No categorical features found for cardinality analysis.")
        
# Outlier Investigation and Interpretation
if st.checkbox("Outlier Investigation"):
    st.subheader("Single Feature Outliers")
    feature = st.selectbox("Select feature for outlier detection", df.columns)

    # Create aligned sections for outlier box plot and interpretation with appropriate gap
    col1, col2 = st.columns([4, 3], gap="large")  # Adjust column widths for better space


    with col1:
        st.subheader(f"Boxplot for : {feature}")
        # Call the custom Plotly outlier function
        st.markdown("<br><br><br>", unsafe_allow_html=True)  # Adds two line breaks to push the plot down
        fig = OutLiersBox(df, feature)
        st.plotly_chart(fig, use_container_width=True)  # Use container width to make it responsive

    with col2:
        st.subheader("Understanding Box Plots and Outliers")
        st.markdown("""
        ****
        - **Box**: Represents the interquartile range (IQR), which contains the middle 50% of the data.Lower Edge: 
                    1st Quartile (Q1). Upper Edge: 3rd Quartile (Q3). Horizontal Line inside the Box: Median          
        - **Whiskers**: The lines that extend from the box to the smallest and largest values within 1.5 * IQR.
        - **Outliers**: Data points that lie outside the whisker range are considered outliers and are displayed as individual dots.
        - **All Points**: Every data point, including outliers.
        - **Only Whiskers**: Displays only the key data range within the whiskers, hiding outliers.
        - **Suspected Outliers**: Data points that fall outside 1.5 * IQR but aren't extreme enough to be definite outliers.
        - **Whiskers and Outliers**: Displays both the whiskers and any definite outliers.
        """)

