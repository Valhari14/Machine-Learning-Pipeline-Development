# **Automated Machine Learning Pipeline Development**

## **Project Overview**

This repository contains the implementation and documentation of a Bachelor’s thesis project focused on the **design and development of an automated machine learning pipeline**. The objective is to streamline the end-to-end lifecycle of machine learning (ML) models—covering data ingestion, preprocessing, model training, evaluation, and deployment—through a user-friendly, no-code interface. The system integrates key principles from **AutoML**, **MLOps**, and **explainable AI**, and has been built with modularity and scalability in mind to suit both academic research and real-world applications.

## **Academic Context**

- **Project Type:** Bachelor’s Thesis  
- **Degree Program:** B.Tech in Artificial Intelligence and Data Science  
- **Academic Year:** 2024–25  
- **Institution:** Yeshwantrao Chavan College of Engineering, India  
- **Project Guide:** Prof. Prarthana A. Deshkar  
- **Industry Collaboration:** Incredo Technologies  
- **Author:** Valhari Meshram, Aman Raut, Viranchi Dakhare, Atharva Nerkar, Aniket Kaloo  


## **Objectives of the Project**

- Automate the machine learning workflow to reduce manual intervention.  
- Provide a user interface for seamless interaction with all pipeline components.  
- Enable effective feature engineering and hyperparameter tuning.  
- Ensure reproducibility, interpretability, and scalability.  
- Support deployment through exportable and containerized models.

## 🛠️ **Tools and Technologies Used**

| Component            | Technology             | Purpose                                                                 |
|---------------------|------------------------|-------------------------------------------------------------------------|
| Programming Language| Python 3.10            | Core development language                                               |
| ML Framework        | Scikit-learn           | Implementation of classification/regression models                     |
| Data Processing     | Pandas, NumPy          | Data cleaning, transformation, and numerical computations               |
| Visualization       | Matplotlib, Seaborn    | Graphical representation of evaluation metrics                         |
| Interface           | Streamlit              | No-code web-based UI for user interaction                              |
| Model Export        | Joblib / Pickle        | Save trained models for reuse                                          |
| Deployment          | Docker                 | Containerize models for production-ready deployment                    |


## **Key Features**

- **Data Upload and Ingestion**: Supports structured data in CSV and Excel formats.  
- **Preprocessing Automation**: Cleans, scales, encodes, and transforms features without user input.  
- **Model Training and Selection**: Offers a suite of supervised learning algorithms with auto-tuning.  
- **Evaluation Metrics**: Provides standard metrics like accuracy, precision, recall, and visual plots.  
- **Model Export**: Saves best-performing models for future deployment in `.pkl` format.  
- **Docker Integration**: Facilitates seamless containerization for deployment.  
- **Customizable Workflow**: Users can adjust model type, validation strategy, and hyperparameters.

## 🚀 **Getting Started**

### 1. Clone the Repository
```bash
git clone https://github.com/Valhari14/Machine-Learning-Pipeline-Development.git
cd Machine-Learning-Pipeline-Development
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app/01_Data_Ingestion.py
```

## **Thesis Report**

The complete thesis report—including the abstract, literature review, methodology, results, discussion, and references—is available under the `/thesis/` directory. This document serves as an academic reference and supports the technical foundations of the implemented system.

## **Demonstration and Test Data**

Sample datasets are included in the `/datasets/` folder for quick experimentation. Users can also upload their own data files via the interface. The pipeline has been tested on multiple classification and regression tasks to ensure robustness and generalizability.

## **Academic and Research Relevance**

This project demonstrates core competencies in:
- Applied machine learning and automation  
- Software engineering for data science tools  
- Model deployment and lifecycle management  
- User-centered ML tool design  
- Research-based system evaluation  

It is suitable as both a practical tool and a foundation for further academic research or postgraduate coursework.

## **License**

This project is distributed under the **MIT License**. See the `LICENSE` file for more details.
