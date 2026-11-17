# Customer Churn & Retention Intelligence Platform

An end-to-end Machine Learning project that predicts customer churn,
identifies customers at risk of leaving, and provides actionable
retention recommendations.

## 1. Project Overview

Customer churn is a major challenge for subscription-based businesses.
Losing existing customers can increase acquisition costs and negatively
affect revenue.

This project uses Machine Learning to analyze customer data, predict the
likelihood of churn, categorize customers based on their risk levels,
and suggest suitable retention strategies.

The solution combines data analysis, predictive modelling, and an
interactive Streamlit application to transform customer data into
actionable business insights.

## 2. Problem Statement

Businesses often struggle to identify customers who are likely to
discontinue their subscriptions. Traditional approaches may identify
churn only after a customer has already left.

The objective of this project is to develop a predictive system that
identifies customers with a high probability of churn before they leave.

By analyzing customer demographics, account information, service usage,
and billing patterns, the system aims to help businesses take proactive
retention measures.

## 3. Business Objectives

-   Identify customers who are likely to churn.
-   Estimate individual customer churn probability.
-   Categorize customers into different risk levels.
-   Understand factors associated with customer churn.
-   Generate actionable customer retention recommendations.
-   Support data-driven business decisions.

## 4. Dataset Description

The project uses the Telco Customer Churn dataset containing 7,043
customer records and 21 columns.

### Feature Categories

  Category               Examples
  ---------------------- -------------------------------------------------
  Customer Information   Gender, Senior Citizen, Partner, Dependents
  Account Information    Tenure, Contract, Payment Method
  Services               Internet Service, Online Security, Tech Support
  Billing Information    Monthly Charges, Total Charges
  Target Variable        Churn

### Target Variable

The target variable is `Churn`.

-   Yes:Customer has discontinued the service.
-   No: Customer has continued the service.

## 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand customer behaviour
and identify patterns associated with churn.

Key areas investigated:

-   Customer demographics.
-   Contract types and churn behaviour.
-   Customer tenure and churn rates.
-   Monthly and total charges.
-   Service subscriptions.
-   Relationships between customer characteristics and churn.

## 6. Data Cleaning and Preprocessing

The dataset was prepared for Machine Learning through the following
steps:

-   Checked for missing values and duplicate records.
-   Handled data type inconsistencies.
-   Cleaned and transformed relevant columns.
-   Encoded categorical variables.
-   Prepared features and target variables.
-   Applied appropriate preprocessing for model training.

## 7. Machine Learning Workflow

The project follows an end-to-end Machine Learning workflow:

1.  Dataset collection and understanding.
2.  Data cleaning and preprocessing.
3.  Exploratory Data Analysis.
4.  Feature preparation.
5.  Model training.
6.  Model comparison.
7.  Hyperparameter tuning using GridSearchCV.
8.  Model evaluation.
9.  Model serialization using Joblib.
10. Integration with Streamlit.

### Models Explored

-   Logistic Regression
-   Random Forest Classifier

The models were compared using appropriate classification evaluation
metrics, and hyperparameter tuning was performed to improve model
performance.

Final selected model:Random Forest Classifier.

## 8. Model Evaluation

The model was evaluated using classification metrics such as:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix



  Evaluation Metric       Logistic Regression     Original Random Forest      Tuned Random Forest
Accuracy                   81%                      78.42%                        77.20%
Precision                  66%                      62%                           55%
Recall                     55%                      48%                           77%
F1-Score                   0.60                     0.54                          0.64
ROC-AUC                    0.847                    0.8213                        0.8448

## 9. Application Features

The project includes an interactive web application developed using
Streamlit.

### Business Overview Dashboard

-   Displays customer and churn-related insights.
-   Presents important business metrics.
-   Provides visualizations for understanding customer behaviour.

### Individual Customer Prediction

-   Accepts individual customer information.
-   Predicts churn probability.
-   Assigns a customer risk category.
-   Provides retention recommendations.

### Bulk Customer Prediction

-   Accepts customer data through CSV upload.
-   Generates predictions for multiple customers.
-   Assigns risk categories.
-   Provides retention recommendations.
-   Allows users to download prediction results as a CSV file.

### Risk Categorization

Customers are categorized based on their predicted churn probability:

-   Low Risk
-   Medium Risk
-   High Risk

Check the thresholds in the application code and document the exact
boundaries here.

### Retention Recommendation Engine

The application generates actionable recommendations based on customer
characteristics and risk factors.

## 10. Technology Stack

  Technology            Purpose
  --------------------- ----------------------
  Python                Programming language
  Pandas                Data manipulation
  NumPy                 Numerical operations
  Matplotlib / Plotly   Data visualization
  Scikit-learn          Machine Learning
  Joblib                Model serialization
  Streamlit             Web application
  Git                   Version control
  GitHub                Source code hosting



## 11. Project Structure




customer_churn_intelligence/
│
├── app.py
├── data_cleaning.py
├── eda.py
├── modelling and its evaluation.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│
├── models/
│   ├── churn_model.joblib
│   └── model_config.json
│
└── outputs/


## 12. Installation and Execution

### Prerequisites

-   Python 3.x
-   Git
-   pip

### Step 1: Clone the Repository

``` bash
git clone https://github.com/Samartha-SR77/customer_churn_intelligence.git
```

### Step 2: Navigate to the Project Folder

``` bash
cd customer_churn_intelligence
```

### Step 3: Install Dependencies

``` bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

``` bash
streamlit run app.py
```

The application will open in your default web browser.

Ensure that any required model files and permitted dataset files are
available in the expected locations before running the application.


## 13. Key Business Insights

The analysis highlighted several patterns associated with customer
churn:

-   Customers with shorter tenure tend to show higher churn rates.
-   Month-to-month contract customers are more likely to churn.
-   Senior citizens exhibit different churn patterns compared with other
    customer groups.
-   Monthly charges and service-related factors can provide useful
    signals for churn analysis.

These insights can help businesses design targeted retention strategies.
Confirm that each statement matches the findings from your own EDA
before publishing.

## 14. Future Improvements

-   Deploy the application to a cloud platform.
-   Improve model performance through additional feature engineering.
-   Introduce model monitoring and performance tracking.
-   Integrate real-time customer data.
-   Explore explainable AI techniques for better prediction
    transparency.

## 15. Author

Samartha S R

AI & Data Science Engineering Student
Siddaganga Institute of Technology,Tumakuru

GitHub: [Samartha-SR77](https://github.com/Samartha-SR77)

------------------------------------------------------------------------

