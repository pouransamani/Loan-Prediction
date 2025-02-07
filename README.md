## Classification of Personal Loan Approval
## Table of Content

- [Project Overview](#Project-Overview)
- [Dataset Description](#Dataset-Description)
- [Notebook prepration](#Notebook-prepration)
- [preproccessing and Exploratory Data Analysis(EDA) ](#preproccessing-and-Exploratory-Data-Analysis-(EDA))
- [Feature Engineering](#Feature-Engineering)
- [Model Development](#Model-Development)
- [Feature Importance](#Feature-Importance)
- [Voting Classifier](#Voting Classifier):
- [Model Tuned performance](#Model-Tuned-Performance)
- [Conclusions](#Conclusions)
- [Recommendations](#Recommendations)



  ## Project Overview
  This project aims to develop a machine learning model to predict whether a customer will be accepted for a personal loan offer (1) or not (0) based on their financial and demographic information. This project is designed to assess classification algorithms, feature engineering, model evaluation, and derive actionable insights from data.
 
  ## Dataset Description
 The file bank_personal_loan.csv contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.
There are no empty or (NaN) values in the dataset. The dataset has a mix of numerical and categorical attributes, but all categorical data are represented with numbers. Dataset in detail which contains customer information is as below;

 - **Age** => Customer's age in completed years
 - **Experience** => Years of professional experience
 - **Income** => Annual income of the customer (1000 Dollar)
 - **Eucation** => Education Level (1: Undergrad; 2: Graduate; 3: Advanced/Professional)
 - **Family** => Family size of the customer (1, 2, 3, 4)
 - **Mortgage** => Value of house mortgage ($1000) if any. 0, 101,...
 - **CCAvg** => Average spending on credit cards per month (1000 Dollar)
 - **Zip Code** => Home Address ZIP code
 - **CreditCard** => Does the customer use a credit card issued by Universal Bank?(0, 1)
 - **Online** => Does the customer use internet banking facilities?(0, 1)
 - **CD Account** => Does the customer have a certificate of deposit (CD) account with this bank?(0, 1)
 - **Securityaccoun** => Does the customer have a securities account with this bank?(0, 1)
 - **Target** variable => Personal Loan, which indicates whether a customer accepted the personal loan offer (1) or not (0).

## Notebook prepration
**1- Import libraries and dataset**
- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt
- import seaborn as sns
- plt.style.use('fivethirtyeight')
- from sklearn.preprocessing import StandardScaler
- from sklearn import metrics
- from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
- from sklearn.preprocessing import LabelEncoder
- from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
- from sklearn.preprocessing import LabelEncoder
- from sklearn.impute import SimpleImputer
- import xgboost as xgb
- import catboost as cb
- from sklearn.decomposition import PCA
- from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
- from sklearn.linear_model import LogisticRegression
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.svm import SVC
- from sklearn import svm
- from sklearn.metrics import confusion_matrix
- from scipy.stats import zscore
- from collections import Counter
- import warnings
- warnings.filterwarnings("ignore")

## preproccessing and Exploratory Data Analysis (EDA)
Data Cleaning:
- Remove duplicate entries.
- Handle missing values.
- Correct inconsistencies in the data.
- Detect and handle outliers.
- Generate summary statistics for the dataset.
- Visualize the distribution of each feature.
- Explore relationships between features using scatter plots, correlation matrices(Heatmap), etc.
- Identify any patterns or trends in the data.

## Feature Engineering:
- Normalize target.
- Using P-Values, chi-square test for identifying statistically significant features. 
- Encode categorical variables.

## Model Development:
- Use the cleaned and preprocessed dataset to build a classification model using DT, LR, KNN, SVM, RF algorithms.
- Split the dataset into training and testing sets.
- Train the model with package of algorithms and evaluate its performance using accuracy measure.
- Explore how model performance changes with different features or transformations using Cross Validation.
- Tuning the model by GridSearchCV
- Summarize findings and assess the models accuracies in classification personal loan receiving.
## Feature Importance:
- Use feature importance technique to identify the most influence varaibles and enhancing models performance.
-For every single algorithm;
   - Tree-based models (Random Forest, and Decision Tree) → Importance based on feature splits.
   - Permutation Importance → Measures how model accuracy changes when a feature is shuffled.
   - Coefficients in Linear Models (Logistic Regression, Linear SVM) → Shows feature weight in prediction.

## Voting Classifier:
 - Combining predictions from different applied machine learning models, provide an average prediction result based on the prediction of all the submodels. 
|Model |Baseline|Cross Validation|Tuned Performance(GridSearchCV)|Voting Classifier(Ensemble Methode)|
|-----|--------|-------|--------- |------|
|Decision Tree| 96.73%|97.20% |97.71% ||
|Logistic Regression| 95.2%|95.62% |95.68% ||
|K Nearest Neighbor| 96.93%|96.82%| 97.11%||
|rbf-SVM| 97.86%|97.57% |97.97% ||
|linear-SVM|95.26%| 95.65%  | --||
|Random Forest| 98.20%|98.14%   |98.25% ||
|Ensemble (Voting Classifier) ||98.17%||                                       97.93% |

## Conclusion:
- The Random Forest model identifies Education, Income, Mortgage, CCAvg, and Family as the most influential factors in predicting personal loan applications.
- Understanding the impact of these features helps banks refine the model and gain a deeper understanding of applicants' financial behavior.

## Recommondation:
- The key factors influencing the prediction of personal loan applications are Education, Income, Mortgage, CCAvg, and Family. These features provide valuable insights that help banks make informed decisions about loan approvals and shape their marketing strategies. By considering these factors, banks can better assess applicants' eligibility and target potential customers more effectively. 

 
     
