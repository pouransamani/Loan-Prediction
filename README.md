## Classification of Personal Loan Approval
## Table of Content

- [Project Overview](#Project-Overview)
- [Objective](#Objective)
- [Dataset Description](#Dataset-Description)
- [Dataset Explanation,Metadata](#Dataset-Explanation-**Metadata**)
- [Notebook prepration](#Notebook-prepration)
- [preproccessing](#preproccessing)
- [Exploratory data Analysis(EDA)](#Exploratory-Data-Analysis-(EDA))
- [Data Wrangling](#Data-Wrangling)
- [Building a Linear Regression Model](#Buliding-a-linear-Regression-Model)
- [The result for baseline model, Cross validation and GridSearchCV](#The-result-for-baseline-model-cross-validation-and-GridSearchCV)
- [Conclusions](#Conclusions)
- [Recommendations](#Recommendations)



  ## Project Overview
  This project aims to develop a machine learning model to predict whether a customer will be accepted for a personal loan offer (1) or not (0) based on their financial and demographic information. This project is designed to assess classification algorithms, feature engineering, model evaluation, and derive actionable insights from data.
 
 ## Dataset
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


# Gemstone Price Prediction
---


