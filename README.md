[![Repo](https://img.shields.io/badge/GitHub-XAI--Interpretable--Machine--Learning-blue?logo=github)](https://github.com/U1186204/XAI-Interpretable-Machine-Learning)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/U1186204/XAI-Interpretable-Machine-Learning/blob/main/Interpretable_ML_Chris.ipynb)
[![CI](https://github.com/U1186204/XAI-Interpretable-Machine-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/U1186204/XAI-Interpretable-Machine-Learning/actions/workflows/ci.yml)
[![CI](https://github.com/U1186204/XAI-Interpretable-Machine-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/U1186204/XAI-Interpretable-Machine-Learning/actions/workflows/ci.yml)


# Interpretable Machine Learning for Churn Prediction

## Project Overview
This project explores different machine learning approaches to predict customer churn in a telecommunications company, focusing on the interpretability-performance tradeoff of various models.

## Key Packages Used
pandas==2.3.2
numpy==2.2.6
matplotlib==3.10.6
seaborn==0.13.2
scikit-learn==1.7.2
statsmodels==0.14.5
scipy==1.16.1
pygam==0.10.1
kagglehub==0.3.13


## Assignment Structure
1. **Exploratory Data Analysis**: Analyzed relationships between features and churn, checking assumptions for different modeling approaches
2. **Linear Regression**: Built and interpreted a linear model treating churn as a continuous variable
3. **Logistic Regression**: Implemented a logistic model treating churn as a binary outcome
4. **Generalized Additive Model (GAM)**: Created a GAM to capture non-linear relationships between features and churn
5. **Model Comparison**: Evaluated models on both performance metrics and interpretability

## Key Findings

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| GAM | The GAM model performed the best in terms of providing strong predictive capabilities. It had the highest accuracy(79%),ROC (.83) and Precision and Recall Scores(.64 and .48 respectively). This suggests that its added complexity was able to provide better model performance scores in general. | GAM while a stronger model when predicting is substantially more complex and does not have coefficients that easily translate into interpretable insights. Less technical audiences could have a harder time understanding the GAM model and its mathematical nuances. |
| Linear Regression | The Linear model is easily interpretable and flexible. Many issues could be addressed with minimal transformations on explanatory variables like making a log transformation on Monthly and Total Charges or removing any variables for which P-values did not explain Churn. | Many assumptions from the linear model were violated when predicting churn. Normality and Heteroscedasticity were 2 examples of assumptions violated when using the linear model to predict our independent variable. Also, while simpler to understand, the linear model did not necessarily present the best accuracy or ROC scores. |
| Logistic Regression | Probably the most interpretable model in this case. Understanding the Odds of Churn increasing or decreasing is quite simple for most audiences. Also many of the flexibility existent in the Linear Regression model could be incorporated into the Logistic Regression | While this model relies on more complex math to implement, its results did not necessarily become outstanding relative to the Linear Model, they were in fact quite close in terms of accuracy, ROC, Precision and Recall. The improvements were minor. |

## Recommendation
Based on the interpretability-performance tradeoff, **Logistic Regression** provides the optimal balance for this business problem. While GAM showed slightly better performance metrics, the logistic model offers substantially better interpretability without significant sacrifices in predictive power. The logistic model can be easily explained to non-technical stakeholders while still providing reliable churn predictions.
