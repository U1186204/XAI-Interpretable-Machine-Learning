#!/usr/bin/env python
# coding: utf-8

# # AIPI 590 - XAI | Assignment 2 - Interpretable ML
# ### Description
# ### Christian Moreira
# 
# #### Include the button below. Change the link to the location in your github repository:
# #### Example: https://colab.research.google.com/github/yourGHName/yourREPOName/blob/yourBranchName/yourFileName.ipynb
# 
# 
# [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/U1186204/XAI-Interpretable-Machine-Learning/blob/main/Interpretable_ML_Chris.ipynb)

# # Install Packages

# # Load Packages

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_ccpr
from statsmodels.stats.diagnostic import linear_reset
import kagglehub
from kagglehub import KaggleDatasetAdapter


# # Data Load

# **The Churn dataset used in this assignment is derived from Kaggle and can be found [HERE](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)**

# In[ ]:


# Set the path to the main CSV file (hardcoded, since listing isn't supported)
file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "blastchar/telco-customer-churn",
    file_path
)


# # Task 1
# Exploratory Data Analysis to check Assumptions: Perform an exploratory analysis of the dataset to understand the relationships between different features and the target variable (churn). Use appropriate visualizations and statistical methods to determine whether assumptions about linear, logistic, and GAM models are met. 

# ### Data Dimensions, Variable Types, and Check for Missing Values

# In[4]:


print("Data Structure")
print("-----------------")
print(f"Dimensions: {df.shape}")
print(f"Data Types:\n{df.dtypes}")
print(f"Missing Values:\n {df.isnull().sum()}")


# ### Interpretation
# - Dataset has a total of 7043 observations & 21 variables
# - With the exception of MonthlyCharges, tenure and senior citizen - most other variables seem categorical, including the explained variable **Churn**
# - No missing fields have been reported suggesting there is not a need for imputations or deletion of observations

# ### Central tendencies for numerical variables, measures of dispersion for numerical variables, distributions

# In[5]:


print("\n Descriptive Statistics for Numerical Variables")
print("------------------")
numeric_columns = df.select_dtypes(include=np.number).columns
print("Central Tendency Measures:")
print(df[numeric_columns].describe().loc[['mean', '50%']])
print("\n Dispersion Measures:")
print(df[numeric_columns].describe().loc[['std', 'min', 'max']])

# Distribution Normality Check
print("\n Distribution Measures:")
print(df[numeric_columns].skew())
print(df[numeric_columns].kurt())


# ### Interpretation

# **Descriptive Statistics**
# - The variable SeniorCitizen appears to be a binary variable indicating whether an individual is a senior or not. 
# - Means and medians provide insights into the central location of the data. The mean tenure is ~32 months and the median 29 months. The mean monthly charges are ~$64 and median ~$70. 
# 
# **Dispersion Measures**
# - The skewness value for tenure of ~.23, which is positive and close to 0 indicates a slight right skew. Similarly, the skewness value of ~-.22 for monthly charges, while close to 0 and negative, suggests a slight left skew. 
# - The  kurtosis value for tenure and monthly charges are -1.38 and -1.25 respectively; These indicate values are spread out and with thin tails, which mean distributions for tenure months and monthly charges in $ should be relatively uniform.  

# ### Checking for Overall Data Quality: Duplicated values & Outliers 

# In[6]:


# Data Quanlity Checks
print("\n Data Quality")
print("------------------")
print(f"Duplicated Rows: {df.duplicated().sum()}")
print("Checking for Inconsistent Values:")
print(df.apply(lambda x: x.value_counts().index[0]).to_frame(name='Most Frequent Value'))


# ### Interpretation

# **Data Quality Assessment**
# - The dataset does not exhibit any inconsistencies: there are no duplicated values or nonsensical values in the dataset. 

# ### Data Encoding for Model EDA
# - Given a moajority of the dataset is Categorical we will do come encodings in order to visualize the relationship between explanatory variables and explained variable(Churn)
# - In addition, when total charges are null, we will assume those values are 0, there are very few of those instances(11 in total) relative to the size of the dataset ~7000+ observations

# In[ ]:


# Only Tenure and MonthlyCharges are numeric
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
df[numeric_columns] = df[numeric_columns].astype(float)

# Cleaning Internet and Phone Services and Encoding them to Binary
internet_dependent_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                              'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in internet_dependent_services:
    df[col] = df[col].replace({'No internet service': 'No'})
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0})

# Encoding all other Binary Variables to 0 and 1, Including Churn
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Encoding Categorical Variables with more than 2 categories using One-Hot Encoding
df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 
                                               'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})


# In[ ]:


# Only Tenure and MonthlyCharges are numeric
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
df[numeric_columns] = df[numeric_columns].astype(float)

# Cleaning Internet and Phone Services and Encoding them to Binary
internet_dependent_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                              'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in internet_dependent_services:
    df[col] = df[col].replace({'No internet service': 'No'})
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0})

# Encoding all other Binary Variables to 0 and 1, Including Churn
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Encoding Categorical Variables with more than 2 categories using One-Hot Encoding
df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 
                                               'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})

# Print Only the NUMBER of unique values in each column
print("\n Unique Values per Column")
print("------------------")
print(df.nunique())


# ### Linearity Check for Linear Regression

# In[23]:


# Drop customerID as it's just an identifier
data = df.drop('customerID', axis=1)

# Scatter plots of continuous predictors vs. churn
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
continuous_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']

for i, var in enumerate(continuous_vars):
    axes[i].scatter(data[var], data['Churn'], alpha=0.3, color='blue')

    # Add lowess smoother to help identify non-linearity
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lowess_result = lowess(data['Churn'], data[var], frac=0.3)
    axes[i].plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2)

    # Add linear regression line for comparison
    x = data[var]
    y = data['Churn']
    coef = np.polyfit(x, y, 1)
    axes[i].plot(x, coef[0] * x + coef[1], 'g--', linewidth=1.5)

    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Churn')
    axes[i].set_title(f'Churn vs {var}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Linearity Check: Scatter Plots with LOWESS Curve', y=1.05, fontsize=16)
plt.show()

# Ramsey's RESET Test - Corrected approach
# Create formula including all variables except customerID
X_vars = [col for col in data.columns if col != 'Churn']
formula = 'Churn ~ ' + ' + '.join(X_vars)

# Fit OLS model
model = ols(formula, data=data).fit()

# Manual implementation of RESET test
# Add powers of fitted values to the model
data['fitted_values'] = model.fittedvalues
data['fitted_values_squared'] = model.fittedvalues**2

# Create new model with squared fitted values
reset_formula = formula + ' + fitted_values_squared'
reset_model = ols(reset_formula, data=data).fit()

# Get F-test results comparing the two models
from scipy import stats
df1 = 1  # number of restrictions
df2 = model.df_resid - 1  # degrees of freedom of restricted model

# Calculate F-statistic 
f_stat = ((model.ssr - reset_model.ssr) / df1) / (reset_model.ssr / reset_model.df_resid)

# Calculate p-value
p_value = 1 - stats.f.cdf(f_stat, df1, df2)

print("\nRamsey's RESET Test Results (Manual Calculation):")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of Freedom: {df1}, {df2}")

if p_value < 0.05:
    print("The RESET test indicates non-linear relationships in the model (p < 0.05).")
    print("Consider using transformations or non-linear models.")
else:
    print("The RESET test does not detect significant non-linear relationships (p >= 0.05).")

# Clean up the added columns
data = data.drop(['fitted_values', 'fitted_values_squared'], axis=1)


# In[24]:


# 2. INDEPENDENCE OF OBSERVATIONS CHECK
from statsmodels.stats.stattools import durbin_watson

# Fit a model to get residuals
X = data.drop('Churn', axis=1)
y = data['Churn']
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X).fit()
residuals = model.resid

# Durbin-Watson test
dw_statistic = durbin_watson(residuals)

plt.figure(figsize=(12, 6))

# Plot residuals in sequence
plt.subplot(1, 2, 1)
plt.plot(residuals, 'o-', markersize=3, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals in Sequence')
plt.xlabel('Observation Number')
plt.ylabel('Residual')
plt.grid(True, alpha=0.3)

# Plot residuals against predicted values
plt.subplot(1, 2, 2)
plt.scatter(model.fittedvalues, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle(f'Independence Check: Durbin-Watson = {dw_statistic:.4f}', y=1.05, fontsize=16)
plt.show()

print(f"\nDurbin-Watson Statistic: {dw_statistic:.4f}")
print("Interpretation:")
if dw_statistic < 1.5:
    print("Positive autocorrelation might be present (DW < 1.5)")
elif dw_statistic > 2.5:
    print("Negative autocorrelation might be present (DW > 2.5)")
else:
    print("No significant autocorrelation detected (1.5 < DW < 2.5)")


# In[25]:


# 3. HOMOSCEDASTICITY CHECK
import statsmodels.stats.diagnostic as diag
import numpy as np

# Breusch-Pagan test for heteroscedasticity
bp_test = diag.het_breuschpagan(residuals, X)

plt.figure(figsize=(18, 6))

# Residuals vs Predicted Values Plot
plt.subplot(1, 3, 1)
plt.scatter(model.fittedvalues, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

# Scale-Location Plot (Sqrt of abs residuals vs predicted values)
plt.subplot(1, 3, 2)
sqrt_abs_resid = np.sqrt(np.abs(residuals))
plt.scatter(model.fittedvalues, sqrt_abs_resid, alpha=0.5)
# Add a LOWESS smoother
lowess_result = lowess(sqrt_abs_resid, model.fittedvalues, frac=0.3)
plt.plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2)
plt.title('Scale-Location Plot')
plt.xlabel('Predicted Values')
plt.ylabel('√|Residuals|')
plt.grid(True, alpha=0.3)

# Residuals vs Leverage
plt.subplot(1, 3, 3)
influence = model.get_influence()
leverage = influence.hat_matrix_diag
plt.scatter(leverage, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Leverage')
plt.xlabel('Leverage')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Homoscedasticity Check', y=1.05, fontsize=16)
plt.show()

print("\nBreusch-Pagan Test Results:")
print(f"LM Statistic: {bp_test[0]:.4f}")
print(f"p-value: {bp_test[1]:.4f}")
print(f"F-statistic: {bp_test[2]:.4f}")
print(f"F p-value: {bp_test[3]:.4f}")

if bp_test[1] < 0.05:
    print("The Breusch-Pagan test indicates heteroscedasticity (p < 0.05).")
    print("Consider using robust standard errors or transforming variables.")
else:
    print("The Breusch-Pagan test does not detect significant heteroscedasticity (p >= 0.05).")


# In[26]:


# 4. NORMALITY OF RESIDUALS CHECK
from scipy import stats
import statsmodels.api as sm

plt.figure(figsize=(18, 6))

# Histogram of residuals
plt.subplot(1, 3, 1)
sns.histplot(residuals, kde=True, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Q-Q plot
plt.subplot(1, 3, 2)
sm.qqplot(residuals, line='45', fit=True, ax=plt.gca())
plt.title('Q-Q Plot of Residuals')
plt.grid(True, alpha=0.3)

# Kernel Density Plot
plt.subplot(1, 3, 3)
sns.kdeplot(residuals, fill=True)
# Add a normal distribution with same mean and std for comparison
x = np.linspace(min(residuals), max(residuals), 100)
plt.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
         'r--', label='Normal Distribution')
plt.title('Kernel Density Plot of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Normality of Residuals Check', y=1.05, fontsize=16)
plt.show()

# Shapiro-Wilk test for normality
# Using a sample if there are too many observations (>5000)
if len(residuals) > 5000:
    residuals_sample = np.random.choice(residuals, size=5000, replace=False)
    shapiro_test = stats.shapiro(residuals_sample)
    print("\nShapiro-Wilk Test Results (on sample of 5000 observations):")
else:
    shapiro_test = stats.shapiro(residuals)
    print("\nShapiro-Wilk Test Results:")

print(f"W statistic: {shapiro_test[0]:.4f}")
print(f"p-value: {shapiro_test[1]:.4f}")

if shapiro_test[1] < 0.05:
    print("The Shapiro-Wilk test indicates that residuals are not normally distributed (p < 0.05).")
    print("Consider transforming variables or using robust regression methods.")
else:
    print("The Shapiro-Wilk test does not reject normality of residuals (p >= 0.05).")


# In[27]:


# 5. MULTICOLLINEARITY CHECK
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create correlation matrix for numerical variables
numeric_data = data[['tenure', 'MonthlyCharges', 'TotalCharges']]
correlation_matrix = numeric_data.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix of Numerical Variables', fontsize=16)
plt.tight_layout()
plt.show()

# Calculate VIF for each predictor
# VIF is calculated using all predictors, not just numerical ones
X = data.drop('Churn', axis=1)
X = sm.add_constant(X)  # Add constant term

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Sort by VIF value
vif_data = vif_data.sort_values("VIF", ascending=False)

# Create a horizontal bar chart for VIF values
plt.figure(figsize=(12, 10))
sns.barplot(x="VIF", y="Variable", data=vif_data)
plt.title('Variance Inflation Factor (VIF) for Each Predictor', fontsize=16)
plt.axvline(x=5, color='r', linestyle='--', label='VIF=5 (Moderate Multicollinearity)')
plt.axvline(x=10, color='darkred', linestyle='--', label='VIF=10 (High Multicollinearity)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nVariance Inflation Factor (VIF) Results:")
print(vif_data)
print("\nInterpretation:")
print("VIF > 10: High multicollinearity")
print("VIF > 5: Moderate multicollinearity")
print("VIF < 5: Low multicollinearity")


# In[29]:


# 6. INFLUENTIAL OUTLIERS CHECK (CORRECTED)
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt
import numpy as np

# Get influence metrics
influence = model.get_influence()
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]
student_resid = influence.resid_studentized_external

# Create a figure with 4 subplots
plt.figure(figsize=(18, 10))

# Plot Cook's distance - removed the problematic parameter
plt.subplot(2, 2, 1)
plt.stem(cooks_d, markerfmt=',')  # Removed use_line_collection parameter
plt.axhline(y=4/len(data), color='r', linestyle='--', 
            label=f'Threshold: 4/{len(data)}')
plt.title("Cook's Distance")
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Leverage vs Residuals Squared
plt.subplot(2, 2, 2)
plt.scatter(leverage, np.square(student_resid), alpha=0.5)
plt.title('Leverage vs Studentized Residuals Squared')
plt.xlabel('Leverage')
plt.ylabel('Studentized Residuals Squared')
plt.grid(True, alpha=0.3)

# Influence Plot (bubble plot of studentized residuals vs leverage)
plt.subplot(2, 2, 3)
# Size points by Cook's distance
size = 100 * cooks_d / max(cooks_d)
plt.scatter(leverage, student_resid, s=size, alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='-')
plt.axhline(y=3, color='r', linestyle='--')
plt.axhline(y=-3, color='r', linestyle='--')
plt.title('Influence Plot')
plt.xlabel('Leverage')
plt.ylabel('Studentized Residuals')
plt.grid(True, alpha=0.3)

# Plot influential observations (highlighting points with high Cook's distance)
plt.subplot(2, 2, 4)
# Identify top influential points
threshold = 4/len(data)  # Common rule of thumb
influential_points = np.where(cooks_d > threshold)[0]
plt.scatter(data['tenure'], data['MonthlyCharges'], alpha=0.3, color='blue')
plt.scatter(data.iloc[influential_points]['tenure'], 
           data.iloc[influential_points]['MonthlyCharges'], 
           color='red', alpha=0.6, s=50)
plt.title(f'Influential Points Highlighted ({len(influential_points)} points)')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Influential Outliers Check', y=1.02, fontsize=16)
plt.show()

print(f"\nNumber of influential observations (Cook's D > {threshold:.4f}): {len(influential_points)}")
if len(influential_points) > 0:
    print("Indices of top 10 most influential observations:")
    top_influential = np.argsort(cooks_d)[-10:][::-1]
    for i, idx in enumerate(top_influential):
        print(f"{i+1}. Index {idx}: Cook's D = {cooks_d[idx]:.4f}, "
              f"Leverage = {leverage[idx]:.4f}, "
              f"Studentized Residual = {student_resid[idx]:.4f}")


# In[ ]:




