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

# In[162]:


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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.stats.diagnostic as diag
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import lasso_path
from statsmodels.stats import diagnostic
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
from scipy.io import loadmat
from pygam import LogisticGAM, GAM, s, l
import kagglehub
from kagglehub import KaggleDatasetAdapter
import warnings
warnings.filterwarnings('ignore')


# # Data Load

# **The Churn dataset used in this assignment is derived from Kaggle and can be found [HERE](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)**

# In[36]:


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

# In[37]:


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

# In[38]:


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

# In[39]:


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

# In[40]:


# Only Tenure and MonthlyCharges are numeric
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
# Replace empty total charge with null and drop them
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df = df.dropna(subset=['TotalCharges'])
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


# ### Linearity Assumption Check for Linear Regression

# In[41]:


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


# ### Independence of Observations Assumption Check for Linear Regression

# In[42]:


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


# ### Homoscedasticifty Assumption Check for Linear Regression

# In[43]:


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


# ### Normality of Residuals Assumption Check for Linear Regression

# In[44]:


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


# ### Multicollinearity Assumption Check for Linear Regression

# In[45]:


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

# Extract high VIF variables for correlation analysis
# Based on VIF results, these are the variables with highest VIF
high_vif_vars = ['MonthlyCharges', 'InternetService', 'PhoneService', 
                 'StreamingMovies', 'StreamingTV', 'TotalCharges', 'MultipleLines', 
                 'tenure', 'DeviceProtection', 'OnlineBackup', 'TechSupport', 'OnlineSecurity']

high_vif_corr = data[high_vif_vars].corr()

# Create a heatmap specifically for high VIF variables
plt.figure(figsize=(12, 10))
sns.heatmap(high_vif_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix of High VIF Variables', fontsize=16)
plt.tight_layout()
plt.show()

# Add correlation values for highest VIF variables
top_vif_vars = vif_data.head(6)['Variable'].tolist()
if 'const' in top_vif_vars:
    top_vif_vars.remove('const')  # Remove constant term from correlation analysi


# ### Interpretation Linear Regression EDA Analysis

# - **Non-Linearity**: The Ramsey's RESET test (p-value = 0.0000) strongly indicates non-linear relationships. When running linear regression we should consider applying transformations like log or polynomial terms to continuous variables (tenure, MonthlyCharges, TotalCharges)
# 
# - **Multicollinearity**: VIF results show multicollinearity, with higher correlation between **(monthly charges & internet service)** and **(total charges and tenure)**. Lasso regularization Or variable dropping/combination should help address this issue. 
# 
# - **Non-Normality**: The Shapiro-Wilk test (p-value = 0.0000) shows that residuals aren't normally distributed. May indicate model is unfit, but could improve with above transformations. 
# 
# - **Heteroscedasticity**: Heteroscedasticity was detected. May indicate model is unfit relative to Logistic Regression, but let's apply some of the above transformations and see if that improves.  
# 
# - **Remove Redundant Features**: Given multicollinearity, we should perform feature selection, potentially using methods like LASSO. 

# ### Linearity Assumption Check for Logistic Regression

# In[46]:


# Drop customerID as it's just an identifier
data = df.drop('customerID', axis=1)

# Continuous predictors
continuous_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Create a dataframe to store results
box_tidwell_results = pd.DataFrame(columns=['Variable', 'Coefficient', 'p-value'])

# Create a figure for visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, var in enumerate(continuous_vars):
    # Create interaction term (X * log(X))
    data[f'{var}_log'] = data[var] * np.log(data[var] + 1)  # Adding 1 to handle zeros

    # Fit logistic regression with variable and its interaction with log
    formula = f'Churn ~ {var} + {var}_log'
    model_bt = smf.logit(formula=formula, data=data).fit(disp=0)

    # Store results - using pandas concat instead of append
    new_row = pd.DataFrame({
        'Variable': [var],
        'Coefficient': [model_bt.params[f'{var}_log']],
        'p-value': [model_bt.pvalues[f'{var}_log']]
    })
    box_tidwell_results = pd.concat([box_tidwell_results, new_row], ignore_index=True)

    # Visualization: Plot empirical logit
    # Group data into bins
    bins = np.linspace(data[var].min(), data[var].max(), 20)
    groups = pd.cut(data[var], bins)

    # Calculate mean predictor and proportion of 1's in each bin
    bin_summary = data.groupby(groups, observed=False)[['Churn', var]].agg({
        'Churn': 'mean', 
        var: 'mean'
    }).reset_index(drop=True)

    # Calculate empirical logit
    bin_summary['logit'] = np.log(bin_summary['Churn'] / (1 - bin_summary['Churn']))

    # Remove infinite values (if any)
    bin_summary = bin_summary.replace([np.inf, -np.inf], np.nan).dropna()

    # Plot empirical logit vs predictor
    axes[i].scatter(bin_summary[var], bin_summary['logit'], color='blue', alpha=0.7)

    # Add linear fit line
    if len(bin_summary) > 1:  # Ensure there are at least 2 points for regression
        x = bin_summary[var]
        y = bin_summary['logit']
        coef = np.polyfit(x, y, 1)
        axes[i].plot(x, coef[0] * x + coef[1], 'r--', linewidth=1.5)

    # Add smooth curve to check for non-linearity
    from statsmodels.nonparametric.smoothers_lowess import lowess
    if len(bin_summary) > 2:  # Need at least 3 points for lowess
        lowess_result = lowess(bin_summary['logit'], bin_summary[var], frac=0.6)
        axes[i].plot(lowess_result[:, 0], lowess_result[:, 1], 'g-', linewidth=2)

    axes[i].set_title(f'Empirical Logit Plot for {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Empirical Logit')
    axes[i].grid(True, alpha=0.3)

    # Clean up
    data = data.drop(f'{var}_log', axis=1)

plt.tight_layout()
plt.suptitle('Linearity of the Logit Check (Box-Tidwell Test)', y=1.05, fontsize=16)
plt.show()

# Display Box-Tidwell test results
print("Box-Tidwell Test Results:")
print(box_tidwell_results)
print("\nInterpretation:")
print("If p-value < 0.05, the linearity assumption is violated for that variable.")
print("If the empirical logit plot shows curvature, consider transforming the variable.")


# ### Complete Separation Assumption Check for Logistic Regression

# In[47]:


# Split data
X = data.drop('Churn', axis=1)
y = data['Churn']

# Add constant
X_sm = sm.add_constant(X)

# Fit full logistic regression model
try:
    logit_model = sm.Logit(y, X_sm).fit(disp=0)

    # Create a figure for visualization of separation
    plt.figure(figsize=(12, 8))

    # Extract parameters and standard errors
    params = logit_model.params
    std_errors = logit_model.bse

    # Calculate coefficient / standard error ratio
    ratio = abs(params / std_errors)

    # Sort for better visualization
    ratio = ratio.sort_values(ascending=False)

    # Create a bar plot
    plt.barh(range(len(ratio)), ratio, align='center')
    plt.yticks(range(len(ratio)), ratio.index)
    plt.axvline(x=2, color='r', linestyle='--', 
                label='Threshold for concern (|coef/se| > 2)')
    plt.axvline(x=4, color='darkred', linestyle='--', 
                label='Potential separation (|coef/se| > 4)')
    plt.title('Coefficient to Standard Error Ratio')
    plt.xlabel('|Coefficient / Standard Error|')
    plt.ylabel('Variables')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Display separation check results
    print("Logistic Regression Coefficient and Standard Error Results:")
    results_df = pd.DataFrame({
        'Coefficient': params,
        'Std Error': std_errors,
        '|Coef/SE|': abs(params / std_errors)
    }).sort_values('|Coef/SE|', ascending=False)

    print(results_df)
    print("\nInterpretation:")
    print("Very large coefficient to standard error ratios (e.g., > 4) may indicate separation.")
    print("Variables with extremely large ratios might perfectly or near-perfectly predict the outcome.")

    # Check for perfect prediction
    large_ratio_vars = results_df[results_df['|Coef/SE|'] > 4].index.tolist()
    if large_ratio_vars:
        print(f"\nPotential separation detected in variables: {', '.join(large_ratio_vars)}")
        print("Consider checking cross-tabulations for these variables.")
    else:
        print("\nNo strong indication of complete or quasi-complete separation.")

except Exception as e:
    print(f"Error in fitting model: {e}")
    print("This error might itself indicate complete separation.")

    # Try to identify problematic variables through individual analysis
    print("\nAttempting to identify problematic variables:")

    for col in X.columns:
        # Create a cross-tabulation
        cross_tab = pd.crosstab(data[col], data['Churn'])

        # Check for any zeros (indication of potential separation)
        if (cross_tab == 0).any().any():
            print(f"Variable '{col}' shows potential separation:")
            print(cross_tab)
            print()


# ### Calibration Check for Logistic Regression

# In[48]:


# Drop customerID as it's just an identifier
data = df.drop('customerID', axis=1)

# Split data into training and testing sets
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Get predicted probabilities
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# Plot calibration curve
plt.figure(figsize=(10, 8))

# Plot perfectly calibrated curve
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

# Plot calibration curve for the model
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
plt.plot(prob_pred, prob_true, 's-', label='Logistic Regression')

# Calculate MSE of the calibration curve
calibration_mse = np.mean((prob_true - prob_pred) ** 2)

# Add histograms for predicted probabilities - changed normed to density
plt.hist(y_pred_prob, range=(0, 1), bins=10, histtype='step', 
         density=True, label='Predicted Probability Distribution')

plt.title('Calibration Curve (Reliability Diagram)')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives (Churn=1)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Add calibration MSE to the plot
plt.text(0.1, 0.9, f'Calibration MSE: {calibration_mse:.4f}', 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

print("Calibration Check Results:")
print(f"Calibration Mean Squared Error: {calibration_mse:.4f}")
print("\nInterpretation:")
print("A well-calibrated model will follow the diagonal line (y=x).")
print("Points above the line indicate underestimation of probabilities.")
print("Points below the line indicate overestimation of probabilities.")
print("Lower calibration MSE indicates better probability estimates.")


# ### Sample Size Check for Logistic Regression

# In[49]:


# Calculate number of predictors (exclude Churn)
n_predictors = data.shape[1] - 1

# Count number of events (Churn=1)
n_events = data['Churn'].sum()

# Calculate events per predictor
events_per_predictor = n_events / n_predictors

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(['Events per Predictor'], [events_per_predictor], color='blue')
plt.axhline(y=10, color='r', linestyle='--', label='Minimum Recommended (10)')
plt.axhline(y=20, color='g', linestyle='--', label='Conservative Recommendation (20)')
plt.title('Events Per Predictor for Logistic Regression')
plt.ylabel('Number of Events per Predictor')
plt.ylim(0, max(30, events_per_predictor * 1.2))  # Set appropriate y-axis limit
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Display results
print("Adequate Sample Size Check Results:")
print(f"Number of predictors: {n_predictors}")
print(f"Number of events (Churn=1): {n_events}")
print(f"Events per predictor: {events_per_predictor:.2f}")
print("\nInterpretation:")
if events_per_predictor >= 20:
    print("EXCELLENT: More than 20 events per predictor (conservative recommendation).")
elif events_per_predictor >= 10:
    print("GOOD: At least 10 events per predictor (minimum recommendation).")
else:
    print("CAUTION: Less than 10 events per predictor.")
    print("Consider reducing the number of predictors or collecting more data.")

# Additional calculation: Maximum number of predictors recommended
max_recommended_predictors = n_events / 10
print(f"\nBased on the number of events, the maximum recommended number of predictors is: {int(max_recommended_predictors)}")

# If using variable selection, show how many variables we can keep
print("\nVariable Importance for Selection (if needed):")
from sklearn.linear_model import LogisticRegression

X = data.drop('Churn', axis=1)
y = data['Churn']

# Fit logistic regression with L1 penalty to get feature importance
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
logreg_l1.fit(X, y)

# Get feature importance and sort
feature_importance = pd.DataFrame({
    'Variable': X.columns,
    'Importance': np.abs(logreg_l1.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Show top variables
print(feature_importance)


# ### Interpretation for Logistic Model Assumptions EDA

# - **Linearity**: We should apply transformations to continuous variables *(tenure, MonthlyCharges, TotalCharges)* as Box-Tidwell test shows significant non-linearity (p<0.05); consider log transformations. 
# 
# - **Multicollinearity**: Removing either *InternetService (VIF=363)* or *MonthlyCharges (VIF=864)* could be beneficial alongside also removing *StreamingTV and StreamingMovies (both VIF>24)* due to high collinearity.
# 
# - **Separation**: Some variables show high coefficient/standard error ratios *(tenure, Contract, PaperlessBilling, PaymentMethod, TotalCharges)*; Penalizing methods like Ridge could address address separation.
# 
# - **Feature Selection**: While sample size is adequate, L1 penalty results could help select select only the most important predictors, which increases model interpretability.  

# ### Non Linear Assumption Check for GAM

# In[50]:


# Continuous predictors
continuous_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Create a figure for visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, var in enumerate(continuous_vars):
    # Create a feature matrix with just this predictor
    X = data[[var]].values
    y = data['Churn'].values

    # Fit a univariate GAM with a smooth term
    gam = LogisticGAM(s(0, n_splines=10, spline_order=3)).fit(X, y)

    # Generate prediction grid
    XX = np.linspace(X.min(), X.max(), 100)[:, np.newaxis]

    # Plot the partial effect
    axes[i].plot(XX[:, 0], gam.predict_proba(XX), 'r-', label='GAM Fit')

    # Add confidence intervals - FIXED HERE
    try:
        # Try the direct method first
        intervals = gam.confidence_intervals(XX, width=0.95)
        if isinstance(intervals, tuple) and len(intervals) == 2:
            lower, upper = intervals
            axes[i].fill_between(XX[:, 0], lower, upper, alpha=0.2, color='r')
        elif hasattr(intervals, 'shape') and intervals.shape[1] == 2:
            # Some versions return an array with shape (n, 2)
            axes[i].fill_between(XX[:, 0], intervals[:, 0], intervals[:, 1], alpha=0.2, color='r')
    except Exception as e:
        print(f"Skipping confidence intervals for {var}: {str(e)}")

    # Add scatter plot of binned data for comparison
    # Group data into bins
    bins = np.linspace(data[var].min(), data[var].max(), 20)
    groups = pd.cut(data[var], bins)

    # Calculate mean predictor and proportion of 1's in each bin
    bin_summary = data.groupby(groups, observed=False)[['Churn', var]].agg({
        'Churn': 'mean', 
        var: 'mean'
    }).reset_index(drop=True)

    # Plot binned points
    axes[i].scatter(bin_summary[var], bin_summary['Churn'], color='blue', alpha=0.7)

    # Add a linear fit line for comparison
    if len(bin_summary) > 1:
        x = bin_summary[var]
        y = bin_summary['Churn']
        coef = np.polyfit(x, y, 1)
        axes[i].plot(x, coef[0] * x + coef[1], 'g--', linewidth=1.5, label='Linear Fit')

    axes[i].set_title(f'GAM Partial Effect for {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Partial Effect on Churn Probability')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Non-Linear Relationships Check for GAM', y=1.05, fontsize=16)
plt.show()

# Fit a multivariate GAM to test significance of smooth terms
# Prepare data
X = data[continuous_vars].values
y = data['Churn'].values

# Fit GAM
gam = LogisticGAM(s(0, n_splines=10) + s(1, n_splines=10) + s(2, n_splines=10)).fit(X, y)

# Display results
print("GAM Summary:")
print(gam.summary())
print("\nInterpretation:")
print("p-value < 0.05 for a smooth term indicates significant non-linearity.")
print("The shape of the partial effect plots shows the nature of the non-linear relationship.")


# ### Smoothing Parameters Assumption Check for GAM

# In[51]:


# Ensure Churn is in binary format (0/1)
if data['Churn'].dtype == 'object':
    print("Converting Churn to binary format...")
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    print(f"Unique values in Churn after conversion: {data['Churn'].unique()}")

# Set up cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Continuous predictors
continuous_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Prepare data
X = data[continuous_vars].values
y = data['Churn'].values

# Try different smoothing parameters
lam_range = np.logspace(-3, 3, 7)  # Logarithmically spaced values
n_splines_range = [4, 5, 7, 10, 15]  # Changed from [3, 5, 7, 10, 15] to ensure n_splines > spline_order

# Store results
results = []

# For each number of splines
for n_splines in n_splines_range:
    # For each smoothing parameter
    for lam in lam_range:
        cv_scores = []

        # Cross-validation loop
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                # Fit GAM with current parameters
                gam = LogisticGAM(
                    s(0, n_splines=n_splines, spline_order=3, lam=lam) + 
                    s(1, n_splines=n_splines, spline_order=3, lam=lam) + 
                    s(2, n_splines=n_splines, spline_order=3, lam=lam)
                ).fit(X_train, y_train)

                # Evaluate on test set
                score = gam.loglikelihood(X_test, y_test)
                cv_scores.append(score)
            except Exception as e:
                print(f"Error with n_splines={n_splines}, lam={lam}: {e}")
                # Use a very low score to ensure this combination is not selected
                cv_scores.append(-np.inf)

        # Store average score
        results.append({
            'n_splines': n_splines,
            'lambda': lam,
            'mean_log_likelihood': np.mean(cv_scores)
        })

# Convert to DataFrame for easier handling
results_df = pd.DataFrame(results)

# Create a heatmap of results
plt.figure(figsize=(12, 8))
heatmap_data = results_df.pivot(index='n_splines', columns='lambda', values='mean_log_likelihood')
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
plt.title('GAM Smoothing Parameter Selection (Higher Log-Likelihood is Better)')
plt.xlabel('Smoothing Parameter (λ)')
plt.ylabel('Number of Splines')
plt.tight_layout()
plt.show()

# Find best parameters
# Filter out -inf values first to avoid selecting errored combinations
valid_results = results_df[results_df['mean_log_likelihood'] > -np.inf]
if len(valid_results) > 0:
    best_result = valid_results.loc[valid_results['mean_log_likelihood'].idxmax()]
else:
    # Fallback to a reasonable default if all combinations failed
    best_result = pd.Series({'n_splines': 10, 'lambda': 0.1, 'mean_log_likelihood': -np.inf})

# Display best smoothing parameters
print("Best Smoothing Parameters:")
print(f"Number of Splines: {best_result['n_splines']}")
print(f"Lambda (Smoothing Parameter): {best_result['lambda']}")
print(f"Mean Log-Likelihood: {best_result['mean_log_likelihood']:.4f}")

# Fit model with best parameters
try:
    best_gam = LogisticGAM(
        s(0, n_splines=int(best_result['n_splines']), spline_order=3, lam=best_result['lambda']) + 
        s(1, n_splines=int(best_result['n_splines']), spline_order=3, lam=best_result['lambda']) + 
        s(2, n_splines=int(best_result['n_splines']), spline_order=3, lam=best_result['lambda'])
    ).fit(X, y)

    # Plot partial dependence with optimal smoothing
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, var in enumerate(continuous_vars):
        # Generate prediction grid
        XX = np.zeros((100, 3))
        XX[:, i] = np.linspace(X[:, i].min(), X[:, i].max(), 100)

        # Set other variables to their means
        for j in range(3):
            if j != i:
                XX[:, j] = X[:, j].mean()

        # Plot the partial effect
        pdep = best_gam.partial_dependence(term=i, X=XX)
        axes[i].plot(XX[:, i], pdep, 'r-', label='Optimal Smoothing', linewidth=2)

        # Alternative approach for confidence intervals using model.std_err_
        try:
            # Check if std_err_ attribute exists
            if hasattr(best_gam, 'std_err_'):
                # We can approximate confidence intervals using standard errors
                # and the shape of the partial dependence
                se = best_gam.std_err_
                # This is a simplified approach - in practice CIs are more complex
                # but this gives a rough visualization
                # Scale the std_err to make it visible on the plot
                scale_factor = 1.0
                pdep_se = np.std(pdep) * scale_factor

                axes[i].fill_between(
                    XX[:, i], 
                    pdep - 1.96 * pdep_se,  # Approximate 95% CI
                    pdep + 1.96 * pdep_se, 
                    alpha=0.2, 
                    color='r',
                    label='Approx. 95% CI'
                )
            else:
                # Alternative - use bootstrap for a crude CI estimate
                # This is a simplified version for visualization purposes
                n_bootstrap = 10
                bootstrap_curves = []

                for _ in range(n_bootstrap):
                    # Create a bootstrap sample - sample with replacement
                    boot_idx = np.random.choice(len(X), size=len(X), replace=True)
                    X_boot, y_boot = X[boot_idx], y[boot_idx]

                    # Fit a model on the bootstrap sample
                    boot_gam = LogisticGAM(
                        s(0, n_splines=int(best_result['n_splines']), spline_order=3, lam=best_result['lambda']) + 
                        s(1, n_splines=int(best_result['n_splines']), spline_order=3, lam=best_result['lambda']) + 
                        s(2, n_splines=int(best_result['n_splines']), spline_order=3, lam=best_result['lambda'])
                    ).fit(X_boot, y_boot)

                    # Get partial dependence for this bootstrap
                    boot_pdep = boot_gam.partial_dependence(term=i, X=XX)
                    bootstrap_curves.append(boot_pdep)

                # Calculate confidence bounds from bootstrapped curves
                bootstrap_curves = np.array(bootstrap_curves)
                lower = np.percentile(bootstrap_curves, 2.5, axis=0)
                upper = np.percentile(bootstrap_curves, 97.5, axis=0)

                axes[i].fill_between(
                    XX[:, i], 
                    lower, 
                    upper, 
                    alpha=0.2, 
                    color='r',
                    label='Bootstrap 95% CI'
                )

        except Exception as e:
            print(f"Could not plot confidence intervals for {var} using alternative method: {e}")

        # Add a scatter plot of binned data for comparison
        try:
            # Group data into bins
            bins = np.linspace(data[var].min(), data[var].max(), 10)
            groups = pd.cut(data[var], bins)

            # Calculate mean predictor and proportion of 1's in each bin
            bin_summary = data.groupby(groups, observed=False)[['Churn', var]].agg({
                'Churn': 'mean', 
                var: 'mean'
            }).reset_index(drop=True)

            # Plot binned points
            axes[i].scatter(bin_summary[var], bin_summary['Churn'], color='blue', alpha=0.7, label='Data Bins')
        except Exception as e:
            print(f"Could not plot binned data for {var}: {e}")

        # Add a linear fit line for comparison
        try:
            x_vals = XX[:, i]
            y_vals = best_gam.predict_proba(XX)

            # Simple linear fit for comparison
            coef = np.polyfit(x_vals, y_vals, 1)
            poly_fit = np.poly1d(coef)

            axes[i].plot(x_vals, poly_fit(x_vals), 'g--', linewidth=1.5, label='Linear Trend')
        except Exception as e:
            print(f"Could not plot linear trend for {var}: {e}")

        # Add a title and labels
        axes[i].set_title(f'GAM Partial Effect for {continuous_vars[i]} (Optimal Smoothing)')
        axes[i].set_xlabel(continuous_vars[i])
        axes[i].set_ylabel('Partial Effect on Churn Probability')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout()
    plt.suptitle('GAM Partial Effects with Optimal Smoothing Parameters', y=1.05, fontsize=16)
    plt.show()

    # Print summary of the best model
    print("\nBest GAM Model Summary:")
    print(best_gam.summary())

    # Print interpretations
    print("\nInterpretation of Partial Effects:")
    for i, var in enumerate(continuous_vars):
        # Generate prediction grid for interpretation
        XX_interp = np.zeros((3, 3))
        XX_interp[:, i] = [X[:, i].min(), X[:, i].mean(), X[:, i].max()]

        # Set other variables to their means
        for j in range(3):
            if j != i:
                XX_interp[:, j] = X[:, j].mean()

        # Get predictions at min, mean, and max values
        pdep_interp = best_gam.partial_dependence(term=i, X=XX_interp)

        print(f"\n{var}:")
        print(f"  - At minimum value ({X[:, i].min():.2f}): Partial effect = {pdep_interp[0]:.4f}")
        print(f"  - At mean value ({X[:, i].mean():.2f}): Partial effect = {pdep_interp[1]:.4f}")
        print(f"  - At maximum value ({X[:, i].max():.2f}): Partial effect = {pdep_interp[2]:.4f}")

        # Direction of effect
        if pdep_interp[2] > pdep_interp[0]:
            print(f"  - Direction: Positive effect (higher {var} → higher probability of churn)")
        elif pdep_interp[2] < pdep_interp[0]:
            print(f"  - Direction: Negative effect (higher {var} → lower probability of churn)")
        else:
            print(f"  - Direction: No clear effect")

        # Non-linearity assessment
        mid_point = (pdep_interp[0] + pdep_interp[2]) / 2
        if abs(pdep_interp[1] - mid_point) < 0.01:
            print(f"  - Linearity: Effect appears approximately linear")
        else:
            print(f"  - Linearity: Effect shows non-linear behavior")

except Exception as e:
    print(f"Error fitting best model: {e}")
    print("Try running with different parameter combinations.")


# ### Concurvity Assumption Check for GAM

# In[52]:


# Continuous predictors
continuous_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Prepare data
X = data[continuous_vars].values
y = data['Churn'].values

# Fit a multivariate GAM for the main model
gam = LogisticGAM(s(0, n_splines=10) + s(1, n_splines=10) + s(2, n_splines=10)).fit(X, y)

# Calculate pairwise correlations between features
corr_matrix = np.corrcoef(X, rowvar=False)

# Create a heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=continuous_vars, yticklabels=continuous_vars)
plt.title('Feature Correlation Matrix (Potential Concurvity)')
plt.tight_layout()
plt.show()

# Manual check for concurvity: Try to predict each feature from the others
concurvity_results = []

for i, target_var in enumerate(continuous_vars):
    # Variables to use as predictors
    predictor_vars = [var for var in continuous_vars if var != target_var]

    # Prepare data
    X_predictors = data[predictor_vars].values
    y_target = data[target_var].values

    # Standardize the target for easier interpretation
    y_target_std = (y_target - y_target.mean()) / y_target.std()

    # Fit a GAM (not LogisticGAM) to predict the continuous target variable
    try:
        # Use regular GAM instead of LogisticGAM for continuous prediction
        predictor_gam = GAM(s(0, n_splines=10) + s(1, n_splines=10)).fit(X_predictors, y_target_std)

        # Calculate R-squared as a measure of concurvity
        y_pred = predictor_gam.predict(X_predictors)
        ss_total = np.sum((y_target_std - y_target_std.mean()) ** 2)
        ss_residual = np.sum((y_target_std - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        concurvity_results.append({
            'Target': target_var,
            'Predictors': ', '.join(predictor_vars),
            'R-squared': r_squared
        })
    except Exception as e:
        print(f"Error modeling {target_var}: {str(e)}")
        # Add a fallback result
        concurvity_results.append({
            'Target': target_var,
            'Predictors': ', '.join(predictor_vars),
            'R-squared': np.nan
        })

# Display concurvity results
concurvity_df = pd.DataFrame(concurvity_results)

# Handle any NaN values
concurvity_df = concurvity_df.fillna(0)

# Create a bar chart of R-squared values
plt.figure(figsize=(12, 6))
plt.bar(concurvity_df['Target'], concurvity_df['R-squared'], color='skyblue')
plt.axhline(y=0.7, color='r', linestyle='--', label='High Concurvity Threshold (R² > 0.7)')
plt.title('Concurvity Check: R-squared when Predicting Each Feature from Others')
plt.xlabel('Target Variable')
plt.ylabel('R-squared')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Concurvity Check Results:")
print(concurvity_df)
print("\nInterpretation:")
print("R-squared values close to 1 indicate high concurvity (similar to multicollinearity).")
print("High concurvity can make it difficult to interpret individual smooth terms.")
print("Consider removing or combining highly concurve variables.")


# ### Complexity and Performance Assumption Check for GAM

# In[53]:


# Ensure Churn is in binary format (0/1)
if data['Churn'].dtype == 'object':
    print("Converting Churn to binary format...")
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    print(f"Unique values in Churn after conversion: {data['Churn'].unique()}")

# Prepare data
X = data.drop('Churn', axis=1).values
y = data['Churn'].values

# Print info about the data
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Unique values in y: {np.unique(y)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define different complexity levels for GAM
complexity_levels = [
    {
        'name': 'Linear Terms Only',
        'model': LogisticGAM(l(0) + l(1) + l(2)),
        'color': 'blue'
    },
    {
        'name': 'Small Splines (n=4)',
        'model': LogisticGAM(s(0, n_splines=4, spline_order=3) + 
                            s(1, n_splines=4, spline_order=3) + 
                            s(2, n_splines=4, spline_order=3)),
        'color': 'green'
    },
    {
        'name': 'Medium Splines (n=7)',
        'model': LogisticGAM(s(0, n_splines=7, spline_order=3) + 
                            s(1, n_splines=7, spline_order=3) + 
                            s(2, n_splines=7, spline_order=3)),
        'color': 'orange'
    },
    {
        'name': 'Large Splines (n=15)',
        'model': LogisticGAM(s(0, n_splines=15, spline_order=3) + 
                            s(1, n_splines=15, spline_order=3) + 
                            s(2, n_splines=15, spline_order=3)),
        'color': 'red'
    }
]

# Store results
results = []

# Train models with different complexities
for config in complexity_levels:
    try:
        model = config['model']
        model.fit(X_train, y_train)

        # Calculate metrics
        train_ll = model.loglikelihood(X_train, y_train)
        test_ll = model.loglikelihood(X_test, y_test)

        # Print available statistics keys for debugging
        print(f"Available statistics for {config['name']}: {list(model.statistics_.keys())}")

        # Safely get AIC and calculate BIC if not available
        aic = model.statistics_.get('AIC', None)

        # If AIC is not available, calculate it
        if aic is None:
            n = len(y_train)
            k = model.statistics_['edof']  # effective degrees of freedom
            aic = -2 * train_ll + 2 * k

        # Calculate BIC manually if not available
        # BIC = -2 * loglikelihood + log(n) * k
        n = len(y_train)
        k = model.statistics_['edof']  # effective degrees of freedom
        bic = -2 * train_ll + np.log(n) * k

        # Get effective degrees of freedom
        edf = model.statistics_['edof']

        results.append({
            'name': config['name'],
            'train_loglikelihood': train_ll,
            'test_loglikelihood': test_ll,
            'AIC': aic,
            'BIC': bic,
            'EDF': edf,
            'color': config['color']
        })
    except Exception as e:
        print(f"Error with model {config['name']}: {e}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

if len(results_df) > 0:
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training vs Testing Log-Likelihood
    bar_width = 0.35
    x = np.arange(len(results_df))
    axes[0, 0].bar(x - bar_width/2, results_df['train_loglikelihood'], 
                width=bar_width, alpha=0.6, label='Training')
    axes[0, 0].bar(x + bar_width/2, results_df['test_loglikelihood'], 
                width=bar_width, alpha=0.6, label='Testing')
    axes[0, 0].set_title('Log-Likelihood by Model Complexity')
    axes[0, 0].set_xlabel('Model Complexity')
    axes[0, 0].set_ylabel('Log-Likelihood')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(results_df['name'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # AIC and BIC
    axes[0, 1].plot(results_df['EDF'], results_df['AIC'], 'o-', color='blue', label='AIC')
    axes[0, 1].plot(results_df['EDF'], results_df['BIC'], 'o-', color='red', label='BIC')

    # Add labels with offset positions to avoid overlap
    for i, row in results_df.iterrows():
        # Add offset to position based on model name to avoid overlap
        y_offset = 10 if i % 2 == 0 else -30
        x_offset = -20 if i == 0 else 0  # Additional offset for first point

        axes[0, 1].annotate(row['name'], 
                           (row['EDF'], row['AIC']), 
                           textcoords="offset points", 
                           xytext=(x_offset, y_offset), 
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    axes[0, 1].set_title('Information Criteria by Effective Degrees of Freedom')
    axes[0, 1].set_xlabel('Effective Degrees of Freedom (EDF)')
    axes[0, 1].set_ylabel('Criterion Value (Lower is Better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training vs Testing Log-Likelihood by EDF
    axes[1, 0].plot(results_df['EDF'], results_df['train_loglikelihood'], 'o-', 
                    color='blue', label='Training')
    axes[1, 0].plot(results_df['EDF'], results_df['test_loglikelihood'], 'o-', 
                    color='green', label='Testing')

    # Add labels with offset positions to avoid overlap
    for i, row in results_df.iterrows():
        # Add offset to position based on model name to avoid overlap
        y_offset = 10 if i % 2 == 0 else -30
        x_offset = -20 if i == 0 else 0  # Additional offset for first point

        axes[1, 0].annotate(row['name'], 
                           (row['EDF'], row['train_loglikelihood']), 
                           textcoords="offset points", 
                           xytext=(x_offset, y_offset), 
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    axes[1, 0].set_title('Log-Likelihood by Effective Degrees of Freedom')
    axes[1, 0].set_xlabel('Effective Degrees of Freedom (EDF)')
    axes[1, 0].set_ylabel('Log-Likelihood')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot partial effects for model with best test log-likelihood
    best_idx = results_df['test_loglikelihood'].idxmax()
    best_model_name = results_df.loc[best_idx, 'name']
    best_model = complexity_levels[int(best_idx)]['model']

    # Create a better partial effects plot that doesn't use twinx
    try:
        continuous_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']
        colors = ['blue', 'green', 'red']

        # Create a single subplot for all three effects
        for i, var in enumerate([0, 1, 2]):
            # Generate prediction grid for this variable
            XX = np.zeros((100, X.shape[1]))
            XX[:, var] = np.linspace(X[:, var].min(), X[:, var].max(), 100)

            # Set other variables to their means
            for j in range(X.shape[1]):
                if j != var:
                    XX[:, j] = X[:, j].mean()

            # Normalize the x-axis for plotting
            x_values = (XX[:, var] - X[:, var].min()) / (X[:, var].max() - X[:, var].min())

            # Plot partial effect with normalized x-axis
            effect = best_model.partial_dependence(term=var, X=XX)
            axes[1, 1].plot(x_values, effect, '-', 
                         label=continuous_vars[i], color=colors[i], linewidth=2)

        axes[1, 1].set_title(f'Partial Effects for Best Model: {best_model_name}')
        axes[1, 1].set_xlabel('Normalized Feature Value (0=min, 1=max)')
        axes[1, 1].set_ylabel('Partial Effect')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3)

    except Exception as e:
        print(f"Error plotting partial effects: {e}")
        axes[1, 1].text(0.5, 0.5, f"Error plotting partial effects: {str(e)}", 
                        ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.suptitle('GAM Model Complexity vs. Performance Analysis', y=1.02, fontsize=16)
    plt.show()

    # Display results table
    print("Model Complexity vs. Performance Results:")
    print(results_df[['name', 'train_loglikelihood', 'test_loglikelihood', 'AIC', 'BIC', 'EDF']])
    print("\nInterpretation:")
    print("- The best model typically balances complexity (EDF) with performance.")
    print("- Look for the model with the lowest AIC/BIC or highest test log-likelihood.")
    print(f"- Best model based on test log-likelihood: {best_model_name}")

    # Add more detailed interpretation
    print("\nDetailed observations:")

    # Compare EDF values
    if results_df['EDF'].nunique() == 1:
        print("- All models have the same effective degrees of freedom (EDF), suggesting similar complexity.")
    else:
        print("- Models show different levels of complexity as measured by EDF.")

    # Look at likelihood differences
    ll_diff = results_df['train_loglikelihood'].max() - results_df['test_loglikelihood'].max()
    if abs(ll_diff) < 0.01:
        print("- Training and testing log-likelihoods are very similar, suggesting good generalization.")
    elif ll_diff > 0:
        print("- Training log-likelihood is higher than testing, suggesting some overfitting.")
    else:
        print("- Testing log-likelihood is higher than training, which is unusual and may indicate data issues.")

    # Compare AIC/BIC values
    aic_range = results_df['AIC'].max() - results_df['AIC'].min()
    if aic_range < 1:
        print("- Very small differences in AIC values suggest that models are similarly effective.")

    # Final recommendation
    best_aic_idx = results_df['AIC'].idxmin()
    best_aic_model = results_df.loc[best_aic_idx, 'name']

    if best_aic_model == best_model_name:
        print(f"- Both AIC and test log-likelihood agree that {best_model_name} is the best model.")
    else:
        print(f"- AIC suggests {best_aic_model} as the best model, while test log-likelihood suggests {best_model_name}.")
        print("- Consider your priorities: parsimony (AIC) or predictive performance (test log-likelihood).")
else:
    print("No models were successfully trained. Please check the errors above.")


# ### Interpretation for GAM Assumptions EDA
# 
# - **Concurvity**: continuous variables *(tenure, MonthlyCharges, TotalCharges)* show extremely high concurvity (R² > 0.91); This makes individual effects harder to interpret. removing TotalCharges might help while keeping monthly charges and tenure. 
# 
# - **Simpler Model Structure**: The complexity analysis shows minimal performance differences between linear terms and spline models *(nearly identical AIC/BIC values)*, but this should consider simpler models limitations as well(Logistic and Lienar Models above)
# 
# - **Non-Linearity**: spline transformations needed for for *tenure (shows strong negative non-linear effect)* and *MonthlyCharges (shows positive non-linear effect)* to accurately obtain their relationships with dependent variable churn. 

# # Task 2

# **Linear Regression: Treat the churn variable as a continuous variable (e.g., 0 for staying, 1 for churning) and build a linear regression model to predict churn. Interpret the coefficients and assess the model's performance.**
# 

# In[54]:


# print all column names
for column in df.columns:
    print(column)


# In[145]:


X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# X['log_tenure'] = np.log1p(X['tenure'])
X['log_TotalCharges'] = np.log1p(X['TotalCharges'])
X['log_MonthlyCharges'] = np.log1p(X['MonthlyCharges'])

# Removing Internet Services & Low explainability variables
X = X.drop(['InternetService', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'OnlineBackup', 'Partner', 'DeviceProtection', 'StreamingTV'], axis=1)
# X = X.drop(['tenure', 'MonthlyCharges', 'TotalCharges'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[146]:


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return model, y_pred


# In[147]:


# Fit models
lr_model, lr_pred = evaluate_model(LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression")


# In[148]:


model = sm.OLS(y, X).fit()
print(model.summary())


# In[166]:


def plot_regression_diagnostics(y_test, y_pred, model_name):
    residuals = y_test - y_pred

    # Create figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Residual Plot (left subplot)
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'Residual Plot - {model_name}')
    ax1.axhline(y=0, color='r', linestyle='--')

    # Add a LOWESS smoother to check for patterns
    lowess_result = lowess(residuals, y_pred, frac=0.6)
    ax1.plot(lowess_result[:, 0], lowess_result[:, 1], 'g-', linewidth=2, label='LOWESS Trend')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q Plot (right subplot)
    stats.probplot(residuals, plot=ax2)
    ax2.set_title(f'Q-Q Plot - {model_name}')
    ax2.grid(True, alpha=0.3)

    # Add statistical test results as text
    shapiro_test = stats.shapiro(residuals)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        'Shapiro-Wilk Test:',
        f'W={shapiro_test[0]:.4f}',
        f'p-value={shapiro_test[1]:.4f}'
    ))
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.suptitle(f'Regression Diagnostics: {model_name}', y=1.05, fontsize=16)
    plt.show()

    # Additional statistics
    print(f"\nDiagnostic Statistics - {model_name}:")
    print(f"Mean of Residuals: {np.mean(residuals):.4f}")
    print(f"Standard Deviation of Residuals: {np.std(residuals):.4f}")

    # Heteroscedasticity test
    try:
        X_bp = sm.add_constant(y_pred)
        bp_test = het_breuschpagan(residuals, X_bp)
        print(f"\nBreusch-Pagan Test for Heteroscedasticity:")
        print(f"LM Statistic: {bp_test[0]:.4f}")
        print(f"p-value: {bp_test[1]:.4f}")

    except Exception as e:
        print(f"Could not perform heteroscedasticity test: {str(e)}")

# Example usage - only for Linear Regression
plot_regression_diagnostics(y_test, lr_pred, 'Linear Regression')


# In[164]:


def plot_model_evaluation(y_true, y_pred, model_name):
    # Create a figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion Matrix (left subplot)
    cm = confusion_matrix(y_true, np.round(y_pred))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax1
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title(f"Confusion Matrix - {model_name}")

    # ROC Curve (right subplot)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    ax2.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"ROC Curve - {model_name}")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.suptitle(f"Model Evaluation: {model_name}", y=1.05, fontsize=16)
    plt.show()


    return roc_auc

# RPint ROC and COnfusion Matrix for Linear Regression
roc_auc_lr = plot_model_evaluation(y_test, lr_pred, "Linear Regression")


# ### Coefficient Interpretation Linear Regression
# 
# | Variable | Interpretation |
# |----------|----------------|
# | SeniorCitizen | Being a senior citizen increases the probability of churning by about 6.1%. This is statistically significant (p < 0.001), which means senior citizens are more likely to leave compared to non-seniors. |
# | Dependents | Having dependents slightly decreases the likelihood of churning by 1.7%. However, this isn't statistically significant (p = 0.09), so we can't be confident that dependents actually impact churn rates. |
# | PhoneService | Having phone service decreases churn probability by about 10.9%. This is highly significant (p < 0.001), suggesting customers with phone service are more loyal to the company. |
# | MultipleLines | Customers with multiple lines are 5.6% more likely to churn. This effect is statistically significant (p < 0.001), which might indicate that customers with more complex services find more reasons to leave. |
# | OnlineSecurity | Having online security decreases churn by 9.6%. This significant effect (p < 0.001) suggests that customers who feel their data is secure are more likely to stay with the company. |
# | TechSupport | Tech support reduces churn probability by 9.5%. This significant effect (p < 0.001) indicates that customers who receive technical assistance are more satisfied and less likely to leave. |
# | StreamingMovies | Customers with streaming movies are 2.5% more likely to churn. This is statistically significant (p = 0.019), though the effect is smaller than other services. |
# | Contract | Each step up in contract length (from month-to-month to one year to two years) decreases churn by 5.2%. This significant effect (p < 0.001) shows longer contracts help retain customers. |
# | PaperlessBilling | Customers with paperless billing are 5.6% more likely to churn. This significant effect (p < 0.001) might suggest these customers are more tech-savvy and willing to switch providers. |
# | PaymentMethod | Each step up in payment method reduces churn probability by 2.5%. This significant effect (p < 0.001) suggests certain payment methods are associated with higher customer loyalty. |
# | log_TotalCharges | A 1% increase in total charges is associated with a 0.11% decrease in churn probability. This highly significant effect (p < 0.001) suggests customers who have spent more with the company over time are less likely to leave. |
# | log_MonthlyCharges | A 1% increase in monthly charges is associated with a 0.29% increase in churn probability. This highly significant effect (p < 0.001) shows that customers with higher monthly bills are much more likely to churn. |

# ### Linear Regression Model Results Interpretation
# | Model Element | Interpretation |
# |----------|----------------|
# | ROC Score | A value of .83 indicates the model has a solid in predicting Churn |
# | R^2 Test | A value of .2567 suggests the model performs just moderately on unseen data |
# | R^2  | A value of .475 suggests the explains 47.5% of the variability in Churn |
# | Auto-Correlation  | After careful variable selection the model did not identify auto correlation with a Durbin Watson Score of 2.01 |
# | Heteroscedasticity | The Breusch-Pagan Test for Heteroscedasticity(LM Statistic: 199.0076 p-value: 0.0000) suggests theere is Heteroscedasticity in residuals |
# | Normality | The Shapiro-Wilk test result (W = 0.9588) indicates there could be a mild deviation from normality in the model residuals. This deviation is observed in the QQ plot where the blue dots slightly deviate from the red line at the tails indicating a relative skew on the data |
# 

# # Logistic Regression

# In[172]:


X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# X['log_tenure'] = np.log1p(X['tenure'])
X['log_TotalCharges'] = np.log1p(X['TotalCharges'])
X['log_MonthlyCharges'] = np.log1p(X['MonthlyCharges'])

# Remove High VIF & Low explainability variables
X = X.drop(['InternetService', 'MonthlyCharges', 'TotalCharges', 'gender', 'OnlineBackup', 'Partner', 'DeviceProtection', 'StreamingTV', 'tenure'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[173]:


# Fit the logistic regression model
logit_model = sm.Logit(y, X).fit()

# Print the summary of the model
print(logit_model.summary())


# In[182]:


# print odds given coefficients of each variable showing increase and decrease in odds 
odds_ratios = pd.Series(np.exp(logit_model.params) - 1, index=X.columns)

# Add them to bar charts displayed horizontally
plt.figure(figsize=(10, 6))
odds_ratios.sort_values().plot(kind='barh', color='skyblue')
plt.title('Odds Ratios for each Feature in Logistic Regression')
plt.xlabel('Odds Ratio Increase/Decrease')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ### Coeficient Interpretation in Logistic Model

# 
# | Variable | Interpretation |
# |----------|----------------|
# | SeniorCitizen | Being a senior citizen increases the odds of churning by about 41%. This is statistically significant (p < 0.001), showing senior citizens are more likely to leave than non-seniors. |
# | Dependents | Having dependents decreases the odds of churning by about 21%. This is statistically significant (p = 0.004), suggesting customers with dependents tend to be more loyal. |
# | PhoneService | Having phone service decreases the odds of churning by about 55%. This is highly significant (p < 0.001), indicating customers with phone service are much less likely to leave. |
# | MultipleLines | Having multiple lines increases the odds of churning by about 100%. This significant effect (p < 0.001) suggests customers with more complex phone services are twice as likely to churn. |
# | OnlineSecurity | Having online security decreases the odds of churning by about 29%. This significant effect (p < 0.001) shows security services help retain customers. |
# | TechSupport | Having tech support decreases the odds of churning by about 25%. This significant effect (p = 0.001) indicates technical assistance helps keep customers satisfied. |
# | StreamingMovies | Having streaming movies increases the odds of churning by about 84%. This is highly significant (p < 0.001), suggesting this service may not meet customer expectations. |
# | Contract | Each step up in contract length (from month-to-month to one year to two years) decreases the odds of churning by about 62%. This very significant effect (p < 0.001) confirms longer contracts strongly reduce churn. |
# | PaperlessBilling | Having paperless billing increases the odds of churning by about 61%. This significant effect (p < 0.001) suggests paperless billing customers may be more likely to shop around. |
# | PaymentMethod | Each step up in payment method decreases the odds of churning by about 18%. This significant effect (p < 0.001) indicates certain payment methods are associated with higher loyalty. |
# | log_TotalCharges | A 1% increase in total charges is associated with a 0.59% decrease in the odds of churning. This highly significant effect (p < 0.001) shows customers who have spent more over time are less likely to leave. |
# | log_MonthlyCharges | A 1% increase in monthly charges is associated with a 0.89% increase in the odds of churning. This highly significant effect (p < 0.001) confirms that higher monthly bills strongly increase churn risk. |

# ### Model Comparison 
# **Linear Model** is statistically significant but has some violations of regression assumptions(Heteroscedasticity and Normality on errors). Good discriminative ability(AUC and accuracy) but it presents only moderate ability in its explanatory power.

# 
