# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : PRATHAMESH MURKUTE

*INTERN ID* : CT06DF2317

*DOMAIN* : DATA ANALYTICS

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

## Predictive Analysis Using Machine Learning

# Project Overview
This project is developed as part of an internship task to demonstrate predictive analysis using machine learning. The objective is to build a regression model that can predict a target variable based on multiple features in a structured dataset. The project utilizes the College dataset (downloaded from Kaggle and commonly referenced as the Walmart dataset) and is implemented in Google Colab for seamless experimentation and reproducibility.

# Dataset
The dataset used for this analysis was obtained from Kaggle and uploaded into Google Drive for access via Google Colab. The dataset contains details about various colleges in the United States and includes both numerical and categorical features. The primary goal of the model is to predict the Out-of-State tuition fees (Outstate) for colleges based on other institutional features.

# Tools & Libraries
The following Python libraries were used for data analysis, modeling, and visualization:

Pandas: For data manipulation and loading.

Matplotlib & Seaborn: For visualizations like heatmaps and scatter plots.

Scikit-learn: For preprocessing, model training, and evaluation.

# Data Preprocessing
Loading Data: The CSV file was loaded into a Pandas DataFrame.

Missing Values: The dataset was checked for null or missing values using df.isnull().sum() to ensure data integrity.

Descriptive Statistics: A summary of data using df.describe() helped in understanding the feature distributions.

Label Encoding: The categorical column Private was converted to numerical format using LabelEncoder.

Feature Dropping: The column Unnamed: 0 was dropped as it served no analytical purpose.

Feature Engineering
Independent variables (X) were selected by excluding the target column Outstate.

The target variable (y) was defined as Outstate, the tuition fees charged to out-of-state students.

# Data Visualization
A correlation heatmap was generated using Seaborn to understand inter-feature relationships and influence on the target variable.

A scatter plot of actual vs predicted values was plotted to visualize model performance.

Model Building
A Linear Regression model from Scikit-learn was used for training. The steps included:

Splitting the dataset into training and testing sets (80/20 ratio).

Fitting the LinearRegression model on the training data.

Predicting the outcomes on the test set.

Evaluation Metrics
The model was evaluated using two metrics:

Mean Squared Error (MSE): Indicates the average squared difference between actual and predicted values.

R-squared (R²): Represents the proportion of variance explained by the model.

# Results:

MSE was displayed to assess error magnitude.

R² score was used to evaluate goodness-of-fit.

# Conclusion

This project demonstrates a simple yet effective application of machine learning in predicting numerical outcomes. It highlights the importance of preprocessing, data exploration, and model evaluation. The use of Linear Regression provided baseline results, and the visualizations aided in understanding model behavior.

Future work could involve testing more complex regression models (like Random Forests or Gradient Boosting), hyperparameter tuning, or feature selection methods to further enhance performance.
