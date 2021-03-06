# Salary-Prediction
Hiring is an integral part of any company. It's the people who make the company. It is really important to hire the right people for the right talent. This not only adds value to your compnay but also helps you plan financially. 

This Dataset consists of 3 files:
1. Train_features.csv - consists of features such as
2. Test_features.csv - consists of features similar to train_features.csv
3. Train_salaries.csv - consists of the target variable

<h3> File description</h4>

- Readme.md : This file.
- Basic EDA : Exploratory Data Analysis of the training dataset to identify dependencies, correlation amoung features and target, and distribution of features and target.
- Salary_prediction_modeling - Model building file

<h3>Goal</h3>

Predict the future employee salaries based on the data of current employees salaries by creating a robust machine learning model.

<h3> About the Dataset</h3>

 
- jobId : The ID of the job. 
- companyId : The ID of the company 
- jobType : The description of the job 
- degree : The degree of the employee 
- major : The education field of the employee 
- industry : The field to which the company belongs 
- yearsExperience : The employee years of experience on the job
- milesFromMetropolis : The distance in miles, the employee lives away from his/her workplace 
- salary (target) : Estimated Salary 

<h4> Numerical variables:</h4>

- yearsExperience
- milesfromMetropolis
- salary

<h4>Categorical variables:</h4>

- jobId
- companyId
- Jobtype
- degree
- major
- industry

<h3> Visualisation of the Target variable: Salary</h3>
<img src = "images/salary_distribution.png" width="800" height ="500">

We see a few outliers, upon further investigation they were reasonable and whichever weren't, were dropped.

<h4> Correlation between features and target</h4>

<img src = "images/industry.png" width="500" height ="300">
<img src = "images/Experience.png" width="500" height ="300">
<img src = "images/major.png" width="500" height ="300">
<img src = "images/Degree.png" width="500" height ="300">
<img src = "images/Distance.png" width="500" height ="300">


In general, we saw some correlation between the features and target as depicted below:

The third file, Salary_prediction_modeling consists of data cleaning, feature engineering and model building.

Three algorithms were picked for model building, and the model with the least MSE was selected as the final model.
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Summary of MSE of all three models:

| Model                        | MSE       |
| -----------------------------|:----------| 
| Linear Regression            | 386.97    | 
| Random Forest Regressor      | 387.29    |   
| Gradient Boosting Regressor  | 357.81  |   





