# Car Price Prediction: Entering the US Market

## Project Overview
This project models car prices based on various features, enabling a Chinese automobile company to understand how different factors influence car pricing in the U.S. market. The company aims to set up a manufacturing unit in the U.S. and compete with U.S. and European counterparts. The model will help the company design and price their cars strategically.

## Business Goal
The primary goal is to develop a regression model that can predict car prices based on independent variables (car features). The model will help the company understand the pricing dynamics of the U.S. automobile market and adjust business strategies accordingly.

## Dataset
The dataset consists of information about various cars sold in the U.S. market, with the following key features:

Numerical Features: wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg, and price.
Categorical Features: CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, enginetype, cylindernumber, and fuelsystem.
The dataset can be found here: https://drive.google.com/file/d/1FHmYNLs9v0Enc-UExEMpitOFGsWvB2dP/view?usp=drive_link.

## Repository Structure
├── CarPrice_Assignment.csv      # Dataset used for the project
├── car_price_prediction.ipynb   # Main Jupyter notebook containing the code
├── README.md                    # This file
└── requirements.txt             # Python dependencies

## Features
he project consists of five key components:

1. Data Loading and Preprocessing

Load the dataset and perform preprocessing steps such as handling missing values and encoding categorical features.
2. Model Implementation

Implement five regression algorithms to predict car prices:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Support Vector Regressor

3. Model Evaluation

Evaluate the models using the following metrics:
R-squared
Mean Squared Error (MSE)
Mean Absolute Error (MAE)

4.Feature Importance Analysis

Identify the most important features influencing car prices using the Random Forest model's feature importance.

5.Hyperparameter Tuning

Perform hyperparameter tuning using RandomizedSearchCV to improve model performance.

## Requirements
The following Python libraries are required to run the project:
pandas
numpy
scikit-learn



## Conclusion
TIn this project, we successfully built and evaluated several machine learning models to predict car prices in the US market. Among the five models implemented—Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and Support Vector Regressor—the Random Forest Regressor proved to be the best performer after hyperparameter tuning.

By optimizing the Random Forest model using RandomizedSearchCV, we identified the best hyperparameters:

n_estimators: 100
min_samples_split: 10
min_samples_leaf: 2
max_depth: None
bootstrap: True

These tuned hyperparameters improved the model’s performance significantly, achieving the following results:

R-squared (R²): 0.9434, indicating that the model explains 94.34% of the variance in car prices.
Mean Squared Error (MSE): 4,465,799, showing a relatively low average squared difference between predicted and actual car prices.
Mean Absolute Error (MAE): 1,462.31, demonstrating a low average absolute difference between predicted and actual prices.

The high R-squared value and low error metrics indicate that the tuned Random Forest model performs exceptionally well in predicting car prices based on the available features. Additionally, feature importance analysis highlighted key variables such as engine size, horsepower, and curb weight, which significantly influence car prices in the US market.

These insights will help the Chinese automobile company make informed decisions regarding car design, pricing strategies, and market positioning as they enter the competitive US automobile market.

## Clone the repository:
git clone https://github.com/vinilsbabu/Car-Price-Prediction-Entering-the-US-Market.git
cd Car-Price-Prediction-Entering-the-US-Market
