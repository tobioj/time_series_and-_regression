# House Price Prediction App
This repository contains a simple house price prediction app built using Linear Regression. The app allows users to input various features of a house, such as the number of bedrooms, bathrooms, square footage, etc., and predicts the price of the house based on these features.

## Contents
Introduction
Installation
Usage
Files
Contributing
License
## Introduction
House price prediction is a common problem in real estate and finance. This project provides a streamlined solution for predicting house prices using a Linear Regression model trained on a dataset containing various house features and their corresponding prices.

## Installation
To run this app, ensure you have Python installed on your system. Install the required dependencies using pip:

`pip install numpy pandas streamlit`
## Usage
To use the app, follow these steps:

Clone the repository to your local machine.
Navigate to the directory containing the code.
Run the following command in your terminal:

* streamlit run main.py
* Access the app through your web browser.

Files
main.py: Contains the Streamlit application code where users can input house details and get price predictions.
model/linear_regression.py: Implements the Linear Regression model used for price prediction.
evaluation.py: Contains functions for evaluating the performance of the model, such as calculating the Root Mean Squared Error (RMSE) and splitting the data into training and testing sets.
data_cleaned.csv: Dataset containing cleaned house data after removing outliers and preprocessing.
clean_data.ipynb: Jupyter Notebook containing the data cleaning and preprocessing steps.