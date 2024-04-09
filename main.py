import numpy as np
import pandas as pd
from  models.regression_model import LinearRegression
import streamlit as st
# Sample data (replace with your actual DataFrame)
data = 'data_cleaned.csv'
df = pd.read_csv(data)
y = df['price'].values  # Assuming 'target_column' is the target variable

# Create and train the model
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']].values

# Normalize features (if needed)
X = (X - X.mean()) / X.std()

# Add bias term
X = np.c_[X, np.ones(X.shape[0])]

# Split data into training and testing sets (if needed)
# Then train the model
model = LinearRegression()
model.fit(X, y)


# Define the app
def app():
    st.title('House predicting app')
    st.write('input your house details')
    
    # Create input widgets for each feature
    bedrooms = st.slider("No_Bedroom", min_value = 1, max_value = 30, step = 1)
    bathrooms = st.slider("No_Bathroom", min_value = 1, max_value = 30, step = 1)
    sqft_living = st.number_input('Square feet living')
    sqft_lot = st.number_input('square feet lot')
    waterfront = st.selectbox("waterfront (1 is Yes 0 is No)", options =(data['waterfront'].unique()))
    view = st.selectbox("production year", options=(list( data['Prod. year'].unique())))
    condition = st.slider('condition (0 is worst, 1 is bad, 2 is fair, 3 is good, 4 is great )', 
    min_value=1, max_value=4, step=1)
    sqft_above= st.number_input("square feet above")
    sqft_basement=st.number_input("square feet of basement")
    yr_built= st.number_input('year built')
    yr_renovated = st.selectbox('year renovated', options=(list(data['yr_renovated'].unique())))
    floors =  st.slider("No_Floor", min_value = 1, max_value = 10, step = 1)
    bias = 0
    # set your values
    df_from_input = pd.DataFrame([{
    'bedrooms': bedrooms, 
    'bathrooms': bathrooms,
    'sqft_living': sqft_living, 
    'sqft_lot': sqft_lot,
    'floors': floors, 
    'waterfront': waterfront, 
    'view': view, 
    'condition': condition, 
    'sqft_above': sqft_above,
    'sqft_basement': sqft_basement, 
    'yr_built': yr_built, 
    'yr_renovated': yr_renovated,
    'bias': bias
  }])

    #price = model.predict(df_from_input)
    #return price

    # Display the predicted price to the user
    if st.button('Submit'):
        price = model.predict(df_from_input)
        st.success(f'your policy is {price[0]:,.2f} dollar.')
    
if __name__ == '_main_':
    app()
    
