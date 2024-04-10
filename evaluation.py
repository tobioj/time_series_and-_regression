import numpy as np
import pandas as pd
from models.regression_model import LinearRegression


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the actual and predicted values.

    Args:
        y_true (array-like): Array of true target values.
        y_pred (array-like): Array of predicted target values.

    Returns:
        float: The RMSE value.
    """
    # Convert inputs to NumPy arrays to ensure compatibility
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate squared differences
    squared_diff = (y_true - y_pred) ** 2
    
    # Calculate mean squared difference
    mean_squared_diff = np.mean(squared_diff)
    
    # Calculate RMSE
    rmse_value = np.sqrt(mean_squared_diff)
    
    return rmse_value

import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the data into random train and test subsets.

    Args:
        X (array-like): Features.
        y (array-like): Target variable.
        test_size (float or int, optional): Proportion of the dataset to include in the test split.
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples. Defaults to 0.2.
        random_state (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        tuple: Tuple containing the train-test split of inputs.
    """
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Determine the number of samples for the test set
    if isinstance(test_size, float):
        test_size = int(test_size * len(X))
    elif isinstance(test_size, int):
        if test_size >= len(X):
            raise ValueError("test_size should be less than the number of samples")
    else:
        raise ValueError("test_size should be a float or an int")

    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Split the data
    X_train = X[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    y_train = y[indices[:-test_size]]
    y_test = y[indices[-test_size:]]

    return X_train, X_test, y_train, y_test