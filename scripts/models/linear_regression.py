import numpy as np


class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, features, target):
        """
        Fit the linear regression model to the provided features and target values.

        Args:
            features (ndarray): The feature matrix of shape (n_samples, n_features).
            target (ndarray): The target values of shape (n_samples,).

        Raises:
            ValueError: If the model has not been trained yet.
        """
        # Add a column of ones to X for the intercept term
        features_with_intercept = np.column_stack((np.ones(len(features)), features))

        # Calculate theta using the normal equation
        self.theta = (
            np.linalg.inv(features_with_intercept.T @ features_with_intercept)
            @ features_with_intercept.T
            @ target
        )

    def predict(self, X):
        """
        Make predictions using the fitted linear regression model.

        Args:
            X (ndarray): The feature matrix of shape (n_samples, n_features) for which
                predictions are to be made.

        Returns:
            ndarray: Predicted target values for the input features.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet.")

        # Add a column of ones to X for the intercept term
        X_with_intercept = np.column_stack((np.ones(len(X)), X))

        # Make predictions
        predictions = X_with_intercept @ self.theta

        return predictions

    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE) between the actual target values and the predicted values.

        Parameters:
            y_true (array): The actual target values.
            y_pred (array): The predicted values.

        Returns:
            float: The Mean Squared Error (MSE).
        """
        # Compute the squared differences between actual and predicted values
        squared_errors = (y_true - y_pred) ** 2

        # Compute the mean of squared errors
        mse = np.mean(squared_errors)

        return mse
