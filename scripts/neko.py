import pandas as pd


class Neko:
    """
    A class to analyze DataFrames.

    Attributes:
        None

    Methods:
        essense(data): Prints an overview of the provided DataFrame.
    """

    def __init__(self):
        self.data = pd.DataFrame()

    def essense(self, data):
        """
        Prints an overview of the provided DataFrame.

        Parameters:
            data (DataFrame): The DataFrame to analyze.

        Returns:
            String: A summary of the DataFrame.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if data.empty:
            return "The DataFrame is empty."

        self.data = data
        return (
            " - Number of rows: "
            + str(data.shape[0])
            + "\n - Number of columns: "
            + str(data.shape[1])
            + "\n - Data types: "
            + str(data.dtypes.unique())
            + "\n - # Missing values: "
            + str(data.isnull().sum().sum())
            + " at columns: "
            + str(data.columns[data.isnull().any()].tolist())
            + "\n - # Duplicate rows: "
            + str(data.duplicated().sum())
            + "\n - Data size: "
            + str(round(data.memory_usage(index=True).sum() / (1024**2), 1))
            + " MB"
        )

    def is_nan(self, data):
        """
        Check if the DataFrame contains NaN values.

        Parameters:
            data (DataFrame): The DataFrame to analyze.

        Returns:
            String: A message indicating if the DataFrame contains NaN values.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if data.empty:
            return "The DataFrame is empty."

        try:
            data.apply(pd.to_numeric, errors="coerce")
        except ValueError:
            return "The DataFrame contains non-numeric values."

        if data.isnull().values.any():
            return data.isnull().sum()
        else:
            return "The DataFrame does not contain any NaN values."

    def is_data_in_range(self, data, column, min_val, max_val):
        """
        Check if the values in the specified column are within the provided range.

        Parameters:
            data (DataFrame): The DataFrame to analyze.
            column (String): The column to analyze.
            min_val (Float): The minimum value of the range.
            max_val (Float): The maximum value of the range.

        Returns:
            Object: A message indicating if the values are within the range.
        """
        # Error handling
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if data.empty:
            return "The DataFrame is empty."

        if column not in data.columns:
            return "The column does not exist in the DataFrame."

        if not isinstance(min_val, (int, float)) or not isinstance(
            max_val, (int, float)
        ):
            raise TypeError("min_val and max_val must be of type int or float.")

        if min_val > max_val:
            raise ValueError("min_val must be less than max_val.")

        if data[column].dtype not in ["int64", "float64"]:
            return "The column must contain numeric values."

        # Check if the values are within the range
        valid_values = data[column].between(min_val, max_val)

        if valid_values.all():
            return {
                "message": "All values are within the range.",
                "is_valid": True,
            }
        else:
            return {
                "message": f"{sum(~valid_values)} invalid values found.",
                "is_valid": False,
            }

    def is_data_valid(self, data, column, valid_values):
        """
        Check if the values in the specified column are within the provided list of valid values.

        Parameters:
            data (DataFrame): The DataFrame to analyze.
            column (String): The column to analyze.
            valid_values (List): The list of valid values.

        Returns:
            String: A message indicating if the values are within the list of valid values.
        """
        # Error handling
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if data.empty:
            return "The DataFrame is empty."

        if column not in data.columns:
            return "The column does not exist in the DataFrame."

        if not isinstance(valid_values, list):
            raise TypeError("valid_values must be a list.")

        if data[column].dtype not in ["int64", "float64"]:
            return "The column must contain numeric values."

        # Check if the values are within the list of valid values
        valid_values = data[column].isin(valid_values)

        if valid_values.all():
            return {
                "message": "All values are within the range.",
                "is_valid": True,
            }
        else:
            return {
                "message": f"{sum(~valid_values)} invalid values found.",
                "is_valid": False,
            }
