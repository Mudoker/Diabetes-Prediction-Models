import pandas as pd
import tabulate

NEKO_ART = r"""

    /\_____/\
   /  o   o  \
  ( ==  ^  == )       Neko has arrived!
   )         (        An data visualizing extension for analyzing DataFrames.
  (           )       Art: https://www.asciiart.eu/animals/cats.
 ( (  )   (  ) )
(__(__)___(__)__)
"""


class Neko:
    """
    A class to analyze DataFrames.

    Attributes:
        None

    Methods:
        essense(data): Prints an overview of the provided DataFrame.

        is_nan(data): Check if the DataFrame contains NaN values.

        is_data_in_range(data, column, min_val, max_val): Check if the values in the specified column are within the provided range.

        is_data_valid(data, column, valid_values): Check if the values in the specified column are within the provided list of valid values.
    """

    def __init__(self):
        print(NEKO_ART)

    def greet(self):
        """
        Prints a greeting message.

        Parameters:
            None

        Returns:
            None
        """
        print(NEKO_ART)

    def essense(self, data, stralign="left", tablefmt="rounded_grid"):
        """
        Prints an overview of the provided DataFrame.

        Parameters:
            data (DataFrame): The DataFrame to analyze.

        Returns:
            String: A summary of the DataFrame.
        """
        payload = {
            "Number of Rows": data.shape[0],
            "Number of Columns": data.shape[1],
            "Data Types": data.dtypes.unique().astype(str).tolist(),
            "Total Missing Values": data.isnull().sum().sum(),
            "Columns with Missing Values": data.columns[data.isnull().any()].tolist(),
            "Number of Duplicates": data.duplicated().sum(),
            "Memory Usage (MB)": round(
                data.memory_usage(index=True).sum() / (1024**2), 1
            ),
        }

        summary_table = [[key, value] for key, value in payload.items()]
        tabulated_summary = tabulate.tabulate(
            summary_table,
            headers=["Attribute", "Value"],
            tablefmt=tablefmt,
            stralign=stralign,
            showindex=True,
        )

        return tabulated_summary

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

    def feature_essence(self, data, feature, stralign="left", tablefmt="rounded_grid"):
        """
        Prints an overview of the provided feature in the DataFrame.

        Parameters:
            data (DataFrame): The DataFrame to analyze.
            feature (String): The feature to analyze.

        Returns:
            String: A summary of the feature.
        """
        payload = {
            "Feature": feature,
            "Data Type": data[feature].dtype,
            "Total Values": data[feature].count(),
            "Unique Values": data[feature].nunique(),
            "Total Missing Values": data[feature].isnull().sum(),
            "Minimum Value": data[feature].min(),
            "Maximum Value": data[feature].max(),
            "Mean": data[feature].mean(),
            "Median": data[feature].median(),
            "Standard Deviation": data[feature].std(),
            "Skewness": data[feature].skew(),
            "IQR": data[feature].quantile(0.75) - data[feature].quantile(0.25),
            "Memory Usage (MB)": round(data[feature].memory_usage() / (1024**2), 1),
        }

        summary_table = [[key, value] for key, value in payload.items()]

        tabulated_summary = tabulate.tabulate(
            summary_table,
            headers=["Attribute", "Value"],
            tablefmt=tablefmt,
            stralign=stralign,
            showindex=True,
        )

        return tabulated_summary
