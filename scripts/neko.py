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