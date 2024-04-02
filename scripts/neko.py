import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tabulate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import statsmodels.api as sm

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
            "Missing Values": data[feature].isnull().sum(),
            "Unique Values": data[feature].nunique(),
            "Minimum Value": data[feature].min(),
            "25th Percentile": data[feature].quantile(0.25),
            "Median": data[feature].median(),
            "75th Percentile": data[feature].quantile(0.75),
            "Maximum Value": data[feature].max(),
            "Mean": data[feature].mean(),
            "Standard Deviation": data[feature].std(),
            "Skewness": data[feature].skew(),
            "Kurtosis": data[feature].kurtosis(),
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

    def plot_pie_chart(
        self,
        ax,
        count,
        title,
        colors,
        title_fontsize=20,
        title_fontweight="semibold",
        axis_label_fontsize=15,
    ):
        """
        Plot a pie chart on the given axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the pie chart.
            count (pandas.Series): The data to be plotted.
            title (str): The title of the pie chart.
            colors (list): List of colors for each pie slice.
            title_fontsize (int, optional): Font size of the title. Default is 20.
            title_fontweight (str, optional): Font weight of the title. Default is "semibold".
            axis_label_fontsize (int, optional): Font size of the axis labels. Default is 15.
        """

        ax.pie(
            count,
            labels=count.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": axis_label_fontsize},
        )
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)

    def plot_histogram_with_polygon(
        self,
        data,
        ax,
        bins=10,
        edgecolor="#ff6172",
        alpha=0.7,
        color="#ffcacf",
        marker="o",
        linestyle="-",
        line_color="r",
        title="",
        xlabel="",
        ylabel="",
        title_fontsize=None,
        title_fontweight=None,
        axis_label_fontsize=None,
        index=False,
    ):
        """
        Plot a histogram with a frequency polygon on the same plot.

        Parameters:
        data (array-like): Input data.
        ax (Axes): Axes for plotting (optional).
        bins (int or sequence, optional): Histogram bin specification. Defaults to 10.
        edgecolor (color, optional): Histogram edge color. Defaults to '#ff6172'.
        alpha (float, optional): Histogram bar transparency. Defaults to 0.7.
        color (color, optional): Histogram bar color. Defaults to '#ffcacf'.
        marker (str, optional): Frequency polygon marker style. Defaults to 'o'.
        linestyle (str, optional): Frequency polygon line style. Defaults to '-'.
        line_color (color, optional): Frequency polygon line color. Defaults to 'r'.
        title (str, optional): Plot title. Defaults to ''.
        xlabel (str, optional): X-axis label. Defaults to ''.
        ylabel (str, optional): Y-axis label. Defaults to ''.
        title_fontsize (int, optional): Title font size.
        title_fontweight (str or int, optional): Title font weight.
        axis_label_fontsize (int, optional): Axis label font size.

        Returns:
        None
        """

        # Create a figure and axis if not provided
        if ax is None:
            fig, ax = plt.subplots()

        # Display data
        data = data.value_counts().sort_index()

        if index:
            # Plot the frequency polygon line
            ax.plot(
                data.index,
                data.values,
                marker=marker,
                linestyle=linestyle,
                color=line_color,
            )

            for x, y in zip(data.index, data.values):
                ax.text(x, y, f"{y}", ha="center", va="bottom", fontsize=11)

            ax.set_xticks(data.index)

        # Plot the histogram bars
        ax.bar(data.index, data.values, color=color, edgecolor=edgecolor, alpha=alpha)

        # Setting plot title, labels, and formatting
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

        # Format x-axis label
        ax.tick_params(axis="x", rotation=0)

        # Display the plot
        plt.show()

    def data_visualization(
        self,
        column_name: str,
        data: pd.DataFrame,
        figsize=(20, 8),
        title_fontsize=20,
        title_fontweight="semibold",
        axis_label_fontsize=15,
        left_chart="violin",
    ):
        """
        Function to visualize the distribution of a column using box plot and KDE plot.

        Parameters:
            column_name (str): The name of the column to visualize.
            data (pd.DataFrame): The DataFrame containing the data.
            figsize (tuple): Size of the figure (default: (20, 8)).
            title_fontsize (int): Font size of the titles (default: 20).
            title_fontweight (str): Font weight of the titles (default: "semibold").
            axis_label_fontsize (int): Font size of the axis labels (default: 15).

        Returns:
            None
        """
        sns.set_style("whitegrid")

        # Create subplots for box plot and KDE plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        column_data = data[column_name]

        # Plot distribution plot
        if left_chart == "boxplot":
            sns.boxplot(data=column_data, ax=axes[0], color="#FF617299")
        else:
            sns.violinplot(data=column_data, ax=axes[0], color="#FF617299")

        axes[0].set_title(
            f"Distribution Plot of {column_name}",
            fontsize=title_fontsize,
            fontweight=title_fontweight,
        )
        axes[0].set_xlabel(column_name, fontsize=axis_label_fontsize)

        # Plot KDE plot
        sns.kdeplot(data=column_data, ax=axes[1], fill=True, color="#ff2c43")
        axes[1].set_title(
            f"KDE Plot of {column_name}",
            fontsize=title_fontsize,
            fontweight=title_fontweight,
        )
        axes[1].set_xlabel(column_name, fontsize=axis_label_fontsize)

        # Set integer ticks on x-axis of box plot
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show()

    def data_frequency(
        self,
        column_name,
        data: pd.DataFrame,
        is_pie_chart=True,
        figsize=(20, 8),
        title_fontsize=20,
        title_fontweight="semibold",
        axis_label_fontsize=15,
        index=True,
    ):
        """
        Function to visualize the frequency distribution of a column using a pie chart and histogram with frequency polygon.

        Parameters:
            column_name (str): The name of the column to visualize.
            data (pd.DataFrame): The DataFrame containing the data.
            is_pie_chart (bool): Flag indicating whether to display a pie chart (default: True).
            figsize (tuple): Size of the figure (default: (20, 8)).
            title_fontsize (int): Font size of the titles (default: 20).
            title_fontweight (str): Font weight of the titles (default: "semibold").
            axis_label_fontsize (int): Font size of the axis labels (default: 15).

        Returns:
            None
        """
        # Create a new figure
        fig, axes = plt.subplots(1, 2 if is_pie_chart else 1, figsize=figsize)

        # Plot pie chart if enabled
        if is_pie_chart:
            value_count = data[column_name].value_counts()
            self.plot_pie_chart(
                axes[0],
                value_count,
                f"Distribution of {column_name}",
                ["#FF617299", "#ffcacf", "#FF6372", "#FF6972", "#FF6E72"],
                title_fontsize=title_fontsize,
                title_fontweight=title_fontweight,
                axis_label_fontsize=axis_label_fontsize,
            )

        # Plot histogram with frequency polygon
        self.plot_histogram_with_polygon(
            data=data[column_name],
            ax=axes[1] if is_pie_chart else axes,
            title="Histogram with Frequency Polygon",
            xlabel="Value",
            ylabel="Occurrence",
            title_fontsize=title_fontsize,
            title_fontweight=title_fontweight,
            axis_label_fontsize=axis_label_fontsize,
            index=index,
        )

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def logistic_regression_analysis(self, data, columns):
        """
        Perform logistic regression analysis on the provided data.

        Parameters:
            data (numpy.ndarray): Input data array.
            columns (list): List of column names. The last column represents the target variable.

        Returns:
            str: 2 tables summarizing the logistic regression analysis results.
        """
        # Get features (X) and target variable (y)
        X_index = [columns.index(col) for col in columns[:-1]]
        y_index = columns.index(columns[-1])
        X = data[:, X_index]
        y = data[:, y_index]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize logistic regression model
        model = LogisticRegression(max_iter=1000)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        X_train_sm = sm.add_constant(X_train)

        # Fit the logistic regression model (without optimization message)
        with np.errstate(invalid="ignore"):
            logit_model = sm.Logit(y_train, X_train_sm)
            result = logit_model.fit(disp=False)

        # Format outputs
        conf_matrix = confusion_matrix(y_test, y_pred)

        conf_matrix_table = [
            ["", "Predicted Negative", "Predicted Positive"],
            ["Actual Negative", conf_matrix[0][0], conf_matrix[0][1]],
            ["Actual Positive", conf_matrix[1][0], conf_matrix[1][1]],
        ]

        print("\nConfusion Matrix:")
        print(
            tabulate.tabulate(
                conf_matrix_table, headers="firstrow", tablefmt="rounded_grid"
            )
        )

        print()

        print(result.summary())
