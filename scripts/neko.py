import pandas as pd
import tabulate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import seaborn as sns

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
        axes,
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
    ):
        """
        Plot a histogram with a frequency polygon on the same plot.

        Parameters:
            data (array-like): Input data to plot.
            axes (matplotlib.axes.Axes): The axes on which to plot the histogram and polygon.
            bins (int or sequence, optional): Specification of histogram bins. Default is 10.
            edgecolor (color, optional): Color of the histogram edges. Default is '#ff6172'.
            alpha (float, optional): Transparency of the histogram bars. Default is 0.7.
            color (color, optional): Color of the histogram bars. Default is '#ffcacf'.
            marker (str, optional): Marker style for the frequency polygon. Default is 'o'.
            linestyle (str, optional): Line style for the frequency polygon. Default is '-'.
            line_color (color, optional): Color of the frequency polygon line. Default is 'r'.
            title (str, optional): Title of the plot. Default is an empty string.
            xlabel (str, optional): Label for the x-axis. Default is an empty string.
            ylabel (str, optional): Label for the y-axis. Default is an empty string.
            title_fontsize (int, optional): Font size for the title. Default is None.
            title_fontweight (str or int, optional): Font weight for the title. Default is None.
            axis_label_fontsize (int, optional): Font size for axis labels. Default is None.
            left_chart (str, optional): Type of left chart (default: "boxplot").
        Returns:
            None
        """

        # Plot histogram
        axes.hist(
            data,
            bins=bins,
            edgecolor=edgecolor,
            alpha=alpha,
            label="Histogram",
            color=color,
        )

        # Plot frequency polygon
        frequencies, bin_edges = np.histogram(data, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        axes.plot(
            bin_centers,
            frequencies,
            marker=marker,
            linestyle=linestyle,
            color=line_color,
            label="Frequency Polygon",
        )

        # Set title and labels
        if title:
            axes.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        if xlabel:
            axes.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        if ylabel:
            axes.set_ylabel(ylabel, fontsize=axis_label_fontsize)

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
            axes=axes[1] if is_pie_chart else axes,
            bins=20,
            title="Histogram with Frequency Polygon",
            xlabel="Value",
            ylabel="Frequency",
            title_fontsize=title_fontsize,
            title_fontweight=title_fontweight,
            axis_label_fontsize=axis_label_fontsize,
        )

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
