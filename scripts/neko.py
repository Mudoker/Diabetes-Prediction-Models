import warnings
import pandas as pd
from imblearn.over_sampling import (
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN,
    RandomOverSampler,
)  # noqa
from imblearn.combine import SMOTEENN  # noqa
from imblearn.under_sampling import RandomUnderSampler, NearMiss  # noqa
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tabulate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

NEKO_ART = r"""
    /\_____/\
   /  x   o  \
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
            "Data Types": ", ".join(data.dtypes.unique().astype(str).tolist()),
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
            "Data Type": data[feature].dtype,
            "Total / Missing Values": f"{data[feature].count()} / {data[feature].isnull().sum()}",
            "Range": f"{data[feature].min()} - {data[feature].max()}",
            "Percentiles (25-75)": f"{data[feature].quantile(0.25)} - {data[feature].quantile(0.75)}",
            "Median": data[feature].median(),
            "Mean": data[feature].mean(),
            "Standard Deviation": data[feature].std(),
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
        colors,
        axis_label_fontsize=15,
    ):
        """
        Plot a pie chart on the given axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the pie chart.
            count (pandas.Series): The data to be plotted.
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

    def plot_histogram_with_polygon(
        self,
        data,
        ax,
        edgecolor="#ff6172",
        alpha=0.7,
        color="#ffcacf",
        marker="o",
        linestyle="-",
        line_color="r",
        ylabel="",
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
                ax.text(x, y, f"{y}", ha="center", va="bottom", fontsize=9)

            ax.set_xticks(data.index)

        # Plot the histogram bars
        ax.bar(data.index, data.values, color=color, edgecolor=edgecolor, alpha=alpha)

        # Setting plot title, labels, and formatting
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

        # Format x-axis label
        ax.tick_params(axis="x", rotation=0)

        # Display the plot
        plt.show()

    def data_frequency(
        self,
        column_name,
        data: pd.DataFrame,
        is_pie_chart=True,
        is_violin_plot=True,
        is_box_plot=False,
        is_kde=False,
        figsize=(20, 8),
        title_fontsize=16,
        title_fontweight="semibold",
        axis_label_fontsize=12,
        index=True,
    ):
        """
        Function to visualize the frequency distribution of a column using a pie chart, violin plot, and histogram with frequency polygon.

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
        fig, axes = plt.subplots(
            1, 1 + is_pie_chart + is_violin_plot + is_box_plot, figsize=figsize
        )
        fig.suptitle(
            f"Distribution of {column_name}",
            fontsize=title_fontsize,
            fontweight=title_fontweight,
        )

        pos = 0
        # Plot pie chart if enabled
        if is_pie_chart:
            value_count = data[column_name].value_counts()
            self.plot_pie_chart(
                axes[0],
                value_count,
                ["#FF617299", "#ffcacf", "#FF6372", "#FF6972", "#FF6E72"],
                axis_label_fontsize=axis_label_fontsize,
            )
            pos += 1

        if is_violin_plot:
            # Plot violin plot
            sns.violinplot(y=data[column_name], ax=axes[pos], color="#FF6172")
            pos += 1

        if is_box_plot:
            # Plot box plot
            sns.boxplot(y=data[column_name], ax=axes[pos], color="#FF6172")
            pos += 1

        # Plot histogram with frequency polygon
        if is_kde:
            sns.kdeplot(data[column_name], ax=axes[pos], color="#FF6172", fill=True)
        else:
            # Plot histogram with frequency polygon
            self.plot_histogram_with_polygon(
                data=data[column_name],
                ax=axes[pos],
                index=index,
            )

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def logistic_regression_analysis(self, data, features, target):
        """
        Perform logistic regression analysis on the provided data.

        Parameters:
            data (DataFrame): Input data DataFrame.
            features (List): List of feature columns.
            target (String): Target column.

        Returns:
            str: Two tables summarizing the logistic regression analysis results.
        """
        # Get features (X) and target variable (y)
        X = data[features]
        y = data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Initialize and train logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Format outputs
        conf_matrix = confusion_matrix(y_test, y_pred)

        conf_matrix_table = [
            ["", "Predicted Negative", "Predicted Positive"],
            ["Actual Negative", conf_matrix[0][0], conf_matrix[0][1]],
            ["Actual Positive", conf_matrix[1][0], conf_matrix[1][1]],
        ]

        output = "\nConfusion Matrix:\n"
        output += tabulate.tabulate(
            conf_matrix_table, headers="firstrow", tablefmt="grid"
        )
        output += "\n\n"

        # Print summary
        output += "\n\nAccuracy: {:.2f}%".format(model.score(X_test, y_test) * 100)

        return output

    def resample_data(self, data, target, method="smote", random_state=42):
        """
        Perform over-sampling using SMOTE on the provided data.

        Parameters:
            data (DataFrame): Input data DataFrame.
            target (String): Target column.
            method (String): The over-sampling method to use. Default is "smote".
            random_state (int): Random state for reproducibility. Default is 42.
        Returns:
            DataFrame: The over-sampled data.
        """
        # Error handling
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if target not in data.columns:
            raise ValueError("The target column does not exist in the DataFrame.")

        # Separate features and target (avoid unnecessary copy)
        X = data.drop(target, axis=1)
        y = data[target]

        # Over-sampling with appropriate sampler based on method
        sampler = {
            "smote": SMOTE(random_state=random_state),
            "random_over": RandomOverSampler(random_state=random_state),
            "borderline_smote": BorderlineSMOTE(random_state=random_state),
            "svm_smote": SVMSMOTE(random_state=random_state),
            "adasyn": ADASYN(random_state=random_state),
            "random_under": RandomUnderSampler(random_state=random_state),
            "nearmiss": NearMiss(),
            "smote_enn": SMOTEENN(random_state=random_state),
        }.get(method)

        if sampler is None:
            raise ValueError(
                "Invalid resampling method. Choose from: 'smote', 'random_over', 'borderline_smote', 'svm_smote', 'adasyn', 'random_under', 'nearmiss', 'smote_enn'."
            )

        # Perform over-sampling on separate arrays
        X_over, y_over = sampler.fit_resample(X, y)

        # Combine features and target into a new DataFrame (clear separation)
        data_over = pd.concat([X_over, pd.DataFrame(y_over, columns=[target])], axis=1)

        return data_over

    def scale_feature(self, data, method="norm"):
        """
        Scale the features in the provided data using the specified method.

        Parameters:
            data (DataFrame): Input data DataFrame.
            method (String): The scaling method to use. Default is "norm".

        Returns:
            DataFrame: The scaled data.
        """
        # Error handling
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Scale features using appropriate scaler based on method
        scaler = {
            "norm": MinMaxScaler(),
            "standard": StandardScaler(),
        }.get(method)

        if scaler is None:
            raise ValueError("Invalid scaling method. Choose 'norm' or 'standard'.")

        if method == "standard" and (data.skew().abs() > 1).any():
            print(
                ">>> Warning: StandardScaler may not work well with highly skewed data. Consider using MinMaxScaler."
            )

        # Apply scaling to all columns
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        return data_scaled

    def log_transform(self, data, columns):
        """
        Log-transform the features in the provided data.

        Parameters:
            data (DataFrame): Input data DataFrame.
            columns (list): List of columns to be log-transformed.

        Returns:
            DataFrame: The log-transformed data.
        """
        # Error handling
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        for column in columns:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame.")

        # Log-transform specified columns
        data_log = data.copy()
        data_log[columns] = np.log(data_log[columns])

        return data_log

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """
        Evaluate the performance of a model using the provided test data.

        Parameters:
            model (object): The trained model to evaluate.
            X_test (DataFrame): The test features.
            y_test (Series): The test target variable.

        Returns:
            str: A classification report summarizing the model performance.
        """
        # Make predictions
        model.fit(X_train, y_train)

        # Step 2: Predictions on training and testing data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Step 3: Generate classification reports
        train_report = classification_report(y_train, y_train_pred)
        test_report = classification_report(y_test, y_test_pred)

        print("Classification Report for Training Data:\n", train_report)
        print("Classification Report for Testing Data:\n", test_report)

        # Step 4: Calculate accuracy scores
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("Accuracy on Training Data:", train_accuracy)
        print("Accuracy on Testing Data:", test_accuracy)

        # Step 5: Compare performance metrics
        if train_accuracy > test_accuracy:
            print("\nThe model is overfitting to the training data.")
        elif train_accuracy < test_accuracy:
            print("\nThe model is underfitting to the training data.")
        else:
            print("\nThe model is performing well on both training and testing data.")

    def post_pruning(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        classifier="descision_tree",
        plot=True,
    ):
        """
        Find the best ccp_alpha value using cost complexity pruning path and plot the train/test accuracy.

        Parameters:
            model (object): The Decision Tree model.
            X_train (DataFrame): The training features.
            y_train (Series): The training target variable.
            X_test (DataFrame): The test features.
            y_test (Series): The test target variable.
            classifier (str, optional): Classifier type. Defaults to "decision_tree".
            plot (bool): Flag to plot the train/test accuracy vs. ccp_alpha. Default is True.

        Returns:
            float: The optimal ccp_alpha value.
        """

        # Error handling
        if classifier not in ["decision_tree", "random_forest"]:
            raise ValueError(
                "Invalid model. Choose 'DecisionTreeClassifier' or 'RandomForestClassifier'."
            )

        # Get CCP alpha values and impurities
        clf = model
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        # Store training and testing scores for each alpha
        train_scores = []
        test_scores = []

        for ccp_alpha in ccp_alphas:
            # Create classifier instance with current CCP alpha
            if classifier == "decision_tree":
                clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
            elif classifier == "random_forest":
                clf = RandomForestClassifier(random_state=42, ccp_alpha=ccp_alpha)

            # Train the model and make predictions
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            test_pred = clf.predict(X_test)

            # Calculate accuracy scores
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)

            # Store scores
            train_scores.append(train_accuracy)
            test_scores.append(test_accuracy)

        # Plot train/test accuracy vs CCP alpha (optional)
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(
                ccp_alphas,
                train_scores,
                marker="o",
                label="Train",
                drawstyle="steps-post",
            )
            plt.plot(
                ccp_alphas,
                test_scores,
                marker="o",
                label="Test",
                drawstyle="steps-post",
            )
            plt.xlabel("ccp_alphas")
            plt.ylabel("Accuracy")
            plt.title("Train/Test Accuracy vs. ccp_alphas")
            plt.legend()
            plt.show()

        # Find the optimal ccp_alpha value that maximizes the test accuracy
        optimal_alpha = ccp_alphas[test_scores.index(max(test_scores))]

        return optimal_alpha
