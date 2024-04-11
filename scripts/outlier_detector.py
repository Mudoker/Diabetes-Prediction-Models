import numpy as np
import tabulate


class OutlierDetector:
    """
    A class to detect outliers in a dataset.

    Methods:
        find_outliers_iqr: Find outliers using the Interquartile Range (IQR) method.
        find_outliers_mad: Find outliers using Median Absolute Deviation (MAD).
        find_outliers_normal: Find outliers using Z-score.
    """

    def __init__(self):
        # constructor
        pass

    def find_outliers_iqr(
        self,
        data,
        threshold=1.5,
        lower_quartile=25,
        upper_quartile=75,
        tablefmt="rounded_grid",
        numalign="center",
    ):
        """
        Find outliers using the Interquartile Range (IQR) method.

        Parameters:
            data (DataFrame): The data for which outliers are to be detected.
            threshold (float, optional): The multiplier for the IQR to determine the range for outliers. Defaults to 1.5.
            lower_quartile (int, optional): The lower quartile. Defaults to 25.
            upper_quartile (int, optional): The upper quartile. Defaults to 75.

        Returns:
            numpy.ndarray: Array indicating outliers.
        """
        # Calculate the first and third quartiles
        quartile_1, quartile_3 = np.percentile(data, [lower_quartile, upper_quartile])

        # Calculate the interquartile range (IQR)
        iqr = quartile_3 - quartile_1

        # Calculate lower and upper limits
        upper_bound = quartile_3 + iqr * threshold
        lower_bound = quartile_1 - iqr * threshold

        # Identify outliers
        outliers = [
            value for value in data if value > upper_bound or value < lower_bound
        ]

        outliers_text = ""
        if outliers:
            # Get unique outliers
            unique_outliers = set(outliers)
            outliers_text += ", ".join(
                str(outlier) for outlier in list(unique_outliers)[:5]
            )
            if len(unique_outliers) > 5:
                outliers_text += ", ..."
        else:
            outliers_text += "None"

        payload = {
            "Unique Outliers": outliers_text,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Threshold": threshold,
            "Total Outliers": len(outliers),
        }

        table = tabulate.tabulate(
            payload.items(),
            headers=["Key", "Value"],
            tablefmt=tablefmt,
            numalign=numalign,
        )

        return {
            "table": table,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "total_outliers": len(outliers),
        }

    def find_outliers_mad(
        self, data, threshold=3.5, tablefmt="rounded_grid", numalign="center"
    ):
        """
        Find outliers using Median Absolute Deviation (MAD).

        Parameters:
            data (array): The data for which outliers are to be detected.
            threshold (float, optional): The threshold value to determine outliers. Defaults to 3.5.
            tablefmt (str, optional): Format for the output table. Defaults to "rounded_grid".
            numalign (str, optional): Alignment for numeric columns. Defaults to "center".

        Returns:
            dict: Dictionary containing outlier detection results.
        """
        # Calculate median
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        # Calculate lower and upper limits
        lower_limit = median - threshold * mad
        upper_limit = median + threshold * mad

        # Identify outliers
        outliers = (data < lower_limit) | (data > upper_limit)
        outliers_indices = np.where(outliers)[0]

        # Format outliers text
        outliers_text = ", ".join(str(data[i]) for i in outliers_indices[:5])
        if len(outliers_indices) > 5:
            outliers_text += ", ..."

        payload = {
            "Outliers": outliers_text,
            "Lower Bound": lower_limit,
            "Upper Bound": upper_limit,
            "Threshold": threshold,
            "Total Outliers": np.sum(outliers),
        }

        table = tabulate.tabulate(
            payload.items(),
            headers=["Key", "Value"],
            tablefmt=tablefmt,
            numalign=numalign,
        )

        return {
            "table": table,
            "lower_bound": lower_limit,
            "upper_bound": upper_limit,
            "total_outliers": np.sum(outliers),
        }

    def find_outliers_normal(data, threshold=3):
        """
        Find outliers using three times the mean.

        Parameters:
            data (array-like): The data for which outliers are to be detected.
            threshold (float, optional): The threshold value to determine outliers. Defaults to 3.

        Returns:
            numpy.ndarray: Boolean array indicating outliers.
        """

        # Calculate mean
        mean = np.mean(data)

        # Calculate threshold
        threshold = threshold * mean

        # Calculate lower and upper limits
        lower_limit = mean - threshold
        upper_limit = mean + threshold

        # Identify outliers
        outliers = (data < lower_limit) | (data > upper_limit)

        return outliers
