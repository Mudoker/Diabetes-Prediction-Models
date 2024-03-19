import numpy as np


class OutlierDetector:
    def __init__(self):
        # constructor
        pass

    def find_outliers_iqr(data, threshold=1.5):
        """
        Find outliers using the Interquartile Range (IQR) method.

        Parameters:
            data (array): The data for which outliers are to be detected.
            threshold (float, optional): The multiplier for the IQR to determine the range for outliers. Defaults to 1.5.

        Returns:
            numpy.ndarray: Array indicating outliers.
        """
        # Calculate the first and third quartiles
        quartile_1, quartile_3 = np.percentile(data, [25, 75])

        # Calculate the interquartile range (IQR)
        iqr = quartile_3 - quartile_1

        # Calculate lower and upper limits
        upper_bound = quartile_3 + iqr * threshold
        lower_bound = quartile_1 - iqr * threshold

        # Identify outliers
        outliers = [
            value for value in data if value > upper_bound or value < lower_bound
        ]

        return outliers

    def find_outliers_mad(data, threshold=3.5):
        """
        Find outliers using Median Absolute Deviation (MAD).

        Parameters:
            data (array): The data for which outliers are to be detected.
            threshold (float, optional): The threshold value to determine outliers. Defaults to 3.5.

        Returns:
            numpy.ndarray: array indicating outliers.
        """

        # Calculate median
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        # Calculate lower and upper limits
        lower_limit = median - threshold * mad
        upper_limit = median + threshold * mad

        # Identify outliers
        outliers = (data < lower_limit) | (data > upper_limit)

        return outliers

    def find_outliers_normal(data, threshold=3):
        """
        Find outliers using three times the mean.

        Parameters:
            data (array-like): The data for which outliers are to be detected.

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
