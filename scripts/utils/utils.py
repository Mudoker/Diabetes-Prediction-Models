import pandas as pd


class Utils:
    def __init__(self):
        # constructor logic
        pass

    def to_numeric(data):
        # Replace errors with 'NA'
        data = data.apply(pd.to_numeric, errors="coerce")
        return data
