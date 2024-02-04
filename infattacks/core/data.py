import pyreadr
import numpy as np
import pandas as pd

class Data:
    """
    A class for handling datasets from different file formats such as CSV and RData.

    Parameters:
        - file_path (str): The path to the dataset file.
        - file_type (str, optional): The type of the dataset file, default is "csv".
        - sep (str, optional): The delimiter used in CSV files. Required if file_type is "csv".
        - r_dataframe_name (str, optional): The name of the R DataFrame if file_type is "RData".
        - na_values (int, optional): The value to replace NaN values with in the dataset, default is -1.

    Attributes:
        - dataframe (pd.DataFrame): The pandas DataFrame containing the dataset.
        - num_rows (int): The number of rows in the dataset.
        - num_cols (int): The number of columns in the dataset.
        - columns (list): The list of column names in the dataset.
    """

    def __init__(self, dataframe=None, file_path=None, file_type="csv", sep=None, r_dataframe_name=None, na_values=-1):
        """
        Initializes a Dataset object.

        Args:
            dataframe (pandas.DataFrame, optional): A pandas DataFrame containing the dataset.
            file_path (str, optional): The path to the dataset file.
            file_type (str, optional): The type of the dataset file, default is "csv".
            sep (str, optional): The delimiter used in CSV files. Required if file_type is "csv".
            r_dataframe_name (str, optional): The name of the R DataFrame if file_type is "RData".
            na_values (int, optional): The value to replace NaN values with in the dataset, default is -1.

        Raises:
            NameError: If required parameters are not declared.
        """
        if dataframe is not None:
            if isinstance(dataframe, pd.DataFrame):
                self.dataframe = dataframe
            else:
                raise TypeError("dataframe must be a pandas.DataFrame object")
        elif file_type == "csv":
            if sep is None:
                raise NameError("sep must be provided for CSV files.")
    
            self.dataframe = pd.read_csv(file_path, sep=sep)
        elif file_type == "RData":
            if r_dataframe_name is None:
                raise NameError("r_dataframe_name must be provided for RData files.")
            
            self.dataframe = pyreadr.read_r(file_path)[r_dataframe_name]
        
        self.dataframe.replace(np.nan, na_values, inplace=True)
        self.num_rows = self.dataframe.shape[0]
        self.num_cols = self.dataframe.shape[1]
        self.columns = self.dataframe.columns.to_list()
