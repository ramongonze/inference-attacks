import os
import pyreadr
import numpy as np
import pandas as pd

class Data:
    """
    A class for handling datasets from file formats CSV, RData and SAS7BDAT.

    Parameters:
        - file_name (str, optional): The path to the dataset file.
        - dataframe (pandas.DataFrame, optional): A pandas DataFrame containing the dataset.
        - sep_csv (str, optional): The delimiter used in CSV files.
        - na_values (int, optional): The value to replace NaN values with in the dataset, default is -1.

    Attributes:
        - dataframe (pd.DataFrame): The pandas DataFrame containing the dataset.
        - num_rows (int): The number of rows in the dataset.
        - num_cols (int): The number of columns in the dataset.
        - columns (list): The list of column names in the dataset.
    """

    def __init__(self, file_name=None, dataframe=None, sep_csv=None, columns=None, encoding="utf-8", na_values=-1):
        """
        Initializes a Dataset object.

        Args:
            file_name (str, optional): The path to the dataset file.
            dataframe (pandas.DataFrame, optional): A pandas DataFrame containing the dataset.
            sep_csv (str, optional): The delimiter used in CSV files.
            columns (list, optional): Columns to be read from a file (valid only for csv files). When not given, it will read all columns. Default is None.
            encoding(str, optional): Encoding to use for UTF when reading/writing (valid only for csv files). Default is "utf-8".
            na_values (int, optional): The value to replace NaN values with in the dataset, default is -1.
        """
        if dataframe is not None:
            if isinstance(dataframe, pd.DataFrame):
                self.dataframe = dataframe
            else:
                raise TypeError("dataframe must be a pandas.DataFrame object")
        elif file_name is not None:
            file_type = self._file_extension(file_name)
            if file_type == ".csv":
                if sep_csv is None:
                    raise NameError("sep_csv must be provided for CSV files")
                self.dataframe = pd.read_csv(file_name, sep=sep_csv, usecols=columns, encoding=encoding)
            elif file_type == ".rdata":
                rdata = pyreadr.read_r(file_name)
                data = next(iter(rdata))
                self.dataframe = rdata[data]
            elif file_type == ".sas7bdat":
                self.dataframe = pd.read_sas(file_name)
            else:
                raise TypeError("The only supported files are csv, rdata and sas7bdat")
        else:
            raise TypeError("Either file_name or dataframe must be given")
        
        self.dataframe.replace(np.nan, na_values, inplace=True)
        self.num_rows = self.dataframe.shape[0]
        self.num_cols = self.dataframe.shape[1]
        self.columns = self.dataframe.columns.to_list()

    def _file_extension(self, file_name):
        """
        Infer the file extension from a given file path.

        Parameters:
            file_name (str): The path to the file.

        Returns:
            str: The file extension in lowercase.
        """
        _, extension = os.path.splitext(file_name)
        return extension.lower()
