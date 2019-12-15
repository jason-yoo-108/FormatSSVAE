import torch 
from torch.utils.data import Dataset, DataLoader
from FormatSSVAE.util.ds_utils import remove_rows_in_col, convert_row_to_lower
from FormatSSVAE.const import ALL_LETTERS
import pandas as pd

class NameDataset(Dataset):
    def __init__(self, csv_path, col_name, max_string_len = 18, format_col_name = None):
        """
        Args:
            csv_file (string): Path to the csv file WITHOUT labels
            col_name (string): The column name corresponding to the people names that'll be standardized
        """
        self.max_string_len = max_string_len
        df = pd.read_csv(csv_path)
        df = self._clean_dataframe(df, col_name)
        self.data_frame = df[col_name]
        self.format_col = df[format_col_name] if format_col_name is not None else None

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        return self.data_frame.iloc[index]

    def add_csv(self, csv_path, col_name):
        """
        Args:
            csv_path (string): Path to csv file WITHOUT labels to be added to self.data_frame
            col_name (string): Name of column in csv with name
        """
        df = pd.read_csv(csv_path)
        df = self._clean_dataframe(df, col_name)
        self.data_frame = self.data_frame.append(df[col_name]).drop_duplicates()

    def _clean_dataframe(self, df, col_name):
        convert_row_to_lower(df, col_name)
        return remove_rows_in_col(df, col_name, list(ALL_LETTERS), max_string_len=self.max_string_len)
