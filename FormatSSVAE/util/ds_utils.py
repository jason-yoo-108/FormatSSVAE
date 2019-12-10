from collections import OrderedDict
import json
import math
import pandas as pd

def load_json(jsonpath: str) -> dict:
    with open(jsonpath) as jsonfile:
        return json.load(jsonfile, object_pairs_hook=OrderedDict)

def remove_rows_in_col(df, column_name: str, accepted_chars: list, max_string_len: int = 18):
    """
    all rows removed from a column that do not
    contain accepted characters
    data_frame: Panda dataframe to be filtered
    column_name: The column name in df to be filtered through
    accepted_chars: List of characters that are acceptable in df row
    """
    return df[df[column_name].apply(lambda x: set(x).issubset(accepted_chars) and len(x) < max_string_len if(isinstance(x,str)) else True)]

def convert_row_to_lower(data_frame, column_name: str):
    """
    Converts all characters in data_frame column to lowercase
    data_frame: Panda dataframe to be filtered
    column_name: The colmun name of the df to be filtered through
    """
    data_frame[column_name] = data_frame[column_name].str.lower()