# -*- coding: utf-8 -*-
"""
01_data_ingestion.py
Load raw descriptors and target data.
"""

import pandas as pd
from typing import Tuple

def load_X_data(file_path: str) -> pd.DataFrame:
    """
    Load descriptors/features CSV file
    Args:
        file_path (str): Path to descriptors CSV
    Returns:
        pd.DataFrame: Raw descriptors
    """
    try:
        df = pd.read_csv(file_path, sep=',')
        print(f"Loaded X data: shape {df.shape}, type {type(df)}")
        return df
    except Exception as e:
        print(f"Error loading X data: {e}")
        raise

def load_y_data(file_path: str) -> pd.DataFrame:
    """
    Load target CSV file
    Args:
        file_path (str): Path to target CSV
    Returns:
        pd.DataFrame: Target values
    """
    try:
        df = pd.read_csv(file_path, sep=',')
        print(f"Loaded y data: shape {df.shape}, type {type(df)}")
        return df
    except Exception as e:
        print(f"Error loading y data: {e}")
        raise

def get_train_test_indices(df: pd.DataFrame, test_indices: list = [4, 9, 14]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by specific indices
    Args:
        df (pd.DataFrame): Full data
        test_indices (list): List of row indices to use as test
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_df, test_df
    """
    is_test = df.index.isin(test_indices)
    is_train = ~is_test
    return df[is_train], df[is_test]

if __name__ == "__main__":
    X_file = "Dataset/descriptors.csv"
    y_file = "Dataset/components_fractions.csv"

    X = load_X_data(X_file)
    y = load_y_data(y_file)

    X_train, X_test = get_train_test_indices(X)
    y_train, y_test = get_train_test_indices(y)

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
