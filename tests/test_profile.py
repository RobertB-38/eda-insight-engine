import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import profile_dataframe


def make_df():
    return pd.DataFrame({
        'age':    [25, 30, 35, 40, None],
        'salary': [50000, 60000, 70000, 80000, 90000],
        'city':   ['Dublin', 'London', 'Dublin', 'Paris', 'London']
    })


def test_profile_shape():
    df = make_df()
    result = profile_dataframe(df)
    assert result['shape']['rows'] == 5
    assert result['shape']['cols'] == 3


def test_profile_missing_pct():
    df = make_df()
    result = profile_dataframe(df)
    # 1 null out of 15 values = 6.67%
    assert result['total_missing_pct'] > 0


def test_profile_columns_exist():
    df = make_df()
    result = profile_dataframe(df)
    assert 'age' in result['columns']
    assert 'salary' in result['columns']
    assert 'city' in result['columns']


def test_numeric_column_has_stats():
    df = make_df()
    result = profile_dataframe(df)
    salary = result['columns']['salary']
    assert 'mean' in salary
    assert 'std' in salary
    assert salary['mean'] == 70000.0


def test_categorical_column_has_top_values():
    df = make_df()
    result = profile_dataframe(df)
    city = result['columns']['city']
    assert 'top_values' in city
    assert 'Dublin' in city['top_values']


def test_unreliable_flag():
    # column with >30% nulls should be flagged unreliable
    df = pd.DataFrame({
        'a': [1, None, None, None, None],  # 80% null
        'b': [1, 2, 3, 4, 5]
    })
    result = profile_dataframe(df)
    assert result['columns']['a']['unreliable'] is True
    assert result['columns']['b']['unreliable'] is False


def test_duplicates_detected():
    df = pd.DataFrame({
        'x': [1, 1, 2],
        'y': [1, 1, 2]
    })
    result = profile_dataframe(df)
    assert result['duplicates'] == 1
