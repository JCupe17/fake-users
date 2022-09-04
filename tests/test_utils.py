import pandas as pd
import pytest

import src.utils as utils


def test_load_input_file(test_file):
    # Load a correct file
    df = utils.load_input_file(test_file)
    assert type(df) == pd.DataFrame

    # Load a file that does not end in .csv
    with pytest.raises(AssertionError):
        utils.load_input_file("test_file.txt")
    # Load a file that does not exist
    with pytest.raises(OSError):
        utils.load_input_file("this_file_doesnt_exist.csv")


def test_check_input_columns(test_df):
    reference_columns = ["UserId", "Event", "Category"]
    assert utils.check_input_columns(test_df, reference_columns)

    reference_columns = ["UserId", "Event", "Category", "Unknown_column"]
    assert not utils.check_input_columns(test_df, reference_columns)


def test_feature_engineering(test_df, reference_df):
    # Test the feature engineering for training
    df = utils.feature_engineering(test_df, training=True)
    columns_training = ["UserId", "Fake", "count_event", "count_category"]
    assert all(_ in df.columns for _ in columns_training)
    assert reference_df.equals(df)

    # Test the feature engineering for evaluation / inference
    df = utils.feature_engineering(test_df, training=False)
    assert "Fake" not in df.columns
    assert reference_df.drop(columns=["Fake"]).equals(df)
