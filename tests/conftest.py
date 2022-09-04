import pandas as pd
import pytest


@pytest.fixture()
def test_file():
    return "tests/test.csv"


@pytest.fixture()
def test_df(test_file):
    return pd.read_csv(test_file)


@pytest.fixture()
def reference_df():
    df = pd.DataFrame(columns=["UserId", "Fake", "count_event", "count_category"])
    df["UserId"] = ["U001", "U002", "U003"]
    df["Fake"] = [1, 0, 0]
    df["count_event"] = [24, 4, 5]
    df["count_category"] = [6, 1, 3]

    return df
