import pandas as pd
import logging


def load_input_file(filename: str) -> pd.DataFrame:
    """Loads a csv file into a pandas DataFrame."""
    try:
        assert filename.endswith(".csv")
        df = pd.read_csv(filename)
    except (AssertionError, OSError) as e:
        logging.error(f"{type(e)}: {e}")
        raise

    return df


def check_input_columns(df: pd.DataFrame, reference_columns: list) -> bool:
    return all(_ in df.columns for _ in reference_columns)


def feature_engineering(df: pd.DataFrame, training: bool) -> pd.DataFrame:
    """Groups the events for a user and counts the number of events and unique categories for each user."""
    df_grouped = pd.DataFrame(columns=["UserId", "count_event", "count_category"])
    columns_to_group = ["UserId"]
    if training:
        columns_to_group.append("Fake")
    if not df.empty:
        df_grouped = df.groupby(columns_to_group).agg(count_event=("Event", "count"),
                                                      count_category=("Category", "nunique"))
        df_grouped = df_grouped.reset_index()
    return df_grouped
