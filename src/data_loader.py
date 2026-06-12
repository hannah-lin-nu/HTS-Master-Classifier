import pandas as pd
from datasets import load_from_disk


def load_cross_dataset(path="data/hts_dataset"):
    """Load the CROSS HTS dataset from local storage.

    Args:
        path (str): Path to the saved Hugging Face dataset.

    Returns:
        pd.DataFrame: Training split as a pandas DataFrame.
    """

    # Load dataset from disk
    dataset = load_from_disk(path)

    # Convert training split to dataframe
    return dataset["train"].to_pandas()


def load_hts_schedule(filepath):
    """Load the U.S. Harmonized Tariff Schedule (HTS) CSV.

    Args:
        filepath (str): Path to the HTS schedule CSV file.

    Returns:
        pd.DataFrame: HTS schedule data.
    """

    # Read HTS schedule
    return pd.read_csv(
        filepath,
        low_memory=False
    )
