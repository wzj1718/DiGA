import os

import pandas as pd
from datasets import load_dataset


DEFAULT_WILDGUARDMIX_TEST_PATH = os.path.join(
    os.environ.get("DIFA_DATA_ROOT", "./data"), "wildguardmix", "test", "wildguard_test.parquet"
)


def load_wildguardmix_test_dataframe() -> pd.DataFrame:
    """Load WildGuardTest from the local ModelScope download when available."""
    local_path = os.environ.get("WILDGUARDMIX_TEST_PATH", DEFAULT_WILDGUARDMIX_TEST_PATH)
    if local_path and os.path.exists(local_path):
        return pd.read_parquet(local_path)

    return load_dataset("allenai/wildguardmix", "wildguardtest")["test"].to_pandas()
