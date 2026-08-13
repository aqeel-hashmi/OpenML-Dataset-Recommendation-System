import json
import openml
from tqdm import tqdm

from . import config

catalog = openml.datasets.list_datasets(output_format="dataframe")

all_metadata = []

for did in tqdm(catalog["did"]):
    try:
        ds = openml.datasets.get_dataset(
            did,
            download_data=False,
        )

        metadata = {}

        for key, value in vars(ds).items():
            try:
                json.dumps(value)
                metadata[key] = value
            except TypeError:
                metadata[key] = str(value)

        all_metadata.append(metadata)

    except Exception as e:
        print(f"Failed {did}: {e}")

with open(config.RAW_DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, indent=2, ensure_ascii=False)