import logging
import random

from src.encoders.utils import create_encoder_dataset
from src.toolkit.labeled import LabeledDag

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random.seed(42)

    toolkit = LabeledDag(
        num_vertices=12,
        label_cardinality=12,
        validation=True,
    )

    create_encoder_dataset(
        toolkit,
        "encoder_dataset",
        200,
        20,
        0.4,
        overwrite=True,
        npartitions=1,
    )