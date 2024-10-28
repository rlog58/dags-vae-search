from typing import Iterable

import dask.dataframe as dd
import igraph as ig
import torch
from dask_ml.model_selection import train_test_split

from src.toolkit.labeled import LabeledDag


def load_model_state(
        model,
        state_name
):
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return


def collate_graph_batch(data: Iterable[ig.Graph]):
    return [g.copy() for g in data]


def split_dataset(
        df: dd.DataFrame,
        output: str,
        train_postfix='_train',
        test_postfix='_test',
        val_postfix='_val',
        test_ratio=0.1,
        val_ratio=0.0,
        seed=42,
):
    df = dd.read_parquet(dataset_path, engine="pyarrow", dtype_backend="pyarrow")

    if test_ratio >= 1 or test_ratio <= 0:
        raise ValueError('test_ratio must be > 0 and < 1')

    if val_ratio >= 1 or test_ratio < 0:
        raise ValueError("'val_ratio' must be > 0 and < 1")

    if test_ratio + val_ratio >= 1:
        raise ValueError('test_ratio and val_ratio must be <= 1')

    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed, shuffle=True)

    if val_ratio > 0:
        train_df.to_parquet(dataset_path + train_postfix)
        test_df.to_parquet(dataset_path + test_postfix)

        print(f"Train size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")
    else:
        rel_val_ratio = val_ratio / (1 - test_ratio)
        train_df, val_df = train_test_split(train_df, test_size=rel_val_ratio, random_state=seed, shuffle=True)

        train_df.to_parquet(dataset_path + train_postfix)
        test_df.to_parquet(dataset_path + test_postfix)
        val_df.to_parquet(dataset_path + val_postfix)

        print(f"Train size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")
        print(f"Validation size: {len(val_df)}")


if __name__ == '__main__':
    graphV = LabeledDag(num_vertices=8, label_cardinality=8)

    dataset_path = "../data/final_structures12.parquet"

    df = dd.read_parquet(dataset_path, engine="pyarrow", dtype_backend="pyarrow")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

    train_df.to_parquet(dataset_path + "_train", engine="pyarrow", schema=graphV.pyarrow_schema)
    test_df.to_parquet(dataset_path + "_test", engine="pyarrow", schema=graphV.pyarrow_schema)
