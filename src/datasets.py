import dask.dataframe as dd
from torch.utils.data import DataLoader, Dataset

from src.toolkit.labeled import LabeledDag
from src.train_utils import collate_graph_batch
import tqdm

class LabeledDagDatasetInMemory(Dataset):
    def __init__(self, dataset_dir: str, toolkit: LabeledDag):
        self.df = dd.read_parquet(dataset_dir, engine="pyarrow", dtype_backend="pyarrow")
        self.df = self.df.compute()

        self.toolkit = toolkit

        self.graphs = [
            self.toolkit.from_dict_to_graph(dict(record))
            for i, record
            in tqdm.tqdm(self.df.iterrows(), total=len(self.df), desc=f"Load graphs in memory")
        ]

        del self.df

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


if __name__ == '__main__':
    toolkit = LabeledDag(num_vertices=8, label_cardinality=8)

    dataset = LabeledDagDatasetInMemory(
        dataset_dir="/experiments/00_bn_asia_200k/data/train",
        toolkit=toolkit,
    )

    dl = DataLoader(
        dataset=dataset,
        batch_size=8,
        collate_fn=collate_graph_batch
    )
