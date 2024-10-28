from typing import Dict, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.toolkit.labeled import LabeledDag


def bn_from_txt(filename):
    rows_list = []

    with open(filename, 'r') as f:
        for i, row in enumerate(f):
            row_dict = {}

            row, y = eval(row)
            n = len(row)

            for j in range(n):
                row_dict['l{}'.format(j)] = row[j][0]

            for j in range(n):
                row_dict['e{}'.format(j)] = "".join(str(x) for x in row[j][1:]) if len(row[j][1:]) > 0 else ""

            rows_list.append(row_dict)

    df = pd.DataFrame(rows_list)

    return df


def bn_parquet_iterator(parquet_file: pq.ParquetFile) -> Iterator[Dict]:
    for i in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(i)
        for batch in row_group.to_batches():
            label_columns = [col_name for col_name in batch.column_names if col_name.startswith("l")]
            edges_columns = [col_name for col_name in batch.column_names if col_name.startswith("e")]

            labels_rows = [batch[col_name] for col_name in label_columns]
            edges_rows = [batch[col_name] for col_name in edges_columns]

            for labels_row, edges_row, metric_value in zip(zip(*labels_rows), zip(*edges_rows), batch['metric']):
                record = {
                    "labels": [elem.as_py() for elem in labels_row],
                    "edges": [elem.as_py() for elem in edges_row],
                    "metric": metric_value.as_py()
                }
                yield record


if __name__ == '__main__':
    graphV = LabeledDag(num_vertices=12, label_cardinality=12)

    df = bn_from_txt('../data/final_structures12.txt')
    table = pa.Table.from_pandas(df)
    table = table.cast(graphV.pyarrow_schema)
    table.validate()
    pq.write_table(table, '../data/final_structures12.parquet', compression='GZIP')
