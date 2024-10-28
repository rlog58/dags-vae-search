import math
import os
import time

import torch
import tqdm
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.datasets import LabeledDagDatasetInMemory
from src.encoders.pace import PaceVae
from src.encoders.pace_utils import PaceDag
from src.toolkit.labeled import LabeledDag
from src.train_utils import collate_graph_batch


def train_batch(batch, model, optimizer):
    optimizer.zero_grad()

    loss, recon, kld = model.loss(batch)

    loss_value = float(loss)

    loss.backward()
    optimizer.step()

    return loss_value, recon, kld


if __name__ == '__main__':
    toolkit = LabeledDag(num_vertices=8, label_cardinality=8)

    BATCH_SIZE = 32
    EPOCHS = 20

    dataset = LabeledDagDatasetInMemory(
        dataset_dir="../data/asia_200k_train/",
        toolkit=toolkit,
    )

    dl = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_graph_batch,
        shuffle=True,
        # num_workers=2,
    )

    model = PaceVae(
        PaceDag(num_vertices=8, label_cardinality=8),
        ninp=32,
        nhead=8,
        nhid=64,
        nlayers=3,
        dropout=0.15,
        fc_hidden=32,
        nz=32
    )
    #  12 + 3 -  402384
    #  20 + 3 -  502744
    # 100 + 3 - 1506344
    print("Model params count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    # Junk.
    # model_name = os.path.join("../results", 'model_checkpoint_100.pth')
    # load_model_state(model, model_name)

    loss_value = math.inf
    time_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        progress_bar = tqdm.tqdm(dl, desc=f"Epoch {epoch} Training")

        for batch in progress_bar:
            loss_value, recon, kld = train_batch(batch, model, optimizer)

            progress_bar.set_description(
                f'Epoch: {epoch}, loss: {loss_value / BATCH_SIZE:.4f}, recon: {recon / BATCH_SIZE:.4f}, kld: {kld / BATCH_SIZE:.4f}')

        scheduler.step(loss_value)

        comp_time = time.time() - time_start
        print('====> Epoch: {0} loss: {1:.4f}, compute time: {2:.4f}'.format(epoch, loss_value / BATCH_SIZE, comp_time))

        model_name = os.path.join("../results", 'model_checkpoint_{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)

    print("Done")

    print("Try to load model")
    loaded_model = torch.load(model_name)
