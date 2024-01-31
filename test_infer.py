

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import fire

from torchvision.models import mobilenet_v3_large
import tqdm.auto as tqdm

from patch_util import *
from dataset import TestDataset



def get_model(ckpt):

    model = mobilenet_v3_large(num_classes=1)
    ret = model.load_state_dict(torch.load(ckpt))
    print(ret)
    return model


def main(
        ckpt_path: str,
        data_dir: str = 'data',
        output_dir: str = 'logits',
        csv_name: str = 'test.csv',
        device: str = 'cuda',
):
    test_df = pd.read_csv(os.path.join(data_dir, csv_name))
    print(test_df)

    ds = TestDataset(test_df, base_dir=data_dir)

    loader = DataLoader(ds, batch_size=None, num_workers=2, shuffle=False)
    model = get_model(ckpt_path)
    model.to(device)
    model.eval()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rets = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(loader)):
            batch = batch.to(device)
            out = model(batch)
            out_np = out.squeeze().cpu().numpy()
            rets.append(out_np)
            np.save(os.path.join(output_dir, f'{i:05d}.npy'), out_np)




if __name__ == '__main__':
    fire.Fire(main)



