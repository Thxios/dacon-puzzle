
import os
import numpy as np
import random as rd
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
import tqdm.auto as tqdm
import fire

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from dataset import TrainDataset, ValidDataset


def get_dataset(data_dir, test_size=0.1, seed=42):
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed
    )

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TrainDataset(train_df, base_dir=data_dir, transform=transform)
    val_ds = ValidDataset(val_df, base_dir=data_dir, transform=transform)

    return train_ds, val_ds


def get_model():
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(1280, 1)
    return model


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    rd.seed(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    rd.seed(worker_seed)


def train(
        data_dir: str,
        output_dir: str,
        lr: float = 2e-4,
        batch_size: int = 128,
        eval_batch_size: int = 256,
        max_epochs: int = 20,
        test_size: int = 6000,
        logging_steps: int = 20,
        device: str = 'cuda:0',
        seed: int = 42,
):
    seed_everything(seed)

    train_ds, val_ds = get_dataset(data_dir, test_size=test_size, seed=seed)
    print(f'train: {len(train_ds)}')
    print(f'val: {len(val_ds)}')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        num_workers=2,
    )


    model = get_model()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def evaluate():
        model.eval()
        logits, labels = [], []
        for x_, y_ in tqdm.tqdm(val_loader, leave=False):
            x_, y_ = x_.to(device), y_.to(device)
            logit = model(x_)
            logit = logit.squeeze()
            logits.append(logit.cpu())
            labels.append(y_.cpu())

        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        loss_ = loss_fn(logits, labels.to(torch.float32)).item()
        preds_ = (logits > 0).long().numpy()
        labels = labels.numpy()
        return dict(
            loss=loss_,
            acc=accuracy_score(labels, preds_),
            f1=f1_score(labels, preds_),
            recall=recall_score(labels, preds_),
        )


    model.to(device)
    os.makedirs(output_dir, exist_ok=True)

    losses = []
    steps = 0
    for epoch in tqdm.tqdm(range(max_epochs)):
        # print(evaluate())
        model.train()
        with tqdm.tqdm(total=len(train_loader), desc=f'Epoch{epoch:02d}') as pbar:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                pred = pred.squeeze()
                y = y.to(torch.float32)
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()
                model.zero_grad()
                pbar.update()

                steps += 1
                losses.append(loss.item())
                if steps % logging_steps == 0:
                    pbar.set_postfix({'loss': f'{np.mean(losses):.4f}'})
                    losses.clear()

        eval_res = evaluate()
        save_name = f'ep{epoch:02d}_acc{eval_res["acc"]:.4f}_f{eval_res["f1"]:.4f}_rec{eval_res["recall"]:.4f}.pt'
        torch.save(model.state_dict(), os.path.join(output_dir, save_name))
        print(eval_res)


if __name__ == '__main__':
    fire.Fire(train)


