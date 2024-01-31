
import os
import pandas as pd
import random as rd
import torch

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from patch_util import *



class BorderDataset(Dataset):
    reorder = True

    def __init__(self, df: pd.DataFrame, base_dir='.', transform=None):
        self.transform = transform

        self.base_dir = base_dir
        self.n_images = len(df)
        self.df = df

    def load_single_row(self, idx):
        _, img_path, *order = self.df.iloc[idx]
        patch = PatchImage(Image.open(os.path.join(self.base_dir, img_path)))
        order = [order.index(i+1) for i in range(n_patches)]
        return patch, order


    def get_image(self, idx, transform_idx):
        p1, p2, is_h = idx2transform(transform_idx)

        patch, order = self.load_single_row(idx)

        if self.reorder:
            border = patch.get_border(order[p1], order[p2], is_h)
        else:
            border = patch.get_border(p1, p2, is_h)
            p1, p2 = order.index(p1), order.index(p2)

        if self.transform is not None:
            border = self.transform(border)

        return border, 1 if is_adjacent(p1, p2, is_h) else 0

    def __len__(self):
        return self.n_images



class TrainDataset(BorderDataset):
    def __getitem__(self, idx):
        transform_idx = rd.randrange(0, n_transforms)
        return self.get_image(idx, transform_idx)


class ValidDataset(BorderDataset):
    reorder = False

    def __init__(self, df, base_dir='.', transform=None, n_per_image=1):
        super().__init__(df, base_dir, transform)
        self.n_per_image = n_per_image

    def __len__(self):
        return self.n_images * self.n_per_image

    def __getitem__(self, idx):
        img_idx, t_idx = idx // self.n_per_image, idx % self.n_per_image
        return self.get_image(img_idx, t_idx)



def image_to_batch(image):
    arr = np.array(image)
    patch = arr.reshape(
        (n_row_col, patch_size, n_row_col, patch_size, n_channels)) \
        .transpose((0, 2, 1, 3, 4)) \
        .reshape((n_patches, patch_size, patch_size, n_channels))
    vertical = np.empty(
        (n_patches, n_patches, margin * 2, patch_size, n_channels),
        dtype=np.uint8
    )
    horizontal = np.empty(
        (n_patches, n_patches, patch_size, margin * 2, n_channels),
        dtype=np.uint8
    )
    vertical[:, :, :margin] = np.expand_dims(patch[:, -margin:], axis=1)
    vertical[:, :, margin:] = np.expand_dims(patch[:, :margin], axis=0)
    horizontal[:, :, :, :margin] = np.expand_dims(patch[:, :, -margin:], axis=1)
    horizontal[:, :, :, margin:] = np.expand_dims(patch[:, :, :margin], axis=0)
    stack = np.stack((vertical, horizontal.swapaxes(2, 3)), axis=2)
    stack = stack.reshape((n_patches * n_patches * 2, 2 * margin, patch_size, n_channels))
    return stack


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir='.'):

        self.base_dir = base_dir
        self.n_images = len(df)
        self.img_path = df['img_path']

        self.transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.img_path[idx])
        batch = image_to_batch(Image.open(img_path))
        batch = torch.tensor(batch / 255, dtype=torch.float32).permute(0, 3, 1, 2)
        batch = self.transform(batch)
        return batch



