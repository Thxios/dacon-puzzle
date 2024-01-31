

import numpy as np



patch_size = 128
n_row_col = 4
margin = 16
n_patches = n_row_col ** 2
image_size = patch_size * n_row_col
n_channels = 3
n_pairs = n_patches * (n_patches - 1)
n_transforms = n_pairs * 2



def coord2idx(r, c):
    return r * n_row_col + c
def idx2coord(idx):
    return idx // n_row_col, idx % n_row_col

def idx2transform(idx):
    if not (0 <= idx < n_transforms):
        raise ValueError(f'{idx} is not a valid transform in [0, {n_transforms})')

    is_horizontal = bool(idx // n_pairs)
    pair_idx = idx % n_pairs
    a, b = pair_idx // (n_patches - 1), pair_idx % (n_patches - 1)
    if b >= a:
        b += 1

    return a, b, is_horizontal

def transform2idx(p1, p2, is_horizontal):
    assert p1 != p2
    assert 0 <= p1 < n_patches and 0 <= p2 < n_patches

    if p2 > p1:
        p2 -= 1
    ret = p1 * (n_patches - 1) + p2
    if is_horizontal:
        ret += n_pairs
    return ret

def is_adjacent(idx1, idx2, is_horizontal):
    r1, c1 = idx2coord(idx1)
    r2, c2 = idx2coord(idx2)

    if not is_horizontal:
        return r1 + 1 == r2 and c1 == c2
    else:
        return c1 + 1 == c2 and r1 == r2



class PatchImage:
    def __init__(self, image):
        img_arr = np.array(image)
        assert img_arr.shape == (image_size, image_size, n_channels)

        self.patches = img_arr.reshape(
            (n_row_col, patch_size, n_row_col, patch_size, n_channels))\
            .transpose((0, 2, 1, 3, 4))\
            .reshape((n_patches, patch_size, patch_size, n_channels))

    def get_whole_image(self):
        return self.patches.reshape(
            (n_row_col, n_row_col, patch_size, patch_size, n_channels))\
            .transpose((0, 2, 1, 3, 4))\
            .reshape((image_size, image_size, n_channels))

    def get_patch(self, idx):
        return self.patches[idx]

    def merge_vertical(self, top_idx, bottom_idx):
        img1, img2 = self.patches[top_idx], self.patches[bottom_idx]
        border = np.empty((margin * 2, patch_size, n_channels), dtype=np.uint8)
        border[:margin] = img1[-margin:]
        border[margin:] = img2[:margin]
        return border

    def merge_horizontal(self, top_idx, bottom_idx):
        img1, img2 = self.patches[top_idx], self.patches[bottom_idx]
        border = np.empty((patch_size, margin * 2, n_channels), dtype=np.uint8)
        border[:, :margin] = img1[:, -margin:]
        border[:, margin:] = img2[:, :margin]
        return border

    def get_border(self, idx1, idx2, is_horizontal):
        if not is_horizontal:
            return self.merge_vertical(idx1, idx2)
        else:
            return self.merge_horizontal(idx1, idx2).swapaxes(0, 1)




