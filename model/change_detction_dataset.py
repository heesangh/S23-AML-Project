# Imports

# PyTorch
import torch
from torch.utils.data import Dataset

# Other
import numpy as np
import random
from skimage import io
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import ceil

# print('IMPORTS OK')


# Global Variables' Definitions
FP_MODIFIER = 10  # Tuning parameter, use 1 if unsure
NORMALISE_IMGS = True

def read_img(path):
    image = io.imread(path)
    # Extract the red, green, and blue channels as separate arrays
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    I = np.stack((r, g, b), axis=2).astype("float")

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()
    return I


def read_img_trio(path,filename):
    """Read cropped Sentinel-2 image pair and change map."""
    #     read images
    img_a = read_img(path + "A/" + filename)
    img_b = read_img(path + "B/" + filename)
    img_label = io.imread(path + "label/" + filename, as_gray=True) != 0

    # crop if necessary
    s1 = img_a.shape
    s2 = img_b.shape
    img_b = np.pad(img_b, ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)), "edge")

    return img_a, img_b, img_label

def reshape_for_torch(I):
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)

class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(
        self,
        path,
        train=True,
        patch_side=96,
        stride=None,
        use_all_bands=False,
        transform=None,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride

        if train:
            fname = "train.txt"
        else:
            fname = "test.txt"

        self.names = read_csv(path + fname)
        self.n_imgs = self.names.shape[0]
        n_pix = 0
        true_pix = 0

        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for idx, im_name in tqdm(self.names.iterrows()):
            # for im_name in tqdm(self.names):
            # load and store each image
            im_name = im_name[0]
            I1, I2, cm = read_img_trio(path,im_name)  # TODO recall
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm

            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()

            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (
                        im_name,
                        [
                            self.stride * i,
                            self.stride * i + self.patch_side,
                            self.stride * j,
                            self.stride * j + self.patch_side,
                        ],
                        [self.stride * (i + 1), self.stride * (j + 1)],
                    )
                    self.patch_coords.append(current_patch_coords)

        self.weights = [
            FP_MODIFIER * 2 * true_pix / n_pix,
            2 * (n_pix - true_pix) / n_pix,
        ]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        I1 = self.imgs_1[im_name][:, limits[0] : limits[1], limits[2] : limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0] : limits[1], limits[2] : limits[3]]

        label = self.change_maps[im_name][limits[0] : limits[1], limits[2] : limits[3]]
        label = torch.from_numpy(1 * np.array(label)).float()

        sample = {"I1": I1, "I2": I2, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip(object):
    """Flip randomly the images in a sample."""

    def __call__(self, sample):
        I1, I2, label = sample["I1"], sample["I2"], sample["label"]

        if random.random() > 0.5:
            I1 = I1.numpy()[:, :, ::-1].copy()
            I1 = torch.from_numpy(I1)
            I2 = I2.numpy()[:, :, ::-1].copy()
            I2 = torch.from_numpy(I2)
            label = label.numpy()[:, ::-1].copy()
            label = torch.from_numpy(label)

        return {"I1": I1, "I2": I2, "label": label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    def __call__(self, sample):
        I1, I2, label = sample["I1"], sample["I2"], sample["label"]

        n = random.randint(0, 3)
        if n:
            I1 = sample["I1"].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            I2 = sample["I2"].numpy()
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I2 = torch.from_numpy(I2)
            label = sample["label"].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {"I1": I1, "I2": I2, "label": label}
