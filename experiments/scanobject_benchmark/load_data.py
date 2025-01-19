import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../src/")
)  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP

# requirements
# for each entry in the dataset:
# - verts as as torch tensor
# - the label as the class index
# in the end ccall the compute math magic and store the output like in the
# examples


def load_h5_file(file_path: str) -> tuple[list, list]:
    with h5py.File(file_path, "r") as f:
        data = f["data"][:]  # Point cloud data (N, num_points, num_features)
        labels = f["label"][:]  # Labels (N, 1)
    return data, labels


class ScanOjectDataset(Dataset):
    def __init__(self, root_dir, k_eig, test: bool, op_cache_dir=None):
        self.root_dir = root_dir
        self.n_class = 15
        # self.split_size = split_size # pass None to take all entries (except those in exclude_dict)
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir

        self.entries = {}

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []

        pa = os.path.join(root_dir, "test" if test else "train")
        # data (2039, 2048, 3)
        # labels (2039)
        data, labels = load_h5_file(pa)

        for ind in range(len(data)):
            verts = np.array(data[ind])
            verts = torch.tensor(verts).float()

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.labels_list.append(labels[ind])

        for ind, label in enumerate(self.labels_list):
            self.labels_list[ind] = torch.tensor(label)

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = diffusion_net.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )


da, la = load_h5_file("data/h5_files/main_split_nobg/training_objectdataset.h5")
# print(len(da))
print(np.array(da[0]))
print(la[2248])
