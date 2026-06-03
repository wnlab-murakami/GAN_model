import glob
import os

import h5py
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset, random_split

import config
import utils


class GANRadarDataset(Dataset):
    """
    GAN 用データセット。

    Generator の condition: クリーン信号 (label/)
    Critic の real sample : 干渉あり信号 (input/)

    __getitem__ の戻り値:
        (clean_norm, interf_norm) — 干渉あり信号の最大絶対値で両方を正規化
    """

    def __init__(
        self,
        real_input_files: list[str],
        imag_input_files: list[str],
        real_label_files: list[str],
        imag_label_files: list[str],
    ):
        self.real_input_files = real_input_files
        self.imag_input_files = imag_input_files
        self.real_label_files = real_label_files
        self.imag_label_files = imag_label_files
        self.chirps_per_file = config.PREPROCESS_CONFIG["chirps_per_file"]

    def __len__(self) -> int:
        return len(self.real_input_files) * self.chirps_per_file

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx = idx // self.chirps_per_file
        chirp_idx = idx % self.chirps_per_file

        # 干渉あり信号 (input) を読み込む
        real_in = utils.load_data_from_txt(self.real_input_files[file_idx])
        imag_in = utils.load_data_from_txt(self.imag_input_files[file_idx])
        interf = np.stack(
            [real_in[chirp_idx, :], imag_in[chirp_idx, :]], axis=-1
        ).astype(np.float32)   # (512, 2)

        # クリーン信号 (label) を読み込む
        real_lb = utils.load_data_from_txt(self.real_label_files[file_idx])
        imag_lb = utils.load_data_from_txt(self.imag_label_files[file_idx])
        clean = np.stack(
            [real_lb[chirp_idx, :], imag_lb[chirp_idx, :]], axis=-1
        ).astype(np.float32)   # (512, 2)

        # 干渉あり信号の max_abs で両方を正規化 (スケール関係を保持)
        interf_batch = np.expand_dims(interf, axis=0)                       # (1, 512, 2)
        interf_norm, max_abs = utils.max_abs_normalize_complex_channels(interf_batch)
        clean_norm = clean / max_abs.squeeze(0)                             # (512, 2)

        return (
            torch.from_numpy(clean_norm.copy()),   # (512, 2)
            torch.from_numpy(interf_norm.squeeze(0).copy()),  # (512, 2)
        )


def get_dataloaders(
    batch_size: int, validation_split: float
) -> tuple[DataLoader, DataLoader]:
    """学習用と検証用の DataLoader を返す。"""
    base = config.DATA_CONFIG["learning_data_path"]

    input_real = os.path.join(base, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["real_dir_name"])
    input_imag = os.path.join(base, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["imag_dir_name"])
    label_real = os.path.join(base, config.DATA_CONFIG["label_dir_name"], config.DATA_CONFIG["real_dir_name"])
    label_imag = os.path.join(base, config.DATA_CONFIG["label_dir_name"], config.DATA_CONFIG["imag_dir_name"])

    real_input_files = natsorted(
        glob.glob(os.path.join(input_real, config.DATA_CONFIG["file_pattern"]))
    )
    if not real_input_files:
        raise FileNotFoundError(f"学習データが見つかりません: {input_real}")

    imag_input_files = [
        f.replace(input_real, input_imag).replace("real_", "imag_")
        for f in real_input_files
    ]
    real_label_files = [
        f.replace(input_real, label_real).replace("input", "label")
        for f in real_input_files
    ]
    imag_label_files = [
        f.replace(input_real, label_imag).replace("input", "label").replace("real_", "imag_")
        for f in real_input_files
    ]

    full_dataset = GANRadarDataset(
        real_input_files, imag_input_files, real_label_files, imag_label_files
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * validation_split))
    n_train = n_total - n_val

    if n_train <= 0:
        raise ValueError("データが少なすぎます。")

    train_ds, valid_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"総サンプル数: {n_total}  学習: {len(train_ds)}  検証: {len(valid_ds)}")
    return train_loader, valid_loader


class GANRadarHDF5Dataset(Dataset):
    """
    HDF5 形式データセット。
    [N_chirps, sequence_length] (2次元) または 
    [N_files, chirps_per_file, sequence_length] (3次元) の両方に対応。
    """

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self._file: h5py.File | None = None
        with h5py.File(hdf5_path, "r") as f:
            for key in ("input_real", "input_imag", "label_real", "label_imag"):
                if key not in f:
                    raise KeyError(f"HDF5 ファイルにキー '{key}' が存在しません: {hdf5_path}")
            
            self.shape = f["input_real"].shape
            if len(self.shape) == 3:
                self.n_samples = self.shape[0] * self.shape[1]
                self.chirps_per_file = self.shape[1]
            else:
                self.n_samples = self.shape[0]
                self.chirps_per_file = 1

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        f = self._get_file()
        
        if len(self.shape) == 3:
            file_idx = idx // self.chirps_per_file
            chirp_idx = idx % self.chirps_per_file
            real_in = f["input_real"][file_idx, chirp_idx].astype(np.float32)
            imag_in = f["input_imag"][file_idx, chirp_idx].astype(np.float32)
            real_lb = f["label_real"][file_idx, chirp_idx].astype(np.float32)
            imag_lb = f["label_imag"][file_idx, chirp_idx].astype(np.float32)
        else:
            real_in = f["input_real"][idx].astype(np.float32)
            imag_in = f["input_imag"][idx].astype(np.float32)
            real_lb = f["label_real"][idx].astype(np.float32)
            imag_lb = f["label_imag"][idx].astype(np.float32)

        interf = np.stack([real_in, imag_in], axis=-1)  # (512, 2)
        clean  = np.stack([real_lb, imag_lb], axis=-1)  # (512, 2)

        interf_batch = np.expand_dims(interf, axis=0)                        # (1, 512, 2)
        interf_norm, max_abs = utils.max_abs_normalize_complex_channels(interf_batch)
        clean_norm = clean / max_abs.squeeze(0)                              # (512, 2)

        return (
            torch.from_numpy(clean_norm.copy()),              # (512, 2)
            torch.from_numpy(interf_norm.squeeze(0).copy()),  # (512, 2)
        )

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()


def get_dataloaders_hdf5(
    batch_size: int, validation_split: float
) -> tuple[DataLoader, DataLoader]:
    """HDF5 形式データから学習用・検証用 DataLoader を返す。"""
    hdf5_path = config.DATA_CONFIG["hdf5_path"]
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 ファイルが見つかりません: {hdf5_path}")

    full_dataset = GANRadarHDF5Dataset(hdf5_path)

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * validation_split))
    n_train = n_total - n_val

    if n_train <= 0:
        raise ValueError("データが少なすぎます。")

    train_ds, valid_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"[HDF5] 総サンプル数: {n_total}  学習: {len(train_ds)}  検証: {len(valid_ds)}")
    return train_loader, valid_loader
