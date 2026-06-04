"""2D モデル用 DataLoader (data_loader_2d.py)

1D 版 (data_loader.py) との違い:
    - 返す単位: 1チャープ (seq_len, 2) → 1ファイルブロック (chirps, seq_len, 2)
    - バッチ形状: (B, chirps, seq_len, 2)
    - 正規化: ブロック全体の max_abs で統一 (チャープ間スケールを保持)

対応する model_2d.py の Generator2D / Critic2D と組み合わせて使用する。
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import config
import utils

_USE_RANGE_FFT: bool = config.PREPROCESS_CONFIG.get("use_range_fft", False)


# ---------------------------------------------------------------------------
# HDF5 形式 — ブロック単位 Dataset
# ---------------------------------------------------------------------------

class GANRadarHDF5Dataset2D(Dataset):
    """
    HDF5 から 1 サンプル = 1 ファイルブロック (chirps, seq_len, 2) で返す Dataset。

    __getitem__ の戻り値:
        (clean_norm, interf_norm)
            shape: (chirps, seq_len, 2) それぞれ

    正規化: ブロック全体の max_abs (チャープ間の振幅比を保持)
    """

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self._file: h5py.File | None = None

        with h5py.File(hdf5_path, "r") as f:
            for key in ("input_real", "input_imag", "label_real", "label_imag"):
                if key not in f:
                    raise KeyError(
                        f"HDF5 ファイルにキー '{key}' が存在しません: {hdf5_path}"
                    )
            self.shape = f["input_real"].shape

        # (N_files, chirps, seq_len) の 3 次元を期待
        if len(self.shape) != 3:
            raise ValueError(
                f"2D Dataset は shape=(N_files, chirps, seq_len) の HDF5 が必要です。"
                f"実際の shape: {self.shape}"
            )
        self.n_files = self.shape[0]

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self) -> int:
        return self.n_files

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        f = self._get_file()

        # (chirps, seq_len) を読み込み
        real_in = f["input_real"][idx].astype(np.float32)   # (chirps, seq_len)
        imag_in = f["input_imag"][idx].astype(np.float32)
        real_lb = f["label_real"][idx].astype(np.float32)
        imag_lb = f["label_imag"][idx].astype(np.float32)

        # (chirps, seq_len, 2)
        interf = np.stack([real_in, imag_in], axis=-1)
        clean  = np.stack([real_lb, imag_lb], axis=-1)

        # レンジFFT: fast-time 方向を周波数領域に変換 (shape は変化しない)
        if _USE_RANGE_FFT:
            interf = utils.apply_range_fft(interf)
            clean  = utils.apply_range_fft(clean)

        # ブロック全体の max_abs で正規化 (utils 関数はそのまま使える)
        # interf shape: (chirps, seq_len, 2) → expand_dims で (1, chirps, seq_len, 2)
        interf_batch = np.expand_dims(interf, axis=0)
        interf_norm, max_abs = utils.max_abs_normalize_complex_channels(interf_batch)
        # max_abs shape: (1, 1, 1) → squeeze して clean に適用
        clean_norm = clean / max_abs.squeeze(0)              # (chirps, seq_len, 2)

        return (
            torch.from_numpy(clean_norm.copy()),                    # (chirps, seq_len, 2)
            torch.from_numpy(interf_norm.squeeze(0).copy()),        # (chirps, seq_len, 2)
        )

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders_2d(
    batch_size: int,
    validation_split: float,
) -> tuple[DataLoader, DataLoader]:
    """
    2D モデル用の学習・検証 DataLoader を返す。

    返すバッチ形状:
        clean       : (B, chirps, seq_len, 2)
        interference: (B, chirps, seq_len, 2)

    HDF5 形式のみ対応 (shape が 3 次元であること)。
    """
    hdf5_path = config.DATA_CONFIG["hdf5_path"]
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 ファイルが見つかりません: {hdf5_path}")

    full_dataset = GANRadarHDF5Dataset2D(hdf5_path)

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * validation_split))
    n_train = n_total - n_val

    if n_train <= 0:
        raise ValueError(
            f"データが少なすぎます (n_total={n_total})。"
            "ファイル数を増やすか validation_split を下げてください。"
        )

    train_ds, valid_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    domain = "range-chirp (FFT適用)" if _USE_RANGE_FFT else "時間領域"
    print(
        f"[2D / HDF5 / {domain}] 総ブロック数: {n_total}  "
        f"学習: {len(train_ds)}  検証: {len(valid_ds)}"
    )
    return train_loader, valid_loader