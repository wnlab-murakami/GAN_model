"""学習済み Generator2D を使ってクリーン信号から合成干渉信号を生成するスクリプト。

1D 版 (generate.py) との違い:
    - model_2d.py の Generator2D を使用
    - 入力単位: 1チャープ → 1ファイルブロック (chirps, seq_len, 2)
    - 正規化: ブロック全体の max_abs を使用

使い方:
    1. config.py の GENERATE_CONFIG_2D["trained_model_path"] に
       学習済みモデルのディレクトリ (G_final.pth があるパス) を設定する。
    2. python generate_2d.py を実行する。

出力構造 (1D 版と同じ):
    generated_data/YYYYMMDD_HHMM_2D/
        variation_01/input/real/real_input_XXXX.txt
        variation_01/input/imag/imag_input_XXXX.txt
        variation_01/label/real/real_label_XXXX.txt
        variation_01/label/imag/imag_label_XXXX.txt
        variation_02/...
        ...
"""

import os

import h5py
import numpy as np
import torch

import config
import model_2d
import utils


def generate_2d() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    # --- 学習済み Generator2D の読み込み ---
    G, _ = model_2d.build_models_2d()
    model_path = os.path.join(
        config.GENERATE_CONFIG_2D["trained_model_path"], "G_final.pth"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません: {model_path}\n"
            "config.py の GENERATE_CONFIG_2D['trained_model_path'] を確認してください。"
        )
    G.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    G.eval().to(device)
    print(f"モデルを読み込みました: {model_path}")

    latent_dim     = config.MODEL_CONFIG["latent_dim"]
    num_variations = config.GENERATE_CONFIG_2D["num_variations"]
    out_base       = config.GENERATE_CONFIG_2D["output_path"]

    # --- HDF5 からクリーン信号ブロックを取得 ---
    hdf5_path = config.DATA_CONFIG["hdf5_path"]
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 ファイルが見つかりません: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        label_real_all = f["label_real"][:]  # (N_files, chirps, seq_len)
        label_imag_all = f["label_imag"][:]

    if label_real_all.ndim != 3:
        raise ValueError(
            f"2D 生成には shape=(N_files, chirps, seq_len) の HDF5 が必要です。"
            f"実際の shape: {label_real_all.shape}"
        )

    n_files = label_real_all.shape[0]
    # (N_files, chirps, seq_len, 2) にまとめる
    clean_blocks = np.stack(
        [label_real_all, label_imag_all], axis=-1
    ).astype(np.float32)  # (N_files, chirps, seq_len, 2)

    print(f"HDF5 から {n_files} ブロック読み込みました: {hdf5_path}")
    print(f"\n{n_files} ブロック × {num_variations} variation を生成します...")

    # --- 生成ループ ---
    for var_idx in range(num_variations):
        var_tag        = f"variation_{var_idx + 1:02d}"
        out_real       = os.path.join(out_base, var_tag, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["real_dir_name"])
        out_imag       = os.path.join(out_base, var_tag, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["imag_dir_name"])
        out_label_real = os.path.join(out_base, var_tag, config.DATA_CONFIG["label_dir_name"], config.DATA_CONFIG["real_dir_name"])
        out_label_imag = os.path.join(out_base, var_tag, config.DATA_CONFIG["label_dir_name"], config.DATA_CONFIG["imag_dir_name"])
        os.makedirs(out_real,       exist_ok=True)
        os.makedirs(out_imag,       exist_ok=True)
        os.makedirs(out_label_real, exist_ok=True)
        os.makedirs(out_label_imag, exist_ok=True)

        for file_idx in range(n_files):
            clean = clean_blocks[file_idx]  # (chirps, seq_len, 2)

            # ブロック全体の max_abs で正規化
            clean_batch = np.expand_dims(clean, axis=0)            # (1, chirps, seq_len, 2)
            clean_norm, max_abs = utils.max_abs_normalize_complex_channels(clean_batch)
            clean_tensor = torch.from_numpy(clean_norm).to(device)  # (1, chirps, seq_len, 2)

            # ランダムノイズで干渉ブロックを生成
            z = torch.randn(1, latent_dim, device=device)
            with torch.no_grad():
                fake_norm = G(clean_tensor, z)  # (1, chirps, seq_len, 2)

            # 逆正規化して元のスケールに戻す
            fake_np = fake_norm.cpu().numpy()   # (1, chirps, seq_len, 2)
            fake    = utils.max_abs_denormalize_complex_channels(fake_np, max_abs)
            fake    = fake.squeeze(0)           # (chirps, seq_len, 2)

            # テキストファイルとして保存 (1行 = 1チャープ, seq_len 列)
            file_num = file_idx + 1
            np.savetxt(
                os.path.join(out_real, f"real_input_{file_num:04d}.txt"),
                fake[:, :, 0],
                fmt="%.6e",
            )
            np.savetxt(
                os.path.join(out_imag, f"imag_input_{file_num:04d}.txt"),
                fake[:, :, 1],
                fmt="%.6e",
            )

            # クリーン信号 (label) を元のスケールに戻して保存
            clean_denorm = utils.max_abs_denormalize_complex_channels(
                clean_norm, max_abs
            ).squeeze(0)  # (chirps, seq_len, 2)
            np.savetxt(
                os.path.join(out_label_real, f"real_label_{file_num:04d}.txt"),
                clean_denorm[:, :, 0],
                fmt="%.6e",
            )
            np.savetxt(
                os.path.join(out_label_imag, f"imag_label_{file_num:04d}.txt"),
                clean_denorm[:, :, 1],
                fmt="%.6e",
            )

        print(f"  {var_tag} 完了")

    print(f"\n生成完了。出力先: {out_base}")


if __name__ == "__main__":
    generate_2d()
