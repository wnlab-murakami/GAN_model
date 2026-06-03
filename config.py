import os
import datetime

DT_NOW = datetime.datetime.now()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_CONFIG = {
    "learning_data_path": os.path.join(BASE_DIR, "learning_data"),
    "test_data_path":     os.path.join(BASE_DIR, "test_data"),
    "input_dir_name":     "input",    # 干渉あり信号 (Criticのreal sample)
    "label_dir_name":     "label",    # クリーン信号   (Generatorのcondition)
    "real_dir_name":      "real",
    "imag_dir_name":      "imag",
    "file_pattern":       "real_input_*.txt",
    # HDF5 設定 (data_format="hdf5" のとき使用)
    "hdf5_path":          os.path.join(BASE_DIR, "learning_data", "gan_combined_data.hdf5"),
    # "txt": テキストファイル形式  /  "hdf5": HDF5 形式
    "data_format":        "hdf5",
}

PREPROCESS_CONFIG = {
    "normalization_method": "max_abs_scaling",
    "sequence_length":      1024,  # モデルに入力するシーケンスの長さ (サンプル数)
    "num_features":         2,    # [実部, 虚部]
    "chirps_per_file":      128,  # 1ファイルあたりのチャープ数 (HDF5 形式で使用)
}

MODEL_CONFIG = {
    "latent_dim":    64,   # ノイズベクトルの次元数
    "base_channels": 64,   # Conv1Dのベースチャンネル数
}

TRAIN_CONFIG = {
    "epochs":            200,
    "batch_size":        16,
    "lr_g":              1e-4,   # Generator 学習率
    "lr_d":              4e-4,   # Critic 学習率 (TTUR: D > G)
    "n_critic":          5,      # Critic 更新回数 / Generator 1回
    "lambda_gp":         10.0,   # Gradient Penalty 重み
    "lambda_l1":         0.0,    # L1 Loss 重み (0 = 使わない、多様性重視)
    "validation_split":  0.2,
    "save_interval":     20,     # 何エポックごとにチェックポイントを保存するか
    "use_wandb":         True,
    "model_save_path":   os.path.join(BASE_DIR, "saved_models", f"GAN_{DT_NOW:%Y%m%d_%H%M}"),
}

GENERATE_CONFIG = {
    # 学習済みモデルのディレクトリ (train.py 実行後に G_final.pth があるパスを指定)
    "trained_model_path":   os.path.join(BASE_DIR, "saved_models", "GAN_20260528_2233"),
    "source_data_path":     os.path.join(BASE_DIR, "learning_data"),
    "output_path":          os.path.join(BASE_DIR, "generated_data", f"{DT_NOW:%Y%m%d_%H%M}"),
    "num_variations":       5,   # 1つのクリーン信号から生成する干渉信号のバリエーション数
}
