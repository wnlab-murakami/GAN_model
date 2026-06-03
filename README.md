# GAN_model — 干渉あり信号生成 (データ拡張用)

クリーン信号を条件として受け取り、合成干渉信号を生成する条件付き GAN。  
学習データが少ない場合に、干渉あり信号をデータ拡張で増やすことを目的とする。

1D モデル (チャープ単位処理) と 2D モデル (ブロック単位処理) の 2 種類を用意している。

---

## アーキテクチャ

### 1D モデル (model.py) — デフォルト

| モジュール | 内容 |
|---|---|
| **Generator** | U-Net 1D (クリーン信号 + ランダムノイズ → 合成干渉信号) |
| **Critic** | PatchGAN 1D (干渉あり信号が本物か偽物かを識別) |
| **学習方式** | WGAN-GP (Wasserstein 距離 + Gradient Penalty) |

```
クリーン信号 (B, 1024, 2)  ← 1チャープ単位
     │
  Encoder × 4  [stride=2 Conv1d]
     │   1024 → 512 → 256 → 128 → 64
     │
  Bottleneck (B, 512, 64)
  + ランダムノイズ z を加算 ← ここで多様な干渉パターンを生成
     │
  Decoder × 4  [ConvTranspose1d + skip connection]
     │   64 → 128 → 256 → 512 → 1024
     │
  合成干渉信号 (B, 1024, 2)
```

### 2D モデル (model_2d.py)

1D モデルとの主な違いは処理単位にある。1D がチャープを 1 件ずつ処理するのに対し、  
2D は 1 ファイル分のブロック (chirps × seq_len) をまとめて処理するため、  
**ドップラー方向 (チャープ間) の相関も学習できる**。

| モジュール | 内容 |
|---|---|
| **Generator2D** | U-Net 2D (クリーン信号ブロック + ランダムノイズ → 合成干渉ブロック) |
| **Critic2D** | PatchGAN 2D (干渉ブロックが本物か偽物かを識別) |
| **学習方式** | WGAN-GP (1D と同じ) |

```
クリーン信号 (B, chirps, seq_len, 2)  ← 1ブロック単位 (例: B×128×512×2)
     │
  Encoder × 4  [stride=2×2 Conv2d]
     │   (128,512) → (64,256) → (32,128) → (16,64) → (8,32)
     │
  Bottleneck (B, 512, 8, 32)
  + ランダムノイズ z を加算
     │
  Decoder × 4  [ConvTranspose2d + skip connection]
     │   (8,32) → (16,64) → (32,128) → (64,256) → (128,512)
     │
  合成干渉信号 (B, chirps, seq_len, 2)
```

| 項目 | 1D モデル | 2D モデル |
|---|---|---|
| 入力単位 | 1 チャープ `(B, seq_len, 2)` | 1 ブロック `(B, chirps, seq_len, 2)` |
| チャープ間相関 | 学習しない | 学習する |
| Conv 層 | Conv1d / ConvTranspose1d | Conv2d / ConvTranspose2d |
| 必要 VRAM | 少 | 多 (batch_size を下げる必要あり) |
| HDF5 shape 要件 | 2D・3D 両対応 | 3D `(N_files, chirps, seq_len)` のみ |

---

## ディレクトリ構成

```
GAN_model/
├── config.py           # データパス・モデル・学習設定
├── model.py            # Generator, Critic, Gradient Penalty (1D)
├── model_2d.py         # Generator2D, Critic2D, Gradient Penalty (2D)
├── data_loader.py      # Dataset / DataLoader — 1D (txt / HDF5 両対応)
├── data_loader_2d.py   # Dataset / DataLoader — 2D (HDF5 専用, ブロック単位)
├── utils.py            # テキスト読み込み・正規化関数
├── train.py            # 学習スクリプト — 1D (WGAN-GP)
├── train_2d.py         # 学習スクリプト — 2D (WGAN-GP)
├── generate.py         # データ拡張スクリプト — 1D
├── generate_2d.py      # データ拡張スクリプト — 2D
├── requirements.txt    # 必要パッケージ
│
├── learning_data/      # 学習データ (以下どちらかの形式を用意)
│   │
│   │  【HDF5 形式 (推奨)】
│   └── gan_combined_data.hdf5   # キー: input_real / input_imag / label_real / label_imag
│                                 # shape: (N_files, chirps_per_file, sequence_length)  ← 2D モデルはこちらのみ対応
│                                 #     or (N_chirps, sequence_length)                  ← 1D モデルのみ対応
│
│      【テキスト形式 (1D モデルのみ)】
│   ├── input/
│   │   ├── real/  real_input_0001.txt ...  ← 干渉あり信号 (実部)
│   │   └── imag/  imag_input_0001.txt ...  ← 干渉あり信号 (虚部)
│   └── label/
│       ├── real/  real_label_0001.txt ...  ← クリーン信号 (実部)
│       └── imag/  imag_label_0001.txt ...  ← クリーン信号 (虚部)
│
├── saved_models/       # 学習済みモデルの保存先 (自動生成)
│   ├── GAN_YYYYMMDD_HHMM/        # 1D モデル
│   │   ├── G_final.pth
│   │   ├── D_final.pth
│   │   └── checkpoint_epoch0020.pth ...
│   └── GAN_YYYYMMDD_HHMM_2D/     # 2D モデル
│       ├── G_final.pth
│       ├── D_final.pth
│       └── checkpoint_epoch0020.pth ...
│
└── generated_data/     # 生成データの出力先 (自動生成)
    ├── YYYYMMDD_HHMM/            # 1D モデルによる生成
    │   ├── variation_01/
    │   │   ├── input/real/real_input_0001.txt ...
    │   │   ├── input/imag/imag_input_0001.txt ...
    │   │   ├── label/real/real_label_0001.txt ...
    │   │   └── label/imag/imag_label_0001.txt ...
    │   └── variation_02/ ...
    └── YYYYMMDD_HHMM_2D/         # 2D モデルによる生成 (出力形式は 1D と同じ)
        ├── variation_01/ ...
        └── variation_02/ ...
```

---

## セットアップ

```bash
pip install -r requirements.txt
```

---

## 使い方

### 1. データの配置

`config.py` の `DATA_CONFIG["data_format"]` に応じてデータを配置する。

**HDF5 形式 (推奨 / 2D モデルを使う場合は必須)**

`learning_data/gan_combined_data.hdf5` を配置する。  
HDF5 ファイルには以下の 4 つのキーが必要である。

| キー | 内容 |
|---|---|
| `input_real` | 干渉あり信号 (実部) |
| `input_imag` | 干渉あり信号 (虚部) |
| `label_real` | クリーン信号 (実部) |
| `label_imag` | クリーン信号 (虚部) |

shape は `(N_files, chirps_per_file, sequence_length)` または `(N_chirps, sequence_length)` の両方に対応する。  
ただし **2D モデルは `(N_files, chirps_per_file, sequence_length)` の 3 次元のみ対応**する。

**テキスト形式 (1D モデルのみ)**

`config.py` の `data_format` を `"txt"` に変更し、`learning_data/` 以下にテキストファイルを配置する。  
1 ファイル = `chirps_per_file` チャープ (行) × `sequence_length` サンプル (列) の形式。

### 2. 設定の変更 (config.py)

必要に応じて以下の項目を変更する。

```python
DATA_CONFIG = {
    "data_format": "hdf5",   # "hdf5" または "txt" (2D モデルは "hdf5" 固定)
    "hdf5_path":   "learning_data/gan_combined_data.hdf5",
}

PREPROCESS_CONFIG = {
    "sequence_length":  1024,  # モデルに入力するシーケンスの長さ (サンプル数)
    "chirps_per_file":  128,   # 1ファイルあたりのチャープ数 (HDF5 形式で使用)
}

TRAIN_CONFIG = {
    "epochs":        200,
    "batch_size":    16,     # 2D モデルは VRAM に応じて 4〜8 に下げる
    "lr_g":          1e-4,   # Generator 学習率
    "lr_d":          4e-4,   # Critic 学習率
    "n_critic":      5,      # Critic 更新回数 / Generator 1 回
    "lambda_gp":     10.0,   # Gradient Penalty 重み
    "lambda_l1":     0.0,    # L1 Loss 重み (0 = 多様性重視, >0 = 実データに近づける)
    "use_wandb":     True,   # wandb でログを記録するか
    "save_interval": 20,     # 何エポックごとにチェックポイントを保存するか
}

MODEL_CONFIG = {
    "latent_dim":    64,   # ノイズベクトルの次元数
    "base_channels": 64,   # Conv のベースチャンネル数 (1D / 2D 共用)
}

GENERATE_CONFIG = {
    "trained_model_path": "saved_models/GAN_YYYYMMDD_HHMM",    # 1D モデルのパス
    "num_variations":     5,
}

GENERATE_CONFIG_2D = {
    "trained_model_path": "saved_models/GAN_YYYYMMDD_HHMM_2D", # 2D モデルのパス
    "num_variations":     5,
}
```

### 3. 学習

**1D モデル**

```bash
python train.py
```

**2D モデル**

```bash
python train_2d.py
```

学習済みモデルはそれぞれ以下に保存される。

| モデル | 保存先 |
|---|---|
| 1D | `saved_models/GAN_YYYYMMDD_HHMM/` |
| 2D | `saved_models/GAN_YYYYMMDD_HHMM_2D/` |

**学習の目安:**
- Loss D が安定して小さな値に収束し、Loss G がゆっくり下がれば正常。
- WGAN-GP では D Loss が負値になることがある (正常動作)。
- Val L1 が下がらない場合はエポック数を増やすか `lambda_l1` を少し上げる。
- 2D モデルは 1 サンプルが `(chirps, seq_len, 2)` と大きいため VRAM 不足になりやすい。その場合は `batch_size` を `4〜8` に下げる。

### 4. データ拡張 (干渉あり信号の生成)

**1D モデル**

`config.py` の `GENERATE_CONFIG["trained_model_path"]` に学習済みモデルのパスを設定してから実行する。

```bash
python generate.py
```

**2D モデル**

`config.py` の `GENERATE_CONFIG_2D["trained_model_path"]` に学習済みモデルのパスを設定してから実行する。

```bash
python generate_2d.py
```

どちらも `generated_data/` 以下に同じディレクトリ構造で出力される。  
各 variation に干渉あり信号 (`input/`) とクリーン信号 (`label/`) の両方が保存される。

---

## 各ファイルの説明

### config.py
全設定を一元管理するファイル。データパス、モデル構造、学習率などをここで変更する。  
1D / 2D で共用する設定 (`MODEL_CONFIG`, `TRAIN_CONFIG` など) と、  
2D 生成専用の設定 (`GENERATE_CONFIG_2D`) が含まれる。

### model.py / model_2d.py
- `Generator` / `Generator2D`: U-Net 型。クリーン信号とノイズから干渉信号を生成する。
- `Critic` / `Critic2D`: PatchGAN 型。干渉信号が本物かどうかをパッチ単位で判定する。
- `compute_gradient_penalty` / `compute_gradient_penalty_2d`: WGAN-GP の勾配ペナルティ計算。
- 2D 版は Conv1d → Conv2d に置き換えており、ボトルネックで空間次元 `(chirps, seq_len)` を同時に扱う。

### data_loader.py / data_loader_2d.py
- `data_loader.py`: 1 チャープを 1 サンプルとして返す。txt / HDF5 両対応。
- `data_loader_2d.py`: 1 ブロック `(chirps, seq_len, 2)` を 1 サンプルとして返す。HDF5 専用。
- どちらも正規化は干渉あり信号の max_abs で統一し、クリーン信号と干渉信号のスケール関係を保持する。

### train.py / train_2d.py
WGAN-GP の学習ループ。1 エポックごとに以下を実行する。
1. Critic を `n_critic` 回更新 (Wasserstein 距離 + Gradient Penalty)
2. Generator を 1 回更新
3. 検証データで L1 距離を計測してログ出力

`train_2d.py` は `model_2d` / `data_loader_2d` を使用し、保存先に `_2D` サフィックスを付加する点のみ異なる。

### generate.py / generate_2d.py
学習済み Generator を使い、クリーン信号から複数バリエーションの合成干渉信号を生成する。  
異なるランダムノイズ `z` を使うことで、1 つのクリーン信号から多様な干渉パターンを生成できる。  
出力形式はどちらも同じテキスト形式 (`input/` + `label/`) であるため、後段の処理をそのまま流用できる。

---

## 注意事項

- **生成時の振幅スケール**: 学習時は干渉信号の max_abs で正規化しているが、生成時はクリーン信号の max_abs を代用するため、振幅が実データとわずかに異なる場合がある (1D / 2D 共通)。
- **2D モデルの HDF5 shape 要件**: `(N_files, chirps_per_file, sequence_length)` の 3 次元が必須。2 次元の HDF5 では `data_loader_2d.py` がエラーを返す。
- **2D モデルの VRAM**: 1 サンプルが `(128, 512, 2)` と大きいため、`batch_size=4〜8` 程度が目安。
- **`lambda_l1` の設定**: `0.0` にすると多様性が最大になる。実データに近い信号が欲しい場合は `1.0〜10.0` に設定する。
- **wandb が不要な場合**: `config.py` の `use_wandb` を `False` に設定する。
- **variation 間の label**: どの variation も同一のクリーン信号から生成しているため、`label/` の内容は各 variation で同じになる (1D / 2D 共通)。