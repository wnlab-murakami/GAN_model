"""2D Convolutional GAN モデル (model_2d.py)

1D モデル (model.py) との違い:
    - 入力単位: 1チャープ (B, T, 2) → 1ファイル分ブロック (B, chirps, T, 2)
    - Conv1d → Conv2d / ConvTranspose1d → ConvTranspose2d
    - ドップラー方向 (チャープ間) の相関も学習できる
    - ボトルネック空間サイズ:  (chirps/16, seq_len/16) = (8, 32) [128×512 の場合]

使い方:
    import model_2d
    G, D = model_2d.build_models_2d()

データ形状 (data_loader_2d.py と組み合わせて使用):
    clean      : (B, chirps, seq_len, 2)
    interference: (B, chirps, seq_len, 2)
"""

import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------------------
# Building blocks (2D 版)
# ---------------------------------------------------------------------------

class EncoderBlock2D(nn.Module):
    """Conv2d (stride=2×2) + optional InstanceNorm2d + LeakyReLU"""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=4, stride=2, padding=1,
                bias=not use_norm,
            )
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock2D(nn.Module):
    """ConvTranspose2d (stride=2×2) + InstanceNorm2d + ReLU + optional Dropout"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=4, stride=2, padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Generator2D  (U-Net 2D)
# ---------------------------------------------------------------------------

class Generator2D(nn.Module):
    """
    2D U-Net Generator。
    クリーン信号ブロックを条件として受け取り、合成干渉ブロックを生成する。

    入力:
        clean : (B, chirps, seq_len, C)   — 例: (B, 128, 512, 2)
        noise : (B, latent_dim)
    出力:
        interference : (B, chirps, seq_len, C)

    エンコーダ: ×4 ダウンサンプリング
        (B, 2, 128, 512)
        → enc1 (B, 64,  64, 256)   stride=2×2
        → enc2 (B, 128, 32, 128)   stride=2×2
        → enc3 (B, 256, 16,  64)   stride=2×2
        → enc4 (B, 512,  8,  32)   stride=2×2  ← ボトルネック

    デコーダ: スキップ接続あり (U-Net)
        → dec4 (B, 256, 16,  64)   + skip enc3 → (B, 512, 16,  64)
        → dec3 (B, 128, 32, 128)   + skip enc2 → (B, 256, 32, 128)
        → dec2 (B,  64, 64, 256)   + skip enc1 → (B, 128, 64, 256)
        → dec1 (B,   2,128, 512)   Tanh 出力
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 64,
        base_ch: int = 64,
    ):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock2D(in_channels,  base_ch,     use_norm=False)
        self.enc2 = EncoderBlock2D(base_ch,      base_ch * 2)
        self.enc3 = EncoderBlock2D(base_ch * 2,  base_ch * 4)
        self.enc4 = EncoderBlock2D(base_ch * 4,  base_ch * 8)  # ボトルネック

        # Noise injection: z → (B, base_ch*8, 1, 1) → ボトルネックに加算
        self.noise_proj = nn.Linear(latent_dim, base_ch * 8)
        self.noise_norm = nn.InstanceNorm2d(base_ch * 8, affine=True)

        # Decoder (スキップ接続で in_ch が 2 倍)
        self.dec4 = DecoderBlock2D(base_ch * 8,      base_ch * 4, dropout=0.5)
        self.dec3 = DecoderBlock2D(base_ch * 4 * 2,  base_ch * 2, dropout=0.5)
        self.dec2 = DecoderBlock2D(base_ch * 2 * 2,  base_ch)
        self.dec1 = nn.ConvTranspose2d(
            base_ch * 2, in_channels,
            kernel_size=4, stride=2, padding=1,
        )

        self.out_act = nn.Tanh()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(
        self, clean: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        # (B, chirps, seq_len, C) → (B, C, chirps, seq_len)
        x = clean.permute(0, 3, 1, 2)

        # Encode
        e1 = self.enc1(x)   # (B, 64,  64, 256)
        e2 = self.enc2(e1)  # (B, 128, 32, 128)
        e3 = self.enc3(e2)  # (B, 256, 16,  64)
        e4 = self.enc4(e3)  # (B, 512,  8,  32)

        # Noise injection: (B, latent_dim) → (B, 512, 1, 1) → broadcast
        nf = self.noise_proj(noise).unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        nf = self.noise_norm(nf.expand_as(e4))                   # (B, 512, 8, 32)
        e4 = e4 + nf

        # Decode with skip connections
        d4 = self.dec4(e4)               # (B, 256, 16,  64)
        d4 = torch.cat([d4, e3], dim=1)  # (B, 512, 16,  64)

        d3 = self.dec3(d4)               # (B, 128, 32, 128)
        d3 = torch.cat([d3, e2], dim=1)  # (B, 256, 32, 128)

        d2 = self.dec2(d3)               # (B,  64, 64, 256)
        d2 = torch.cat([d2, e1], dim=1)  # (B, 128, 64, 256)

        d1 = self.dec1(d2)               # (B,   2,128, 512)
        out = self.out_act(d1)

        # (B, C, chirps, seq_len) → (B, chirps, seq_len, C)
        return out.permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# Critic2D  (PatchGAN 2D, WGAN-GP 用)
# ---------------------------------------------------------------------------

class Critic2D(nn.Module):
    """
    2D PatchGAN Critic。
    interference と clean を channel 方向に連結して入力する。

    入力:
        interference : (B, chirps, seq_len, C)
        clean        : (B, chirps, seq_len, C)
    出力:
        scores : (B, 1, H', W')  — PatchGAN スコア (WGAN-GP: sigmoid なし)
    """

    def __init__(self, in_channels: int = 2, base_ch: int = 64):
        super().__init__()

        # concat(interference, clean) → (B, in_ch*2, chirps, seq_len)
        self.net = nn.Sequential(
            # Layer 1: InstanceNorm なし
            nn.Conv2d(in_channels * 2, base_ch,     4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(base_ch,     base_ch * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (B, 1, H', W')
            nn.Conv2d(base_ch * 8, 1, kernel_size=3, stride=1, padding=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, interference: torch.Tensor, clean: torch.Tensor
    ) -> torch.Tensor:
        # (B, chirps, seq_len, C) → (B, C, chirps, seq_len)
        x_i = interference.permute(0, 3, 1, 2)
        x_c = clean.permute(0, 3, 1, 2)
        x = torch.cat([x_i, x_c], dim=1)  # (B, 2C, chirps, seq_len)
        return self.net(x)                 # (B, 1, H', W')


# ---------------------------------------------------------------------------
# Gradient Penalty (WGAN-GP) — 2D 版
# ---------------------------------------------------------------------------

def compute_gradient_penalty_2d(
    critic: Critic2D,
    clean: torch.Tensor,
    real_interf: torch.Tensor,
    fake_interf: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP の勾配ペナルティ (2D 版)。"""
    B = real_interf.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)  # 1D は (B,1,1) だが 2D は (B,1,1,1)

    interpolated = (
        alpha * real_interf.detach() + (1.0 - alpha) * fake_interf.detach()
    ).requires_grad_(True)

    score = critic(interpolated, clean)

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.reshape(B, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_models_2d() -> tuple[Generator2D, Critic2D]:
    """設定に基づいて Generator2D と Critic2D を構築して返す。"""
    G = Generator2D(
        in_channels=config.PREPROCESS_CONFIG["num_features"],
        latent_dim=config.MODEL_CONFIG["latent_dim"],
        base_ch=config.MODEL_CONFIG["base_channels"],
    )
    D = Critic2D(
        in_channels=config.PREPROCESS_CONFIG["num_features"],
        base_ch=config.MODEL_CONFIG["base_channels"],
    )
    return G, D
