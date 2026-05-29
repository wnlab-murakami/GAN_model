import torch
import torch.nn as nn
import config


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Conv1d (stride=2) + optional InstanceNorm + LeakyReLU"""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm1d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """ConvTranspose1d (stride=2) + InstanceNorm + ReLU + optional Dropout"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Generator  (U-Net 1D, condition + noise → synthetic interference)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    クリーン信号を条件として受け取り、乱数ノイズと組み合わせて
    合成干渉信号を生成する U-Net 型 Generator。

    入力:
        clean : (B, T, C) = (B, 512, 2)
        noise : (B, latent_dim)
    出力:
        interference : (B, T, C) = (B, 512, 2)

    エンコーダ/デコーダの各ステップで 1/2 ダウン・アップサンプリング (×4)。
    ボトルネック (B, base_ch*8, 32) にノイズを加算して多様な信号を生成する。
    """

    def __init__(self, in_channels: int = 2, latent_dim: int = 64, base_ch: int = 64):
        super().__init__()

        # --- Encoder (512 → 256 → 128 → 64 → 32) ---
        self.enc1 = EncoderBlock(in_channels,  base_ch,     use_norm=False)  # 512→256
        self.enc2 = EncoderBlock(base_ch,      base_ch * 2)                  # 256→128
        self.enc3 = EncoderBlock(base_ch * 2,  base_ch * 4)                  # 128→64
        self.enc4 = EncoderBlock(base_ch * 4,  base_ch * 8)                  # 64→32 (bottleneck)

        # --- Noise injection: z → ボトルネック特徴マップへ加算 ---
        # base_ch*8 チャンネル分の補正値を各時刻に broadcast
        self.noise_proj = nn.Linear(latent_dim, base_ch * 8)
        self.noise_norm = nn.InstanceNorm1d(base_ch * 8, affine=True)

        # --- Decoder (32 → 64 → 128 → 256 → 512) ---
        # skip connection を cat してから次の block へ渡すため入力 ch を 2 倍にしている
        self.dec4 = DecoderBlock(base_ch * 8,      base_ch * 4, dropout=0.5)  # 32→64
        self.dec3 = DecoderBlock(base_ch * 4 * 2,  base_ch * 2, dropout=0.5)  # 64→128 (cat enc3)
        self.dec2 = DecoderBlock(base_ch * 2 * 2,  base_ch)                   # 128→256 (cat enc2)
        # 最終層: cat enc1 → 出力 ch = in_channels
        self.dec1 = nn.ConvTranspose1d(
            base_ch * 2, in_channels, kernel_size=4, stride=2, padding=1
        )                                                                        # 256→512

        self.out_act = nn.Tanh()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → (B, C, T)
        x = clean.permute(0, 2, 1)

        # Encode
        e1 = self.enc1(x)   # (B, 64,  256)
        e2 = self.enc2(e1)  # (B, 128, 128)
        e3 = self.enc3(e2)  # (B, 256, 64)
        e4 = self.enc4(e3)  # (B, 512, 32)

        # Noise injection: (B, latent_dim) → (B, base_ch*8, 1) → broadcast add
        nf = self.noise_proj(noise).unsqueeze(-1)   # (B, 512, 1)
        nf = self.noise_norm(nf.expand_as(e4))      # (B, 512, 32)
        e4 = e4 + nf

        # Decode with skip connections
        d4 = self.dec4(e4)                   # (B, 256, 64)
        d4 = torch.cat([d4, e3], dim=1)      # (B, 512, 64)

        d3 = self.dec3(d4)                   # (B, 128, 128)
        d3 = torch.cat([d3, e2], dim=1)      # (B, 256, 128)

        d2 = self.dec2(d3)                   # (B, 64,  256)
        d2 = torch.cat([d2, e1], dim=1)      # (B, 128, 256)

        d1 = self.dec1(d2)                   # (B, 2,   512)
        out = self.out_act(d1)

        # (B, C, T) → (B, T, C)
        return out.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Critic  (PatchGAN 1D, WGAN-GP 用: sigmoid なし)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    PatchGAN 形式の Critic。
    干渉あり信号とクリーン信号を channel 方向に連結して入力し、
    各パッチが本物か偽物かを示すスコアを出力する (WGAN-GP なので sigmoid なし)。

    入力:
        interference : (B, T, C) = (B, 512, 2)
        clean        : (B, T, C) = (B, 512, 2)
    出力:
        scores : (B, 1, L')  (PatchGAN スコア, L'=32)
    """

    def __init__(self, in_channels: int = 2, base_ch: int = 64):
        super().__init__()

        # concat(interference, clean) → (B, in_ch*2, 512)
        self.net = nn.Sequential(
            # Layer 1: InstanceNorm なし
            nn.Conv1d(in_channels * 2, base_ch,     4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv1d(base_ch,     base_ch * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv1d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(base_ch * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv1d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(base_ch * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (B, 1, 32)
            nn.Conv1d(base_ch * 8, 1, kernel_size=3, stride=1, padding=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, interference: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        x_i = interference.permute(0, 2, 1)   # (B, C, T)
        x_c = clean.permute(0, 2, 1)
        x = torch.cat([x_i, x_c], dim=1)      # (B, 2C, T)
        return self.net(x)                     # (B, 1, 32)


# ---------------------------------------------------------------------------
# Gradient Penalty (WGAN-GP)
# ---------------------------------------------------------------------------

def compute_gradient_penalty(
    critic: Critic,
    clean: torch.Tensor,
    real_interf: torch.Tensor,
    fake_interf: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP の勾配ペナルティを計算する。"""
    B = real_interf.size(0)
    alpha = torch.rand(B, 1, 1, device=device)

    # 実サンプルと偽サンプルの線形補間
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

def build_models() -> tuple[Generator, Critic]:
    """設定に基づいて Generator と Critic を構築して返す。"""
    G = Generator(
        in_channels=config.PREPROCESS_CONFIG["num_features"],
        latent_dim=config.MODEL_CONFIG["latent_dim"],
        base_ch=config.MODEL_CONFIG["base_channels"],
    )
    D = Critic(
        in_channels=config.PREPROCESS_CONFIG["num_features"],
        base_ch=config.MODEL_CONFIG["base_channels"],
    )
    return G, D
