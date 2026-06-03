"""WGAN-GP による 2D 条件付き GAN の学習スクリプト (train_2d.py)

train.py との違い:
    - model_2d.py の Generator2D / Critic2D を使用
    - data_loader_2d.py の get_dataloaders_2d() を使用 (ブロック単位バッチ)
    - compute_gradient_penalty_2d() を使用 (alpha shape が 4 次元)
    - config.py はそのまま共用
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
import data_loader_2d
import model_2d

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    # --- wandb の初期化 ---
    use_wandb = config.TRAIN_CONFIG["use_wandb"] and _WANDB
    if use_wandb:
        wandb.init(
            project="Radar-GAN-DataAugmentation-2D",
            name=f"run2d_{config.DT_NOW:%Y%m%d_%H%M%S}",
            config={
                "model":         "2D",
                "lr_g":          config.TRAIN_CONFIG["lr_g"],
                "lr_d":          config.TRAIN_CONFIG["lr_d"],
                "batch_size":    config.TRAIN_CONFIG["batch_size"],
                "epochs":        config.TRAIN_CONFIG["epochs"],
                "n_critic":      config.TRAIN_CONFIG["n_critic"],
                "lambda_gp":     config.TRAIN_CONFIG["lambda_gp"],
                "lambda_l1":     config.TRAIN_CONFIG["lambda_l1"],
                "latent_dim":    config.MODEL_CONFIG["latent_dim"],
                "base_channels": config.MODEL_CONFIG["base_channels"],
            },
        )
    elif config.TRAIN_CONFIG["use_wandb"] and not _WANDB:
        print("wandb がインストールされていません。ログなしで学習します。")

    # --- データローダー (ブロック単位: (B, chirps, seq_len, 2)) ---
    train_loader, valid_loader = data_loader_2d.get_dataloaders_2d(
        batch_size=config.TRAIN_CONFIG["batch_size"],
        validation_split=config.TRAIN_CONFIG["validation_split"],
    )

    # --- モデルの構築 ---
    G, D = model_2d.build_models_2d()
    G = G.to(device)
    D = D.to(device)

    # --- オプティマイザ (Adam + TTUR) ---
    opt_G = optim.Adam(G.parameters(), lr=config.TRAIN_CONFIG["lr_g"], betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=config.TRAIN_CONFIG["lr_d"], betas=(0.5, 0.9))

    latent_dim = config.MODEL_CONFIG["latent_dim"]
    lambda_gp  = config.TRAIN_CONFIG["lambda_gp"]
    lambda_l1  = config.TRAIN_CONFIG["lambda_l1"]
    n_critic   = config.TRAIN_CONFIG["n_critic"]

    # 保存先は 1D モデルと区別するため "_2D" サフィックスを付加
    save_path = config.TRAIN_CONFIG["model_save_path"] + "_2D"
    os.makedirs(save_path, exist_ok=True)

    print(f"\n学習開始 (WGAN-GP 2D)  エポック数: {config.TRAIN_CONFIG['epochs']}\n")

    for epoch in range(config.TRAIN_CONFIG["epochs"]):
        G.train()
        D.train()

        total_d_loss = 0.0
        total_g_loss = 0.0
        n_g_steps    = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:>3}/{config.TRAIN_CONFIG['epochs']}",
        )

        for clean, real_interf in pbar:
            clean       = clean.to(device)        # (B, chirps, seq_len, 2)
            real_interf = real_interf.to(device)  # (B, chirps, seq_len, 2)
            B = clean.size(0)

            # ============================================================
            # Step 1: Critic を n_critic 回更新
            # ============================================================
            for _ in range(n_critic):
                z = torch.randn(B, latent_dim, device=device)

                with torch.no_grad():
                    fake_interf = G(clean, z)

                score_real = D(real_interf, clean)
                score_fake = D(fake_interf, clean)
                gp = model_2d.compute_gradient_penalty_2d(
                    D, clean, real_interf, fake_interf, device
                )

                d_loss = score_fake.mean() - score_real.mean() + lambda_gp * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

                total_d_loss += d_loss.item()

            # ============================================================
            # Step 2: Generator を 1 回更新
            # ============================================================
            z = torch.randn(B, latent_dim, device=device)
            fake_interf = G(clean, z)

            score_fake = D(fake_interf, clean)
            g_loss = -score_fake.mean()

            if lambda_l1 > 0.0:
                g_loss = g_loss + lambda_l1 * nn.L1Loss()(fake_interf, real_interf)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            total_g_loss += g_loss.item()
            n_g_steps    += 1

            pbar.set_postfix(
                D=f"{d_loss.item():.4f}",
                G=f"{g_loss.item():.4f}",
            )

        avg_d = total_d_loss / (len(train_loader) * n_critic)
        avg_g = total_g_loss / n_g_steps

        # --- 検証: Generator の生成信号と実信号の L1 距離 ---
        G.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for clean_v, real_interf_v in valid_loader:
                clean_v       = clean_v.to(device)
                real_interf_v = real_interf_v.to(device)
                z_v = torch.randn(clean_v.size(0), latent_dim, device=device)
                fake_v = G(clean_v, z_v)
                val_l1 += nn.L1Loss()(fake_v, real_interf_v).item()
        avg_val_l1 = val_l1 / len(valid_loader)

        print(
            f"Epoch {epoch + 1:>3}: "
            f"D={avg_d:.6f}  G={avg_g:.6f}  Val L1={avg_val_l1:.6f}"
        )

        if use_wandb:
            wandb.log({
                "epoch":  epoch + 1,
                "d_loss": avg_d,
                "g_loss": avg_g,
                "val_l1": avg_val_l1,
            })

        # --- チェックポイント保存 ---
        if (epoch + 1) % config.TRAIN_CONFIG["save_interval"] == 0:
            ckpt = os.path.join(save_path, f"checkpoint_epoch{epoch + 1:04d}.pth")
            torch.save(
                {
                    "epoch":            epoch + 1,
                    "G_state_dict":     G.state_dict(),
                    "D_state_dict":     D.state_dict(),
                    "opt_G_state_dict": opt_G.state_dict(),
                    "opt_D_state_dict": opt_D.state_dict(),
                },
                ckpt,
            )
            print(f"  チェックポイントを保存: {ckpt}")

    # --- 最終モデルを保存 ---
    torch.save(G.state_dict(), os.path.join(save_path, "G_final.pth"))
    torch.save(D.state_dict(), os.path.join(save_path, "D_final.pth"))
    print(f"\n学習完了。保存先: {save_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
