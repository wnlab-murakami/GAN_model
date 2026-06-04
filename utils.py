import numpy as np


def load_data_from_txt(filepath: str) -> np.ndarray:
    """テキストファイルを読み込み NumPy 配列として返す。"""
    with open(filepath, "r") as f:
        lines = f.readlines()
    data = [list(map(float, line.strip().split())) for line in lines if line.strip()]
    return np.array(data, dtype=np.float32)


def max_abs_normalize_complex_channels(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    複素数データ (2 チャンネル: 実部・虚部) をサンプルごとに最大絶対値で正規化する。

    Args:
        data: shape (..., sequence_length, 2)
    Returns:
        normalized: shape (..., sequence_length, 2)
        max_abs   : shape (..., 1, 1)  逆正規化に使用
    """
    real = data[..., 0]
    imag = data[..., 1]
    abs_val = np.sqrt(real ** 2 + imag ** 2)
    max_abs = np.max(abs_val, axis=-1, keepdims=True)   # (..., 1)
    max_abs = np.expand_dims(max_abs, axis=-1)           # (..., 1, 1)
    normalized = data / max_abs
    return normalized, max_abs


def max_abs_denormalize_complex_channels(
    normalized: np.ndarray, max_abs: np.ndarray
) -> np.ndarray:
    """最大絶対値正規化の逆変換。"""
    return normalized * max_abs

# ---------------------------------------------------------------------------
# レンジFFT / IFFT
# ---------------------------------------------------------------------------
 
def apply_range_fft(data: np.ndarray) -> np.ndarray:
    """
    時間領域 IQ 信号にレンジ FFT (fast-time 方向) を適用してレンジ-チャープマップを返す。
 
    入力の実部・虚部を複素数として再結合し、sequence_length 軸方向に FFT をかける。
    出力は FFT 結果の実部・虚部を再分割した形で返すため、shape は変化しない。
 
    Args:
        data: shape (..., sequence_length, 2)  — [..., 0]=実部, [..., 1]=虚部
    Returns:
        range_map: shape (..., sequence_length, 2)  — FFT 後の実部・虚部
    """
    real = data[..., 0]
    imag = data[..., 1]
    # 複素信号として結合し sequence_length 軸 (axis=-1) に FFT
    iq = real + 1j * imag                              # (..., sequence_length)
    fft_out = np.fft.fft(iq, axis=-1)                  # (..., sequence_length)
    range_map = np.stack(
        [fft_out.real, fft_out.imag], axis=-1
    ).astype(np.float32)                               # (..., sequence_length, 2)
    return range_map
 
 
def apply_range_ifft(data: np.ndarray) -> np.ndarray:
    """
    レンジ-チャープマップに逆 FFT を適用して時間領域 IQ 信号に戻す。
 
    apply_range_fft の逆変換。generate.py / generate_2d.py で
    時間領域の出力が必要なときに使用する。
 
    Args:
        data: shape (..., sequence_length, 2)  — FFT 後の実部・虚部
    Returns:
        iq_signal: shape (..., sequence_length, 2)  — 時間領域の実部・虚部
    """
    real = data[..., 0]
    imag = data[..., 1]
    fft_out = real + 1j * imag                         # (..., sequence_length)
    iq = np.fft.ifft(fft_out, axis=-1)                 # (..., sequence_length)
    iq_signal = np.stack(
        [iq.real, iq.imag], axis=-1
    ).astype(np.float32)                               # (..., sequence_length, 2)
    return iq_signal
 