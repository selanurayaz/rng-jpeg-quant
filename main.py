import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import Counter

# =========================================================
# 1) RSÜ / PRNG: XORSHIFT32 (Educational Purpose)
# =========================================================
class XorShift32:
    def __init__(self, seed: int = 2463534242):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def rand_int(self, low: int, high: int) -> int:
        # inclusive low, inclusive high
        r = self.next_u32()
        return low + (r % (high - low + 1))

    def rand_float01(self) -> float:
        return self.next_u32() / 0xFFFFFFFF


# =========================================================
# 2) RSÜ Output Helpers (bit/byte stream)
# =========================================================
def rng_bytes(rng: XorShift32, n_bytes: int) -> bytes:
    out = bytearray()
    for _ in range(n_bytes):
        out.append(rng.rand_int(0, 255))
    return bytes(out)

def bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for b in data:
        for i in range(7, -1, -1):  # MSB -> LSB
            bits.append((b >> i) & 1)
    return bits


# =========================================================
# 3) Statistical Tests (Educational)
# =========================================================
def monobit_test(bits: list[int]) -> dict:
    n = len(bits)
    ones = sum(bits)
    zeros = n - ones
    return {
        "n_bits": n,
        "zeros": zeros,
        "ones": ones,
        "p_zeros": zeros / n if n else 0.0,
        "p_ones": ones / n if n else 0.0,
        "diff": abs(ones - zeros),
    }

def chisquare_test_bytes(data: bytes) -> dict:
    """
    Chi-square test over byte frequencies (0..255).
    This is a basic uniformity check (educational).
    """
    n = len(data)
    expected = n / 256.0 if n else 0.0
    counts = Counter(data)

    chi2 = 0.0
    if expected > 0:
        for v in range(256):
            obs = counts.get(v, 0)
            chi2 += ((obs - expected) ** 2) / expected

    return {
        "n_bytes": n,
        "expected_per_value": expected,
        "chi2": chi2,
        "df": 255,
    }

def runs_test(bits: list[int]) -> dict:
    """
    Runs (seri) testi: 0/1 değişim sayısı.
    """
    n = len(bits)
    if n == 0:
        return {"n_bits": 0, "runs": 0, "zeros": 0, "ones": 0}

    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            runs += 1

    ones = sum(bits)
    zeros = n - ones
    return {"n_bits": n, "runs": runs, "zeros": zeros, "ones": ones}

def run_rsu_tests(seed: int, n_bytes: int = 200_000) -> dict:
    rng = XorShift32(seed)
    data = rng_bytes(rng, n_bytes)
    bits = bytes_to_bits(data)

    mono = monobit_test(bits)
    chi = chisquare_test_bytes(data)
    run = runs_test(bits)

    return {
        "seed": seed,
        "n_bytes": n_bytes,
        "monobit": mono,
        "chisquare": chi,
        "runs": run,
    }


# =========================================================
# 4) JPEG Quantization Experiment (Reference Table)
# =========================================================
QSTD_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.int32)

def ensure_dirs() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("images", exist_ok=True)

def file_kb(path: str) -> float:
    return os.path.getsize(path) / 1024.0

def save_jpeg_opencv(img_bgr: np.ndarray, out_path: str, quality: int) -> None:
    cv2.imwrite(out_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

def generate_quant_table_from_rng(seed: int, alpha: float = 0.15) -> np.ndarray:
    """
    Create an 8x8 'quantization-like' table by perturbing QSTD_LUMA.
    (OpenCV doesn't allow custom quant tables directly, so we visualize & map it.)
    """
    rng = XorShift32(seed)
    q = QSTD_LUMA.astype(np.float32).copy()

    noise = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            r = (rng.rand_float01() * 2.0) - 1.0
            noise[i, j] = r

    q_new = q * (1.0 + alpha * noise)
    q_new = np.clip(np.round(q_new), 1, 255).astype(np.int32)

    # soft increase with frequency
    for i in range(8):
        for j in range(8):
            freq_weight = 1 + (i + j) * 0.03
            q_new[i, j] = int(np.clip(round(q_new[i, j] * freq_weight), 1, 255))

    return q_new

def plot_and_save_quant_tables(q_std: np.ndarray, q_new: np.ndarray) -> None:
    plt.figure()
    plt.title("Standard JPEG Quantization Table (Luma)")
    plt.imshow(q_std, interpolation="nearest")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/quant_table_std.png", dpi=160)
    plt.close()

    plt.figure()
    plt.title("RNG-derived Quantization-like Table")
    plt.imshow(q_new, interpolation="nearest")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/quant_table_rng.png", dpi=160)
    plt.close()


# =========================================================
# 5) Simple XOR "Encryption" Demo (Educational)
# =========================================================
def xor_encrypt_text(text: str, seed: int) -> bytes:
    rng = XorShift32(seed)
    data = text.encode("utf-8")
    out = bytearray()
    for b in data:
        out.append(b ^ rng.rand_int(0, 255))
    return bytes(out)

def xor_apply_keystream(data: bytes, seed: int) -> bytes:
    rng = XorShift32(seed)
    out = bytearray()
    for b in data:
        out.append(b ^ rng.rand_int(0, 255))
    return bytes(out)


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()

    # ---- Settings ----
    seed = 123456
    alpha = 0.15
    base_quality = 50
    rsu_test_bytes = 200_000  # 200k bytes ~ 1.6M bits
    # ------------------

    # =========================
    # RSÜ Statistical Tests
    # =========================
    rsu = run_rsu_tests(seed=seed, n_bytes=rsu_test_bytes)
    mono = rsu["monobit"]
    chi = rsu["chisquare"]
    run = rsu["runs"]

    print("\n=== RSÜ Statistical Tests (Educational) ===")
    print(f"Seed={seed} | Bytes={rsu['n_bytes']} | Bits={mono['n_bits']}")
    print(f"Monobit: zeros={mono['zeros']} ones={mono['ones']} p1={mono['p_ones']:.4f} p0={mono['p_zeros']:.4f}")
    print(f"Chi-square (bytes 0..255): chi2={chi['chi2']:.2f} df={chi['df']}")
    print(f"Runs: {run['runs']} (zeros={run['zeros']}, ones={run['ones']})")

    with open("results/rsu_tests.txt", "w", encoding="utf-8") as f:
        f.write("RSÜ Statistical Tests (Educational)\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Generated bytes: {rsu['n_bytes']}\n")
        f.write(f"Generated bits: {mono['n_bits']}\n\n")
        f.write("Monobit Test:\n")
        f.write(f"  zeros={mono['zeros']} ones={mono['ones']}\n")
        f.write(f"  p(1)={mono['p_ones']:.6f} p(0)={mono['p_zeros']:.6f}\n")
        f.write(f"  diff={mono['diff']}\n\n")
        f.write("Chi-square Test (0..255 byte frequencies):\n")
        f.write(f"  chi2={chi['chi2']:.6f} df={chi['df']}\n")
        f.write(f"  expected per value={chi['expected_per_value']:.3f}\n\n")
        f.write("Runs Test:\n")
        f.write(f"  runs={run['runs']} zeros={run['zeros']} ones={run['ones']}\n")

    # =========================
    # JPEG Quantization Experiment
    # =========================
    image_paths = sorted(glob.glob("images/*.jpg") + glob.glob("images/*.png") + glob.glob("images/*.jpeg"))
    if not image_paths:
        raise SystemExit("No images found in ./images. Put at least one image (e.g., images/test1.jpg).")

    q_new = generate_quant_table_from_rng(seed, alpha=alpha)

    std_mean = float(QSTD_LUMA.mean())
    new_mean = float(q_new.mean())
    ratio = new_mean / std_mean if std_mean != 0 else 1.0
    quality_rng = int(np.clip(round(base_quality / ratio), 5, 95))

    plot_and_save_quant_tables(QSTD_LUMA, q_new)

    rows = []
    for p in image_paths:
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        name = os.path.splitext(os.path.basename(p))[0]
        out_std = f"results/{name}_std_q{base_quality}.jpg"
        out_rng = f"results/{name}_rng_q{quality_rng}.jpg"

        save_jpeg_opencv(img_bgr, out_std, base_quality)
        save_jpeg_opencv(img_bgr, out_rng, quality_rng)

        rec_std = cv2.imread(out_std, cv2.IMREAD_COLOR)
        rec_rng = cv2.imread(out_rng, cv2.IMREAD_COLOR)

        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        std_rgb = cv2.cvtColor(rec_std, cv2.COLOR_BGR2RGB)
        rng_rgb = cv2.cvtColor(rec_rng, cv2.COLOR_BGR2RGB)

        psnr_std = psnr(orig_rgb, std_rgb, data_range=255)
        psnr_rng = psnr(orig_rgb, rng_rgb, data_range=255)

        rows.append({
            "image": name,
            "std_quality": base_quality,
            "rng_quality": quality_rng,
            "std_kb": file_kb(out_std),
            "rng_kb": file_kb(out_rng),
            "std_psnr": psnr_std,
            "rng_psnr": psnr_rng,
        })

    print("\n=== JPEG Experiment Summary ===")
    print(f"Seed={seed}, alpha={alpha}")
    print(f"Base JPEG quality={base_quality}")
    print(f"RNG-derived table mean ratio={ratio:.3f} => mapped rng_quality={quality_rng}")
    print("\nImage | StdKB | RngKB | StdPSNR | RngPSNR")
    print("----- | ----- | ----- | ------- | -------")
    for r in rows:
        print(f"{r['image']} | {r['std_kb']:.1f} | {r['rng_kb']:.1f} | {r['std_psnr']:.2f} | {r['rng_psnr']:.2f}")

    with open("results/summary.md", "w", encoding="utf-8") as f:
        f.write("# Results Summary\n\n")
        f.write(f"- Seed: `{seed}`\n- alpha: `{alpha}`\n")
        f.write(f"- Base quality (standard JPEG): `{base_quality}`\n")
        f.write(f"- Mapped quality (RNG-derived): `{quality_rng}`\n")
        f.write(f"- Mean ratio (q_new/q_std): `{ratio:.3f}`\n\n")
        f.write("| Image | Std Quality | RNG Quality | Std Size (KB) | RNG Size (KB) | Std PSNR | RNG PSNR |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['image']} | {r['std_quality']} | {r['rng_quality']} | {r['std_kb']:.1f} | {r['rng_kb']:.1f} | {r['std_psnr']:.2f} | {r['rng_psnr']:.2f} |\n")

    print("\nSaved: results/summary.md, results/rsu_tests.txt and outputs in results/")

    # =========================
    # Simple Encryption Demo
    # =========================
    demo_text = "StudentID:123456"
    encrypted = xor_encrypt_text(demo_text, seed)
    decrypted = xor_apply_keystream(encrypted, seed).decode("utf-8")

    print("\n=== Simple Encryption Demo (Educational) ===")
    print("Plaintext       :", demo_text)
    print("Encrypted (hex) :", encrypted.hex())
    print("Decrypted       :", decrypted)
    print("NOTE: XOR demo is NOT cryptographically secure. Included for educational purposes only.")


if __name__ == "__main__":
    main()
