# experiment2_calibration_ablation.py  (v3 CORRECT)
# ===================================================
# Tests 4 calibration image types to show which produces the correct
# INT8 step size for KL730 NPU deployment.
#
# The KL730 toolchain sets INT8 step = 2 * max_activation / 255.
# If calibration images produce the wrong (too small) activation range,
# the derived step is wrong and real golf inference peaks overflow the
# INT8 range, causing clipping and near-zero heatmap outputs.
#
# GOLF frames produce the correct activation range because the final
# conv layer activates strongly (peak ~1.0) only on real human bodies
# in the expected pose. Random noise gives ~0.2 max activation,
# making the step ~5x too small.
#
# ─────────────────────────────────────────────────────────────────
# SETUP  (edit the CONFIG section below before running)
# ─────────────────────────────────────────────────────────────────
# 1. Install dependencies:
#      pip install onnxruntime opencv-python matplotlib pandas numpy
#
# 2. Set HRNET to the path of your exported ONNX model:
#      hrnet_golf_v2_scaled.onnx  (the x1000-scaled version)
#
# 3. Set GOLF_DIR to a directory with .mp4 golf swing videos
#    (searched recursively). Face-on swing videos work best.
#
# 4. Set OUT_DIR to where you want results saved.
#
# 5. Run:
#      conda activate mmpose
#      python experiment2_calibration_ablation.py
#
#    Outputs saved to OUT_DIR:
#      calibration_comparison.png   <- Figure for paper (step ratio + activation)
#      calibration_results.csv      <- Full numeric results table
# ─────────────────────────────────────────────────────────────────

import os, sys, glob, random
import numpy as np
import cv2
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — set these three paths before running
# ══════════════════════════════════════════════════════════════════════════

HRNET    = ""   # e.g. "/home/user/models/hrnet_golf_v2_scaled.onnx"
GOLF_DIR = ""   # e.g. "/home/user/videos"  (folder with .mp4 swing videos)
OUT_DIR  = ""   # e.g. "/home/user/results/exp2"

# ══════════════════════════════════════════════════════════════════════════

if not HRNET or not GOLF_DIR or not OUT_DIR:
    raise ValueError(
        "Please set HRNET, GOLF_DIR, and OUT_DIR in the CONFIG section above."
    )

BAKED  = 1000.0   # x1000 scale factor baked into the ONNX model weights
N_CALIB = 50
N_TEST  = 30
IH, IW  = 256, 192
os.makedirs(OUT_DIR, exist_ok=True)


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    nrm = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return nrm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def load_golf(d, n):
    imgs = []
    vids = [v for v in glob.glob(os.path.join(d, '**', '*.mp4'), recursive=True)
            if '_h264' not in v and '_slow' not in v]
    for vid in vids:
        if len(imgs) >= n: break
        cap = cv2.VideoCapture(vid)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot < 3: cap.release(); continue
        for fi in np.linspace(0, tot - 1, min(8, tot)).astype(int):
            if len(imgs) >= n: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if ok: imgs.append(cv2.resize(fr, (IW, IH)))
        cap.release()
    # Pad with augmented copies if not enough unique frames
    while len(imgs) < n and imgs:
        b = random.choice(imgs).copy()
        imgs.append(
            np.clip(b.astype(float) * np.random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
        )
    return imgs[:n]


def make_conditions(golf_all, n_calib):
    np.random.seed(42); random.seed(42)

    # Type 1: pure random noise — no structure
    noise = [np.random.randint(0, 256, (IH, IW, 3), dtype=np.uint8)
             for _ in range(n_calib)]

    # Type 2: uniform grey at varying brightness — zero variance
    grey = [np.full((IH, IW, 3), int(255 * i / max(n_calib - 1, 1)), dtype=np.uint8)
            for i in range(n_calib)]

    # Type 3: ImageNet-style gradients + blobs — natural but no humans
    inet = []
    for _ in range(n_calib):
        img = np.zeros((IH, IW, 3), dtype=np.uint8)
        c1  = np.random.randint(80, 200, 3).astype(float)
        c2  = np.random.randint(80, 200, 3).astype(float)
        for y in range(IH):
            a = y / IH
            img[y] = np.clip(c1 * (1 - a) + c2 * a, 0, 255).astype(np.uint8)
        for _ in range(np.random.randint(2, 6)):
            cx, cy = np.random.randint(20, IW - 20), np.random.randint(20, IH - 20)
            cv2.circle(img, (cx, cy), np.random.randint(10, 40),
                       np.random.randint(30, 220, 3).tolist(), -1)
        inet.append(img)

    # Type 4: real golf face-on frames — correct domain
    golf = golf_all[:n_calib]

    return [
        (noise, "Random noise\n(np.random.randint)"),
        (grey,  "Uniform grey\n(constant input)"),
        (inet,  "ImageNet-style\n(natural, no humans)"),
        (golf,  "Golf face-on\n(correct domain \u2713)"),
    ]


def run_condition(sess, in_nm, out_nm, calib_imgs, test_tensors, label):
    short = label.split('\n')[0]
    print(f"\n  [{short}]")

    # Run calibration images through the model to get their activation range
    outs = []
    for img in calib_imgs:
        t   = to_tensor(img)
        out = sess.run([out_nm], {in_nm: t})[0] / BAKED
        outs.append(out)
    stack      = np.concatenate(outs, axis=0)
    c_min      = float(stack.min())
    c_max      = float(stack.max())
    abs_max    = max(abs(c_min), abs(c_max))
    calib_step = (2.0 * abs_max) / 255.0   # INT8 step derived from this calibration set

    print(f"    Calib activation range: [{c_min:.4f}, {c_max:.4f}]")
    print(f"    Derived INT8 step:       {calib_step:.6f}")

    # Run held-out golf test images to get the TRUE (correct) activation range
    test_outs = []
    for t in test_tensors:
        out = sess.run([out_nm], {in_nm: t})[0] / BAKED
        test_outs.append(out)
    test_stack   = np.concatenate(test_outs, axis=0)
    t_min        = float(test_stack.min())
    t_max        = float(test_stack.max())
    true_abs_max = max(abs(t_min), abs(t_max))
    true_step    = (2.0 * true_abs_max) / 255.0   # correct step for golf inference

    # Step ratio: 1.0 = perfect, <1.0 = step too small -> peaks overflow at inference
    step_ratio = calib_step / (true_step + 1e-9)

    # Simulate quantization with calibration-derived step and measure peak preservation
    peak_pcts, snrs, displs = [], [], []
    for out_fp in test_outs:
        q     = np.clip(np.round(out_fp / calib_step), -128, 127).astype(np.int8)
        out_q = q.astype(np.float32) * calib_step
        n_kpts  = out_fp.shape[1]
        matched = 0
        for k in range(n_kpts):
            a, b = out_fp[0, k], out_q[0, k]
            pa   = np.unravel_index(np.argmax(a), a.shape)
            pb   = np.unravel_index(np.argmax(b), b.shape)
            d    = float(np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2))
            displs.append(d)
            if d < 2.0: matched += 1
            flat = b.flatten()
            pk   = float(flat.max())
            bg   = float(np.sort(flat)[:-3].mean()) if len(flat) > 3 else 1e-8
            snrs.append(pk / (abs(bg) + 1e-8))
        peak_pcts.append(100.0 * matched / n_kpts)

    result = {
        'condition':            label.replace('\n', ' '),
        'calib_range':          f"[{c_min:.3f},{c_max:.3f}]",
        'true_golf_range':      f"[{t_min:.3f},{t_max:.3f}]",
        'calib_max_activation': round(abs_max, 3),
        'calib_int8_step':      round(calib_step, 6),
        'true_int8_step':       round(true_step, 6),
        'true_max_activation':  round(true_abs_max, 3),
        'step_ratio':           round(step_ratio, 3),
        'peak_preservation':    round(float(np.mean(peak_pcts)), 1),
        'mean_snr':             round(float(np.mean(snrs)), 1),
        'mean_displacement':    round(float(np.mean(displs)), 2),
        'step_match':           'CORRECT' if abs(step_ratio - 1.0) < 0.2 else 'WRONG',
    }
    print(f"    True golf step:          {true_step:.6f}")
    print(f"    Step ratio (calib/true): {step_ratio:.3f}  \u2192 {result['step_match']}")
    print(f"    Peak preservation:       {result['peak_preservation']:.1f}%")
    print(f"    Mean displacement:       {result['mean_displacement']:.2f} cells")
    return result


def main():
    print("=" * 65)
    print("Experiment 2: Calibration Dataset Quality Ablation")
    print("=" * 65)
    print("\nKey metric: does calibration data produce the CORRECT INT8 step?")
    print("Step ratio (calib/true) near 1.0 = correct calibration\n")

    if not os.path.exists(HRNET):
        print(f"ERROR: model not found at: {HRNET}")
        sys.exit(1)

    sess   = ort.InferenceSession(HRNET, providers=['CPUExecutionProvider'])
    in_nm  = sess.get_inputs()[0].name
    out_nm = sess.get_outputs()[0].name
    print(f"Model: {os.path.basename(HRNET)}")

    print("\nLoading golf frames...")
    all_golf = load_golf(GOLF_DIR, N_CALIB + N_TEST)
    print(f"  {len(all_golf)} frames total")
    test_golf    = all_golf[N_CALIB:N_CALIB + N_TEST]
    test_tensors = [to_tensor(img) for img in test_golf]
    print(f"  Calibration pool: {N_CALIB}")
    print(f"  Test set (held-out): {len(test_tensors)}")

    conditions   = make_conditions(all_golf, N_CALIB)
    results      = []
    for imgs, label in conditions:
        r = run_condition(sess, in_nm, out_nm, imgs, test_tensors, label)
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'calibration_results.csv'), index=False)

    # ── Figure: Calibration Domain Sensitivity ────────────────────────────
    short = ['Random\nnoise', 'Uniform\ngrey', 'ImageNet\nstyle', 'Golf frames\n(ours) \u2605']
    col   = ['#E24B4A', '#888780', '#BA7517', '#185FA5']
    oi    = 3   # Golf frames = correct domain, index 3

    step_ratios     = [r['step_ratio']           for r in results]
    max_activations = [r['calib_max_activation']  for r in results]
    true_golf_max   = results[oi]['true_max_activation']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Figure 8 \u2014 Experiment 3: Calibration Domain Sensitivity\n'
        'INT8 step ratio and peak preservation across four calibration image types.',
        fontsize=11, fontweight='bold'
    )

    # ── Left: Step ratio ─────────────────────────────────────────────────
    ax1.set_title('Calibration Step Ratio\n(calib step / true step, ideal = 1.0)', fontsize=10)
    bars1 = ax1.bar(short, step_ratios, color=col, edgecolor='white', linewidth=0.5)
    bars1[oi].set_edgecolor('#185FA5'); bars1[oi].set_linewidth(2.5)

    ax1.axhline(1.0, color='green',   linestyle='-',  lw=2.0, label='Correct (1.0\u00d7)')
    ax1.axhline(0.8, color='#DAA520', linestyle='--', lw=1.2, label='\u00b120% tolerance')
    ax1.axhline(1.2, color='#DAA520', linestyle='--', lw=1.2)

    ax1.set_ylabel('Step ratio', fontsize=10)
    ax1.set_ylim(0, 1.35)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=9)

    for bar, v, i in zip(bars1, step_ratios, range(4)):
        is_correct = abs(v - 1.0) < 0.2
        txt_color  = '#185FA5' if i == oi else ('#E24B4A' if not is_correct else 'black')
        mark       = '\u2713' if is_correct else '\u2717'
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f'{v:.3f}\u00d7\n{mark}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color=txt_color)

    # ── Right: Max activation range ───────────────────────────────────────
    ax2.set_title('Calibration Activation Range\n'
                  '(max activation observed, must match inference)', fontsize=10)
    bars2 = ax2.bar(short, max_activations, color=col, edgecolor='white', linewidth=0.5)
    bars2[oi].set_edgecolor('#185FA5'); bars2[oi].set_linewidth(2.5)

    ax2.axhline(true_golf_max, color='green', linestyle='--', lw=2.0,
                label=f'True golf max (~{true_golf_max:.2f})')
    ax2.set_ylabel('Max activation value', fontsize=10)
    ax2.set_ylim(0, true_golf_max * 1.35)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', labelsize=9)

    for bar, v in zip(bars2, max_activations):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + true_golf_max * 0.02,
                 f'{v:.3f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, 'calibration_comparison.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {p1}")

    # ── Paper table ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE VI \u2014 Calibration Ablation (copy into paper)")
    print("=" * 80)
    print(f"  {'Condition':<28} {'Calib range':>15} {'Calib step':>11} "
          f"{'True step':>10} {'Ratio':>6} {'Peak%':>7} {'Step?':>8}")
    print("  " + "\u2500" * 88)
    for r in results:
        mk = " \u2190" if 'Golf' in r['condition'] else "  "
        print(f"  {r['condition'][:26]:<28} {r['calib_range']:>15} "
              f"{r['calib_int8_step']:>11.6f} "
              f"{r['true_int8_step']:>10.6f} "
              f"{r['step_ratio']:>6.3f} "
              f"{r['peak_preservation']:>6.1f}% "
              f"{r['step_match']:>8}{mk}")
    print(f"\nOutputs: {OUT_DIR}")
    return df


if __name__ == '__main__':
    main()
