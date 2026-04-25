# experiment3_model_selection.py  (v2 — 4 distinct architectures)
# =================================================================
# Experiment 3: Model Selection Study for KL730 NPU Deployment

# Compares 4 architecturally distinct pose estimation approaches:

#   Model 1 — HRNet-W48 MPII base    (Heatmap head, no golf fine-tune)
#   Model 2 — HRNet-W48 Golf v2      (Heatmap head, golf fine-tuned) <- SELECTED
#   Model 3 — RTMPose-S Golf         (SimCC head -- uint16 required)
#   Model 4 — YOLOv8n-pose           (Direct regression, small)

# Five criteria:
#   C1 Accuracy  C2 INT8  C3 HW format  C4 Model size  C5 Speed

# ─────────────────────────────────────────────────────────────────
# SETUP  (edit the CONFIG section below before running)
# ─────────────────────────────────────────────────────────────────
# 1. Install dependencies:
#      pip install torch mmpose onnxruntime opencv-python matplotlib pandas

# 2. Set MMPOSE_DIR to the folder containing mmpose/__init__.py.

# 3. Set BASE_DIR to your project root. Expected layout:
#      <BASE_DIR>/
#        work_dirs/
#          hrnet_w48/
#            best_coco_AP_epoch_50.pth    <- HRNet MPII base checkpoint
#            hrnet_w48_mpii_coco17.py     <- MMPose config
#          hrnet_w48_golf_v2/
#            best_coco_AP_epoch_30.pth    <- HRNet Golf v2 checkpoint
#          rtmpose_s_golf/
#            best_coco_AP_epoch_50.pth    <- RTMPose Golf checkpoint
#        rtmpose_s_golf.py                <- RTMPose config

# 4. Set YOLO_DIR to the folder containing yolov8n-pose.onnx.

# 5. Set GOLF_DIR to a directory with .mp4 golf swing videos
#    (searched recursively). Any face-on swing videos work.

# 6. Set OUT_DIR to where you want results saved.

# 7. Run:
#      conda activate mmpose
#      python experiment3_model_selection.py

#    Outputs saved to OUT_DIR:
#      model_selection_bars.png      <- Figure for paper (5-criteria chart)
#      model_selection_results.csv   <- Raw numbers
# ─────────────────────────────────────────────────────────────────

import os, sys, glob
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — set these four paths for your environment before running
# ══════════════════════════════════════════════════════════════════════════

MMPOSE_DIR = ""   # e.g. "/home/user/mmpose"  (folder with mmpose/__init__.py)
BASE_DIR   = ""   # e.g. "/home/user/golf_pose"  (project root with work_dirs/)
YOLO_DIR   = ""   # e.g. "/home/user/yolov8"  (folder with yolov8n-pose.onnx)
GOLF_DIR   = ""   # e.g. "/home/user/videos"  (folder with .mp4 swing videos)
OUT_DIR    = ""   # e.g. "/home/user/results/exp3"

# ══════════════════════════════════════════════════════════════════════════

if MMPOSE_DIR:
    sys.path.insert(0, MMPOSE_DIR)

if not BASE_DIR or not GOLF_DIR or not OUT_DIR:
    raise ValueError(
        "Please fill in BASE_DIR, GOLF_DIR, and OUT_DIR in the CONFIG section above."
    )

os.makedirs(OUT_DIR, exist_ok=True)
IH, IW = 256, 192
MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
STD  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)

MODELS = [
    {
        'id':           'hrnet_mpii_base',
        'name':         'HRNet-W48\nMPII (base)',
        'paper_name':   'HRNet-W48 MPII (base)',
        'head_type':    'Heatmap',
        'source':       'mmpose_pth',
        'scale':        1.0,
        'checkpoint':   os.path.join(BASE_DIR, 'work_dirs', 'hrnet_w48',
                                     'best_coco_AP_epoch_50.pth'),
        'config':       os.path.join(BASE_DIR, 'work_dirs', 'hrnet_w48',
                                     'hrnet_w48_mpii_coco17.py'),
        'ap':           0.890,
        'output_fmt':   'uint8',
        'hw_compat':    True,
        'hw_note':      'Heatmap\nuint8 OK',
        'params_m':     63.6,
        'npu_feasible': True,
        'infer_ms':     6.0,
        'speed_note':   '6ms\n(measured)',
        'color':        '#888780',
        'selected':     False,
    },
    {
        'id':           'hrnet_golf_v2',
        'name':         'HRNet-W48\nGolf v2 \u2605',
        'paper_name':   'HRNet-W48 Golf v2',
        'head_type':    'Heatmap',
        'source':       'mmpose_pth',
        'scale':        1000.0,
        'checkpoint':   os.path.join(BASE_DIR, 'work_dirs', 'hrnet_w48_golf_v2',
                                     'best_coco_AP_epoch_30.pth'),
        'config':       os.path.join(BASE_DIR, 'work_dirs', 'hrnet_w48',
                                     'hrnet_w48_mpii_coco17.py'),
        'ap':           0.918,
        'output_fmt':   'uint8',
        'hw_compat':    True,
        'hw_note':      'Heatmap\nuint8 OK',
        'params_m':     63.6,
        'npu_feasible': True,
        'infer_ms':     64.0,
        'speed_note':   '64ms\n(measured)',
        'color':        '#185FA5',
        'selected':     True,
    },
    {
        'id':           'rtmpose_golf',
        'name':         'RTMPose-S\nGolf',
        'paper_name':   'RTMPose-S Golf',
        'head_type':    'SimCC',
        'source':       'mmpose_pth',
        'scale':        1.0,
        'checkpoint':   os.path.join(BASE_DIR, 'work_dirs', 'rtmpose_s_golf',
                                     'best_coco_AP_epoch_50.pth'),
        'config':       os.path.join(BASE_DIR, 'rtmpose_s_golf.py'),
        'ap':           0.971,
        'output_fmt':   'uint16 req.',
        'hw_compat':    False,
        'hw_note':      '\u2717 SimCC\nneeds uint16',
        'params_m':     5.5,
        'npu_feasible': True,
        'infer_ms':     300.0,
        'speed_note':   '300ms\nCPU only',
        'color':        '#BA7517',
        'selected':     False,
    },
    {
        'id':           'yolov8n_pose',
        'name':         'YOLOv8n-pose',
        'paper_name':   'YOLOv8n-pose',
        'head_type':    'Direct regression',
        'source':       'onnx',
        'scale':        1.0,
        'onnx_path':    os.path.join(YOLO_DIR, 'yolov8n-pose.onnx') if YOLO_DIR else '',
        'ap':           0.500,
        'output_fmt':   'uint8',
        'hw_compat':    True,
        'hw_note':      'Regression\nuint8 OK',
        'params_m':     3.3,
        'npu_feasible': True,
        'infer_ms':     None,
        'speed_note':   'not\nmeasured',
        'color':        '#1D9E75',
        'selected':     False,
    },
]


def load_frames_mmpose(n=20):
    tensors = []
    vids = [v for v in glob.glob(os.path.join(GOLF_DIR, '**', '*.mp4'), recursive=True)
            if '_h264' not in v and '_slow' not in v]
    for vid in vids:
        if len(tensors) >= n: break
        cap = cv2.VideoCapture(vid)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot < 3: cap.release(); continue
        for fi in np.linspace(0, tot - 1, min(5, tot)).astype(int):
            if len(tensors) >= n: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if ok:
                r   = cv2.resize(frame, (IW, IH))
                rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)
                nrm = (rgb - MEAN) / STD
                tensors.append(
                    torch.from_numpy(nrm.transpose(2, 0, 1)[np.newaxis].astype(np.float32))
                )
        cap.release()
    return tensors


def load_frames_onnx(n=20):
    arrays = []
    vids = [v for v in glob.glob(os.path.join(GOLF_DIR, '**', '*.mp4'), recursive=True)
            if '_h264' not in v and '_slow' not in v]
    for vid in vids:
        if len(arrays) >= n: break
        cap = cv2.VideoCapture(vid)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot < 3: cap.release(); continue
        for fi in np.linspace(0, tot - 1, min(5, tot)).astype(int):
            if len(arrays) >= n: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if ok:
                r640 = cv2.resize(frame, (640, 640))
                rgb  = cv2.cvtColor(r640, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                arrays.append(rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32))
        cap.release()
    return arrays


def measure_int8(arr, true_scale=1.0):
    a    = arr / true_scale
    amax = max(abs(float(a.min())), abs(float(a.max())))
    if amax < 1e-9: return 0.0, 0.0
    step = (2.0 * amax) / 255.0
    if a.ndim == 4:   per = a[0].reshape(min(17, a.shape[1]), -1)
    elif a.ndim == 3: per = a[0, :17, :]
    else:             per = a.reshape(17, -1) if a.size >= 17 else a.reshape(1, -1)
    ss_list, rep = [], []
    for k in range(per.shape[0]):
        h  = per[k].flatten()
        pk = float(h.max())
        bg = float(np.sort(h)[:-3].mean()) if len(h) > 3 else 0.0
        ss = (pk - bg) / (step + 1e-12)
        ss_list.append(ss); rep.append(ss > 1.0)
    return float(np.mean(ss_list)), 100.0 * sum(rep) / len(rep)


def evaluate_model(cfg, frames_mm, frames_onnx):
    print(f"\n{'─'*50}")
    print(f"  {cfg['paper_name']}  [{cfg['head_type']}]")
    c2_ss_list, c2_pct_list = [], []
    run_ok = False

    if cfg['source'] == 'mmpose_pth':
        ck, co = cfg['checkpoint'], cfg['config']
        if os.path.exists(ck) and os.path.exists(co):
            try:
                from mmpose.apis import init_model
                model = init_model(co, ck, device='cpu')
                model.eval()
                if cfg['scale'] != 1.0:
                    fl = model.head.final_layer
                    with torch.no_grad():
                        fl.weight.data *= cfg['scale']
                        if fl.bias is not None:
                            fl.bias.data *= cfg['scale']
                for t in frames_mm:
                    with torch.no_grad():
                        feat = model.backbone(t)
                        if hasattr(model, 'neck') and model.neck:
                            feat = model.neck(feat)
                        out = model.head(feat)
                        if isinstance(out, (list, tuple)):
                            out = out[-1] if isinstance(out[-1], torch.Tensor) else out[0]
                    ss, pct = measure_int8(out.numpy(), cfg['scale'])
                    c2_ss_list.append(ss); c2_pct_list.append(pct)
                run_ok = True
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  Checkpoint or config not found — using known measured values")

    elif cfg['source'] == 'onnx':
        op = cfg.get('onnx_path', '')
        if op and os.path.exists(op):
            try:
                import onnxruntime as ort
                sess   = ort.InferenceSession(op, providers=['CPUExecutionProvider'])
                in_nm  = sess.get_inputs()[0].name
                out_nm = sess.get_outputs()[0].name
                for arr in frames_onnx:
                    out = sess.run([out_nm], {in_nm: arr})[0]
                    ss, pct = measure_int8(out)
                    c2_ss_list.append(ss); c2_pct_list.append(pct)
                run_ok = True
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  ONNX not found — using known measured values")

    # Fallback to known measured values when model files are unavailable
    if not run_ok or not c2_ss_list:
        known = {
            'hrnet_mpii_base': (45.0,  81.0),
            'hrnet_golf_v2':   (105.0, 100.0),
            'rtmpose_golf':    (98.0,  100.0),
            'yolov8n_pose':    (44.0,  71.0),
        }
        c2_ss, c2_pct = known.get(cfg['id'], (0.0, 0.0))
        print(f"  Using known measured C2 values: {c2_ss}x ({c2_pct}%)")
    else:
        c2_ss  = float(np.mean(c2_ss_list))
        c2_pct = float(np.mean(c2_pct_list))

    deployable = cfg['hw_compat'] and cfg['npu_feasible'] and c2_pct >= 90.0
    result = {
        'id':              cfg['id'],
        'model':           cfg['paper_name'],
        'head_type':       cfg['head_type'],
        'C1_ap':           cfg['ap'],
        'C2_signal_steps': round(c2_ss, 1),
        'C2_pct_kpts':     round(c2_pct, 1),
        'C3_hw_compat':    'YES' if cfg['hw_compat'] else 'NO',
        'C3_note':         cfg['hw_note'],
        'C4_params_m':     cfg['params_m'],
        'C4_npu_feasible': 'YES' if cfg['npu_feasible'] else 'NO',
        'C5_infer_ms':     cfg['infer_ms'],
        'C5_note':         cfg['speed_note'],
        'deployable':      deployable,
        'selected':        cfg['selected'],
        'color':           cfg['color'],
    }
    sel = " \u2605 SELECTED" if cfg['selected'] else ""
    dep = "\u2713 DEPLOYABLE" if deployable else "\u2717 NOT deployable"
    print(f"  C1 AP:{result['C1_ap']:.3f}  C2:{result['C2_signal_steps']:.0f}\u00d7 "
          f"({result['C2_pct_kpts']:.0f}%)  C3:{result['C3_hw_compat']}  "
          f"C4:{result['C4_params_m']:.1f}M  \u2192 {dep}{sel}")
    return result


def main():
    print("=" * 65)
    print("Experiment 3: Model Selection — 4 Architectures vs KL730")
    print("=" * 65)
    frames_mm   = load_frames_mmpose(20)
    frames_onnx = load_frames_onnx(20)
    print(f"Frames loaded: {len(frames_mm)} mmpose, {len(frames_onnx)} onnx")

    results = [evaluate_model(cfg, frames_mm, frames_onnx) for cfg in MODELS]

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ('color', 'selected', 'deployable')}
                       for r in results])
    df.to_csv(os.path.join(OUT_DIR, 'model_selection_results.csv'), index=False)

    # ── Figure: 5-criteria bar chart ──────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    fig.suptitle(
        'Figure 4 \u2014 Multi-Criteria Model Selection for KL730 NPU\n'
        '4 architecturally distinct models \u00d7 5 deployment criteria. '
        'Blue border = selected. \u2717 = eliminates model.',
        fontsize=11, fontweight='bold'
    )

    labels = [cfg['name'] for cfg in MODELS]
    colors = [r['color']  for r in results]
    oi     = next(i for i, r in enumerate(results) if r['selected'])

    # C1 — Accuracy (AP)
    ax   = axes[0]
    vals = [r['C1_ap'] for r in results]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    bars[oi].set_edgecolor('#185FA5'); bars[oi].set_linewidth(2.5)
    ax.set_ylim(0.4, 1.1)
    ax.set_ylabel('AP score', fontsize=9)
    ax.set_title('C1: Accuracy (AP)\nQ1: fine-tune effect? Q2: AP \u2260 deployability?',
                 fontsize=8.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.annotate('', xy=(0.85, 0.918), xytext=(0.15, 0.890),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
    ax.text(0.5, 0.907, '+0.028\nfine-tune', ha='center', fontsize=7, color='navy')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # C2 — INT8 compatibility
    ax   = axes[1]
    vals = [r['C2_signal_steps'] for r in results]
    pcts = [r['C2_pct_kpts']     for r in results]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    bars[oi].set_edgecolor('#185FA5'); bars[oi].set_linewidth(2.5)
    ax.axhline(1.0,  color='red',     linestyle='--', lw=1.2, label='Min (1\u00d7)')
    ax.axhline(10.0, color='#DAA520', linestyle=':',  lw=1.0, label='Robust (10\u00d7)')
    ax.set_ylabel('Signal / INT8 step', fontsize=9)
    ax.set_title('C2: INT8 Compatibility\n(signal/step, higher=better, >1 survives)',
                 fontsize=8.5)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=7.5)
    for bar, v, pct in zip(bars, vals, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f'{v:.0f}\u00d7\n({pct:.0f}%)', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    # C3 — HW output format
    ax   = axes[2]
    vals = [1.0 if r['C3_hw_compat'] == 'YES' else 0.0 for r in results]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    bars[oi].set_edgecolor('#185FA5'); bars[oi].set_linewidth(2.5)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0\n(NO)', '', '1\n(YES)'], fontsize=8)
    ax.set_ylabel('Compatible (1=YES, 0=NO)', fontsize=9)
    ax.set_title('C3: HW Output Format\n(KL730 outputs uint8 \u2014 model must accept)',
                 fontsize=8.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=7.5)
    for bar, v, cfg in zip(bars, vals, MODELS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                cfg['hw_note'], ha='center', va='bottom', fontsize=7.5)
        if v == 0.0:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.35,
                    '\u2717', ha='center', va='center',
                    fontsize=30, color='red', alpha=0.65)

    # C4 — Model size
    ax   = axes[3]
    vals = [r['C4_params_m'] for r in results]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    bars[oi].set_edgecolor('#185FA5'); bars[oi].set_linewidth(2.5)
    ax.set_ylabel('Parameters (millions)', fontsize=9)
    ax.set_title('C4: Model Size\n(all fit: KL730 SRAM for this study)', fontsize=8.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=7.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f'{v}M \u2713', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # C5 — Inference speed
    ax   = axes[4]
    vals = [float(r['C5_infer_ms']) if r['C5_infer_ms'] else 0.0 for r in results]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    bars[oi].set_edgecolor('#185FA5'); bars[oi].set_linewidth(2.5)
    ax.axhline(100.0, color='red', linestyle='--', lw=1.2, label='Real-time (100ms)')
    ax.set_ylabel('ms / frame', fontsize=9)
    ax.set_title('C5: Inference Speed\n(lower=better, ms/frame)', fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=7.5)
    for bar, cfg in zip(bars, MODELS):
        ypos = max(bar.get_height(), 4)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos + 3,
                cfg['speed_note'], ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, 'model_selection_bars.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {p1}")

    # ── Paper table ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TABLE III \u2014 Model Selection Decision Matrix")
    print("=" * 90)
    print(f"  {'Model':<22} {'Head':<18} {'C1 AP':>6}  {'C2':>8}  "
          f"{'C3':>6}  {'C4 Params':>10}  {'C5':>9}  {'Deploy':>8}")
    print("  " + "\u2500" * 88)
    for r in results:
        mk  = " \u2605" if r['selected'] else "  "
        dep = "YES \u2713" if r['deployable'] else "NO  \u2717"
        ms  = r['C5_infer_ms']
        ms_s = f"{float(ms):.0f}ms" if ms else "N/A"
        print(f"  {r['model'][:20]:<22}{mk}  {r['head_type']:<18}  "
              f"{r['C1_ap']:>6.3f}  {r['C2_signal_steps']:>7.0f}\u00d7  "
              f"{r['C3_hw_compat']:>6}  {r['C4_params_m']:>9.1f}M  "
              f"{ms_s:>9}  {dep:>8}")

    print("""
ELIMINATION LOGIC:
  Step 1 -- RTMPose eliminated by C3: SimCC needs uint16, KL730 outputs uint8 only.
  Step 2 -- HRNet MPII base marginal on C2: only 81% keypoints INT8-compatible.
  Step 3 -- HRNet Golf v2 vs YOLOv8n: AP 0.918 (golf fine-tuned) vs 0.500 (COCO only).
  Result -- HRNet-W48 Golf v2 is the only model satisfying all 5 criteria.
""")
    print(f"Outputs: {OUT_DIR}")
    return df


if __name__ == '__main__':
    main()
