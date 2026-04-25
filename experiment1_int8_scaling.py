# experiment1_analysis.py
# ========================
# Generates Figure 7 — Fine-Tuning Progression.

# Shows AP score and INT8 compatibility across three training stages:
#   Stage 1: HRNet MPII (base pretrained)
#   Stage 2: Golf v1 (1,472 frames)
#   Stage 3: Golf v2 (15,068 frames) <- deployed

# Key findings:
#   - Golf fine-tuning incidentally sharpens heatmap peaks -> INT8 compatible
#   - v2 AP slightly lower than v1 due to harder face-on evaluation set (not regression)
#   - INT8 compatibility improves monotonically across all 3 stages: 45x -> 74x -> 105x

# ─────────────────────────────────────────────────────────────────
# SETUP  (edit the CONFIG section below before running)
# ─────────────────────────────────────────────────────────────────
# 1. Install dependencies:
#      pip install matplotlib pandas numpy

# 2. Set OUT_DIR to where you want the figure and CSV saved.

# 3. Run:
#      python experiment1_analysis.py

#    Outputs saved to OUT_DIR:
#      finetuning_progression.png   <- Figure for paper
#      finetuning_progression.csv   <- Raw numbers
# ─────────────────────────────────────────────────────────────────


import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — set this path before running
# ══════════════════════════════════════════════════════════════════════════

OUT_DIR = ""   # e.g. "/home/user/results/exp1"

# ══════════════════════════════════════════════════════════════════════════

if not OUT_DIR:
    raise ValueError("Please set OUT_DIR in the CONFIG section above.")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Measured results across 3 fine-tuning stages ─────────────────────────
stages = [
    {
        'label_x':    'HRNet MPII\n(base)',
        'ap':          0.890,
        'kpts_pct':    81,
        'signal_step': 45,
        'color':       '#888780',
    },
    {
        'label_x':    'Golf v1\n(1,472 frames)',
        'ap':          0.929,
        'kpts_pct':    95,
        'signal_step': 74,
        'color':       '#1D9E75',
    },
    {
        'label_x':    'Golf v2\n(15,068 frames) \u2605',
        'ap':          0.918,
        'kpts_pct':    100,
        'signal_step': 105,
        'color':       '#185FA5',
    },
]

ap_scores = [s['ap']          for s in stages]
kpts_pcts = [s['kpts_pct']    for s in stages]
sig_steps = [s['signal_step'] for s in stages]
colors3   = [s['color']       for s in stages]
x_labels  = [s['label_x']    for s in stages]

# Save CSV
df = pd.DataFrame([{k: v for k, v in s.items() if k != 'color'} for s in stages])
df.to_csv(os.path.join(OUT_DIR, 'finetuning_progression.csv'), index=False)

# ── Figure: Fine-Tuning Progression ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Figure 7 \u2014 Experiment 2: Fine-Tuning Progression\n'
    'AP score and INT8 compatibility across three training stages.',
    fontsize=11, fontweight='bold'
)

# ── Left panel: AP score bars + trend line ───────────────────────────────
ax1.set_title('C1: AP Score per Training Stage', fontsize=10)

bars1 = ax1.bar(x_labels, ap_scores, color=colors3, width=0.5,
                edgecolor='white', linewidth=0.5)
bars1[2].set_edgecolor('#185FA5'); bars1[2].set_linewidth(2.5)

# Trend line connecting bar tops
ax1.plot(np.arange(len(stages)), ap_scores, 'o--', color='#185FA5',
         linewidth=2, markersize=8, zorder=5)

ax1.set_ylim(0.86, 0.96)
ax1.set_ylabel('AP score', fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', labelsize=9)

# Value labels on bars
for bar, v in zip(bars1, ap_scores):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.0008,
             f'{v:.3f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

# Annotation: v2 AP drop is due to harder eval set, not capability regression
ax1.annotate('v2 AP slightly lower\n(harder face-on eval set)',
             xy=(2.0, 0.918), xytext=(1.45, 0.925),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
             fontsize=8, color='red', ha='center', style='italic')

# ── Right panel: INT8 compatibility — paired bars ────────────────────────
ax2.set_title('C2: INT8 Compatibility per Stage\n'
              '(% keypoints ok and signal/step)', fontsize=10)

x = np.arange(len(stages))
w = 0.35

# Dark bars = % keypoints INT8-ok, light bars = signal/step ratio
bars_kpts = ax2.bar(x - w / 2, kpts_pcts, width=w, color=colors3,
                    label='Kpts INT8-ok (%)', edgecolor='white', linewidth=0.5)
bars_sig  = ax2.bar(x + w / 2, sig_steps,  width=w, color=colors3,
                    alpha=0.45, label='Signal/step \u00d7',
                    edgecolor='white', linewidth=0.5)

# Blue border on selected stage (Golf v2)
bars_kpts[2].set_edgecolor('#185FA5'); bars_kpts[2].set_linewidth(2.5)
bars_sig[2].set_edgecolor('#185FA5');  bars_sig[2].set_linewidth(2.5)

ax2.axhline(95, color='red', linestyle='--', lw=1.5, label='95% threshold')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontsize=9)
ax2.set_ylabel('Value', fontsize=10)
ax2.set_ylim(0, 130)
ax2.legend(fontsize=8, loc='upper left')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Value labels
for bar, v in zip(bars_kpts, kpts_pcts):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 1,
             f'{v}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, v in zip(bars_sig, sig_steps):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 1,
             f'{v}\u00d7', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
p1 = os.path.join(OUT_DIR, 'finetuning_progression.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved: {p1}")

# ── Print paper table ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE \u2014 Fine-Tuning Progression")
print("=" * 70)
print(f"  {'Stage':<35} {'AP':>6}  {'Kpts INT8-ok':>13}  {'Signal/step':>12}")
print("  " + "\u2500" * 70)
for s in stages:
    mk = " \u2190" if '\u2605' in s['label_x'] else "  "
    print(f"  {s['label_x'].replace(chr(10), ' '):<35}{mk}  "
          f"{s['ap']:>6.3f}  {s['kpts_pct']:>12}%  {s['signal_step']:>11}\u00d7")

print("""
KEY FINDINGS:
  1. MPII base:  81% keypoints INT8-compatible -- marginal failure on KL730
  2. Golf v1:    95% compatible -- fine-tuning sharpens heatmap peaks
  3. Golf v2:   100% compatible -- all 17 keypoints survive INT8 quantization
  4. v2 AP (0.918) slightly lower than v1 (0.929) because the v2 evaluation
     set contains only harder face-on frames -- NOT a capability regression.
  5. Signal/step improves monotonically: 45x -> 74x -> 105x
""")
print(f"Outputs: {OUT_DIR}")
