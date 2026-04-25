# ─────────────────────────────────────────────────────────────────
# USAGE:
#   python process_golf_pose.py \
#       --video  /path/to/input.mp4 \
#       --save   /path/to/output.mp4 \
#       --save_json /path/to/phases.json
#
# NOTE: This is the CPU-only version using rtmlib (RTMPose).
#       The board NPU version is in board_api.py.
#       Parameters (alpha, window) match the deployed board_api.py config.
# ─────────────────────────────────────────────────────────────────

# process_golf_pose.py — Golf pose with swing phase detection + KeypointSmoother

import argparse, json, os, time, signal
import cv2
import numpy as np

os.environ.setdefault("DISPLAY", ":0")

_stop = False
def _sig(sig, frame):
    global _stop; _stop = True
try:
    signal.signal(signal.SIGINT, _sig)
except ValueError:
    pass

KP = {
    'nose':0,'l_eye':1,'r_eye':2,'l_ear':3,'r_ear':4,
    'l_sho':5,'r_sho':6,'l_elb':7,'r_elb':8,
    'l_wri':9,'r_wri':10,'l_hip':11,'r_hip':12,
    'l_kne':13,'r_kne':14,'l_ank':15,'r_ank':16
}

SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,11),(6,12),(11,12),
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
]

PHASE_COLORS = {
    'Address':        (  0, 200,   0),
    'Takeaway':       (  0, 165, 255),
    'Backswing':      (  0,   0, 255),
    'Top':            (255,   0, 255),
    'Downswing':      (255, 165,   0),
    'Impact':         (  0, 255, 255),
    'Follow-through': (255, 255,   0),
    'Finish':         (180, 255, 180),
}

JOINT_COLORS = {
    'head':  (255, 220, 100),
    'torso': (100, 220, 255),
    'l_arm': ( 50, 150, 255),
    'r_arm': (255, 100, 100),
    'l_leg': ( 50, 255, 150),
    'r_leg': (255, 200,  50),
}

EDGE_COLORS = [
    JOINT_COLORS['head'],  JOINT_COLORS['head'],
    JOINT_COLORS['head'],  JOINT_COLORS['head'],
    JOINT_COLORS['torso'], JOINT_COLORS['torso'],
    JOINT_COLORS['torso'], JOINT_COLORS['torso'],
    JOINT_COLORS['l_arm'], JOINT_COLORS['l_arm'],
    JOINT_COLORS['r_arm'], JOINT_COLORS['r_arm'],
    JOINT_COLORS['l_leg'], JOINT_COLORS['l_leg'],
    JOINT_COLORS['r_leg'], JOINT_COLORS['r_leg'],
]

NODE_COLORS = [
    JOINT_COLORS['head'],  JOINT_COLORS['head'],
    JOINT_COLORS['head'],  JOINT_COLORS['head'],
    JOINT_COLORS['head'],  JOINT_COLORS['l_arm'],
    JOINT_COLORS['r_arm'], JOINT_COLORS['l_arm'],
    JOINT_COLORS['r_arm'], JOINT_COLORS['l_arm'],
    JOINT_COLORS['r_arm'], JOINT_COLORS['l_leg'],
    JOINT_COLORS['r_leg'], JOINT_COLORS['l_leg'],
    JOINT_COLORS['r_leg'], JOINT_COLORS['l_leg'],
    JOINT_COLORS['r_leg'],
]


class KeypointSmoother:
    """
    Temporal smoother for keypoints using exponential moving average.
    Lower alpha = smoother but more lag.
    alpha=0.35, window=6 matches board_api.py deployed configuration.
    """
    def __init__(self, alpha=0.35, window=6):
        self.alpha   = alpha
        self.window  = window
        self.smooth  = None
        self.s_conf  = None

    def update(self, kpts, conf, score_thr=0.1):
        if kpts is None or conf is None:
            return self.smooth, self.s_conf
        if self.smooth is None:
            self.smooth = kpts.copy()
            self.s_conf = conf.copy()
            return self.smooth.copy(), self.s_conf.copy()
        # EMA update — only update keypoints with decent confidence
        for i in range(len(kpts)):
            if conf[i] >= score_thr * 0.3:
                self.smooth[i] = (self.alpha * kpts[i] +
                                  (1.0 - self.alpha) * self.smooth[i])
                self.s_conf[i] = (self.alpha * conf[i] +
                                  (1.0 - self.alpha) * self.s_conf[i])
        return self.smooth.copy(), self.s_conf.copy()

    def reset(self):
        self.smooth = None
        self.s_conf = None


class SwingPhaseDetector:
    def __init__(self, window=12):
        self.window        = window
        self.history       = []
        self.phase         = 'Address'
        self.phase_history = []

    def update(self, frame_idx, kpts, conf, conf_thr, frame_h):
        l_wri = KP['l_wri']; r_wri = KP['r_wri']
        l_hip = KP['l_hip']; r_hip = KP['r_hip']

        # Use lower threshold for wrist detection
        wrist_thr = min(conf_thr, 0.05)
        wrist_y = None
        if conf[l_wri] >= wrist_thr and conf[r_wri] >= wrist_thr:
            wrist_y = (kpts[l_wri,1] + kpts[r_wri,1]) / 2.0 / frame_h
        elif conf[l_wri] >= wrist_thr:
            wrist_y = kpts[l_wri,1] / frame_h
        elif conf[r_wri] >= wrist_thr:
            wrist_y = kpts[r_wri,1] / frame_h

        if wrist_y is None:
            self.phase_history.append(self.phase)
            return self.phase

        self.history.append((frame_idx, wrist_y))
        if len(self.history) > self.window * 3:
            self.history.pop(0)

        if len(self.history) < self.window:
            self.phase_history.append('Address')
            return 'Address'

        recent   = [y for _,y in self.history[-self.window:]]
        smooth_y = np.mean(recent)

        if len(self.history) >= self.window * 2:
            prev  = np.mean([y for _,y in self.history[-self.window*2:-self.window]])
            vel   = smooth_y - prev
        else:
            vel = 0.0

        hip_y = None
        if conf[l_hip] >= conf_thr and conf[r_hip] >= conf_thr:
            hip_y = (kpts[l_hip,1] + kpts[r_hip,1]) / 2.0 / frame_h

        all_y   = [y for _,y in self.history]
        min_y   = min(all_y)
        max_y   = max(all_y)
        range_y = max_y - min_y
        VEL_THRESH = 0.005  # sensitive for slow-motion video

        if range_y < 0.02:
            phase = 'Address'
        elif vel < -VEL_THRESH:
            phase = 'Takeaway' if smooth_y > min_y + range_y * 0.7 else 'Backswing'
        elif vel > VEL_THRESH:
            if smooth_y < max_y - range_y * 0.3:
                phase = 'Downswing'
            elif hip_y and smooth_y >= hip_y - 0.05:
                phase = 'Impact'
            else:
                phase = 'Follow-through'
        else:
            if smooth_y <= min_y + range_y * 0.15:
                phase = 'Top'
            elif smooth_y >= max_y - range_y * 0.15 and range_y > 0.05:
                phase = 'Impact'
            elif len(self.phase_history) > 0 and \
                 self.phase_history[-1] in ('Impact','Follow-through'):
                phase = 'Finish'
            else:
                phase = self.phase

        PHASE_ORDER = ['Address','Takeaway','Backswing','Top',
                       'Downswing','Impact','Follow-through','Finish']
        if self.phase in PHASE_ORDER and phase in PHASE_ORDER:
            cur_idx = PHASE_ORDER.index(self.phase)
            new_idx = PHASE_ORDER.index(phase)
            if new_idx >= cur_idx or phase == 'Address':
                self.phase = phase
        else:
            self.phase = phase

        self.phase_history.append(self.phase)
        return self.phase


def draw_elegant_skeleton(frame, kpts, conf, conf_thr, phase):
    vis = frame.copy()
    for idx, (a,b) in enumerate(SKELETON_EDGES):
        if conf[a] >= conf_thr and conf[b] >= conf_thr:
            p1 = tuple(np.round(kpts[a]).astype(int))
            p2 = tuple(np.round(kpts[b]).astype(int))
            ec = EDGE_COLORS[idx]
            cv2.line(vis, p1, p2, (0,0,0), 7, cv2.LINE_AA)
            cv2.line(vis, p1, p2, ec, 4, cv2.LINE_AA)
            cv2.line(vis, p1, p2,
                     tuple(min(255,c+80) for c in ec), 1, cv2.LINE_AA)
    for i in range(17):
        if conf[i] >= conf_thr:
            p  = tuple(np.round(kpts[i]).astype(int))
            jc = NODE_COLORS[i]
            cv2.circle(vis, p, 9, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(vis, p, 7, jc, -1, cv2.LINE_AA)
            cv2.circle(vis, p, 3, (255,255,255), -1, cv2.LINE_AA)
    return vis


def draw_hud(vis, phase, frame_idx, total, infer_fps, detected):
    h, w = vis.shape[:2]
    color = PHASE_COLORS.get(phase, (255,255,255))
    overlay = vis.copy()
    cv2.rectangle(overlay, (0,0), (w,55), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
    cv2.putText(vis, phase.upper(), (10,38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,0), 4)
    cv2.putText(vis, phase.upper(), (10,38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)
    cv2.rectangle(vis, (0,0), (6,55), color, -1)
    pct = f"{100*frame_idx/max(total,1):.0f}%"
    cv2.putText(vis, f"{pct}  KPts:{detected}/17",
                (w-220,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(vis, f"FPS:{infer_fps:.1f}",
                (w-100,44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)
    bar_h = 6
    cv2.rectangle(vis, (0,h-bar_h), (w,h), (40,40,40), -1)
    prog_w = int(w * frame_idx / max(total,1))
    cv2.rectangle(vis, (0,h-bar_h), (prog_w,h), color, -1)
    return vis


def draw_phase_legend(vis, current_phase):
    h, w = vis.shape[:2]
    phases = list(PHASE_COLORS.keys())
    x0, y0 = w-175, 65
    box_h = len(phases)*22+10
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0-5,y0-5), (w-5,y0+box_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
    for i, phase in enumerate(phases):
        c = PHASE_COLORS[phase]
        y = y0 + i*22 + 16
        cv2.rectangle(vis, (x0,y-10), (x0+12,y+2), c, -1)
        is_cur = (phase == current_phase)
        txt_color = c if is_cur else (120,120,120)
        weight = 2 if is_cur else 1
        cv2.putText(vis, phase, (x0+18,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_color, weight)
    return vis


def main():
    global _stop
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--save",       default="./result.mp4"  # set output path with --save /your/path/result.mp4)
    ap.add_argument("--save_json",  default=None  # set JSON output path with --save_json /your/path/phases.json)
    ap.add_argument("--conf_thr",   type=float, default=0.3)
    ap.add_argument("--mode",       default="lightweight")
    ap.add_argument("--max_frames", type=int,   default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    print(f"Loading RTMPose ({args.mode})...")
    from rtmlib import Body
    body = Body(mode=args.mode, backend="onnxruntime", device="cpu")
    print("Loaded!")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fw}x{fh}  {fps_in:.1f}FPS  {total} frames")

    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (fw,fh))
    detector = SwingPhaseDetector(window=12)
    smoother = KeypointSmoother(alpha=0.35, window=6)
    frame_id = 0; all_frames = []; t_start = time.time()

    while not _stop:
        ok, frame = cap.read()
        if not ok: break

        t0 = time.time()
        keypoints, scores = body(frame)
        infer_fps = 1.0 / max(time.time()-t0, 1e-6)

        kpts_raw = keypoints[0] if len(keypoints) > 0 else None
        conf_raw = scores[0]    if len(scores)    > 0 else None

        kpts, conf = smoother.update(kpts_raw, conf_raw, args.conf_thr)

        phase = 'Address'
        if kpts is not None and conf is not None:
            phase = detector.update(frame_id, kpts, conf, args.conf_thr, fh)

        if kpts is not None and conf is not None:
            vis      = draw_elegant_skeleton(frame, kpts, conf, args.conf_thr, phase)
            detected = int(np.sum(conf >= args.conf_thr))
        else:
            vis = frame.copy(); detected = 0

        vis = draw_hud(vis, phase, frame_id, total, infer_fps, detected)
        vis = draw_phase_legend(vis, phase)

        if args.save_json:
            fkpts = []
            if kpts is not None:
                for i in range(17):
                    fkpts.append({
                        "x":    round(float(kpts[i,0]),2),
                        "y":    round(float(kpts[i,1]),2),
                        "conf": round(float(conf[i]),4),
                        "valid": bool(conf[i] >= args.conf_thr)
                    })
            all_frames.append({"frame_id":frame_id,"phase":phase,"keypoints":fkpts})

        writer.write(vis)
        frame_id += 1

        if frame_id % 30 == 0:
            elapsed = time.time()-t_start
            eta = (total-frame_id)/max(frame_id/elapsed,1e-6)
            print(f"  {frame_id}/{total} Phase:{phase:15s} KPts:{detected}/17 ETA:{eta:.0f}s")

        if args.max_frames and frame_id >= args.max_frames:
            break

    cap.release(); writer.release()

    if args.save_json and all_frames:
        with open(args.save_json,"w") as f:
            json.dump({"total_frames":frame_id,"frames":all_frames}, f, indent=2)
        print(f"JSON: {args.save_json}")

    print(f"\nDone! {frame_id} frames → {args.save}")


if __name__ == "__main__":
    main()
