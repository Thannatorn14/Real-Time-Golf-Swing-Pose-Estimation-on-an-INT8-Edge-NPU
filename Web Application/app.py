#  ─────────────────────────────────────────────────────────────────
#  CONFIGURE PATHS AND ADDRESSES BELOW
#  ─────────────────────────────────────────────────────────────────

#  IP address and port of the KL730 board running board_api.py
#  Example: "http://192.168.1.110:5000"
# BOARD_URL = "http://<BOARD_IP>:5000"

#  URL of Ollama running locally (for AI coaching)
#  Example: "http://localhost:11434"
# OLLAMA_URL = "http://localhost:11434"

#  Directory containing pro golfer NPZ sequence files and pose videos
#  Example on Windows: r"C:\path\to\video_outputs"
#  Example on Linux:   "/path/to/video_outputs"
# PROS_DIR = "/path/to/pro_sequences"

#  Same directory used for pro video serving
# PRO_VIDEOS_DIR = PROS_DIR

#  ─────────────────────────────────────────────────────────────────
# app.py — Golf Analysis Web App (runs on PC)
# BOARD_URL  = set in CONFIG section below
# Ollama:    set in CONFIG section below

# Run:
#     pip install flask requests numpy
#     python app.py
#     Open: http://localhost:8080  (default)

import glob

import os, json
import requests
import numpy as np
from flask import (Flask, render_template, request,
                   jsonify, Response, stream_with_context, send_file)

app = Flask(__name__)

# BOARD_URL set in CONFIG section above
# OLLAMA_URL set in CONFIG section above
# PROS_DIR set in CONFIG section above
MODEL      = "gemma2:2b"
# PRO_VIDEOS_DIR set in CONFIG section above
# ── Feature extraction + DTW ──────────────────────────────────────────
# ── Feature extraction — MUST match reprocess_pros_to_36.py exactly ──
L_SHLDR,R_SHLDR = 5,6
L_ELBOW,R_ELBOW = 7,8
L_WRIST,R_WRIST = 9,10
L_HIP,  R_HIP   = 11,12
L_KNEE, R_KNEE  = 13,14
L_ANKLE,R_ANKLE = 15,16

def _angle(a, b, c):
    ba=a-b; bc=c-b
    cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
    return float(np.degrees(np.arccos(np.clip(cos,-1.0,1.0))))

def kpts_to_features(kpts, img_h=480, img_w=640):
    """
    Exact same as reprocess_pros_to_36.py:
    - Normalize x by img_w, y by img_h (NOT by hip distance)
    - Use kpts[1:] (skip nose kpt[0]) -> 16x2 = 32 pos features
    - 4 angles: L-arm, R-arm, L-leg, R-leg
    Total: 36 features
    """
    norm = kpts.copy().astype(np.float32)
    norm[:, 0] /= (img_w + 1e-8)
    norm[:, 1] /= (img_h + 1e-8)
    pos_feats = norm[1:].flatten()   # skip nose (kpt[0]), 16x2=32
    ang_feats = np.array([
        _angle(kpts[L_SHLDR], kpts[L_ELBOW], kpts[L_WRIST]),
        _angle(kpts[R_SHLDR], kpts[R_ELBOW], kpts[R_WRIST]),
        _angle(kpts[L_HIP],   kpts[L_KNEE],  kpts[L_ANKLE]),
        _angle(kpts[R_HIP],   kpts[R_KNEE],  kpts[R_ANKLE]),
    ], dtype=np.float32) / 180.0  # normalize to 0-1 to match position scale
    return np.concatenate([pos_feats, ang_feats])  # 36 features

def kpts_to_features_seq(kpts_seq, img_h=480, img_w=640):
    """Process sequence of keypoints (T,17,2) -> (T,36)"""
    return np.stack([kpts_to_features(kpts_seq[t], img_h, img_w)
                     for t in range(kpts_seq.shape[0])])

def downsample(X, T=120):
    if X.shape[0] == T: return X
    old_idx = np.linspace(0, len(X)-1, len(X))
    new_idx = np.linspace(0, len(X)-1, T)
    out = np.zeros((T, X.shape[1]), dtype=np.float32)
    for d in range(X.shape[1]):
        out[:, d] = np.interp(new_idx, old_idx, X[:, d])
    return out

def dtw_distance(X, Y, step=3):
    Xs=X[::step]; Ys=Y[::step]
    T1,T2=Xs.shape[0],Ys.shape[0]
    d=np.full((T1+1,T2+1),np.inf); d[0,0]=0
    for i in range(1,T1+1):
        for j in range(1,T2+1):
            c=float(np.linalg.norm(Xs[i-1]-Ys[j-1]))
            d[i,j]=c+min(d[i-1,j],d[i,j-1],d[i-1,j-1])
    return float(d[T1,T2])

def extract_features(frames, img_h=480, img_w=640):
    """Extract features from board JSON frames.
    Works with both RTMPose (conf~0.9) and HRNet (conf~0.1-0.7)
    Matches reprocess_pros_to_36.py exactly.
    """
    kpts_list = []

    # Auto-detect resolution from keypoint positions
    all_x, all_y = [], []
    for f in frames[:20]:
        for kp in f.get('keypoints', []):
            if kp.get('conf', 0) > 0.1:
                all_x.append(kp['x']); all_y.append(kp['y'])
    if all_x:
        max_x, max_y = max(all_x), max(all_y)
        # Snap to common resolutions
        for w in [640, 1280, 1920]:
            if max_x < w * 1.1:
                img_w = w; break
        for h in [480, 720, 1080]:
            if max_y < h * 1.1:
                img_h = h; break

    for f in frames:
        kps = f.get('keypoints', [])
        if len(kps) == 17:
            valid = sum(1 for kp in kps
                        if kp.get('conf', 0) > 0.01 or kp.get('valid', False))
            if valid >= 5:
                kpts_list.append([[kp['x'], kp['y']] for kp in kps])

    if len(kpts_list) < 5:
        return None

    feats = []
    for kpts in kpts_list:
        kpts_arr = np.array(kpts, dtype=np.float32)
        feats.append(kpts_to_features(kpts_arr, img_h, img_w))

    X = np.stack(feats).astype(np.float32)
    X = np.nan_to_num(X)
    return downsample(X, T=120)

PRO_DESC = {
    'Jeeno1':    'Jeeno — compact backswing, strong hip rotation',
    'Jeeno2':    'Jeeno — full shoulder turn, aggressive downswing',
    'Jeeno3':    'Jeeno — balanced finish, consistent tempo',
    'Jeeno4':    'Jeeno — wide arc, powerful follow-through',
    'Nataliya1': 'Nataliya — smooth tempo, excellent balance',
    'Adam_Scott_app1':        'Adam Scott — classic upright swing, perfect posture',
    'adamscott1':             'Adam Scott — consistent ball striking, strong hip drive',
    'adamscott2':             'Adam Scott — full shoulder turn, late wrist release',
    'adamscott3':             'Adam Scott — balanced follow-through, high finish',
    'adamscott4':             'Adam Scott — wide takeaway, smooth transition',
    'adamscott5':             'Adam Scott — powerful downswing, excellent impact',
    'adamscott6':             'Adam Scott — controlled tempo, solid fundamentals',
    'rorymcllroy1':           'Rory McIlroy — explosive hip rotation, massive power',
    'rorymcllroy2':           'Rory McIlroy — wide arc, aggressive through impact',
    'rorymcllroy3':           'Rory McIlroy — consistent tempo, high ball flight',
    'tigerwood4':             'Tiger Woods — controlled power, precise ball striking',
    'Nelly_Korda':            'Nelly Korda — smooth rhythm, excellent weight transfer',
    'Xander_Schauffele_app1': 'Xander Schauffele — athletic swing, strong rotation',
    'Xander_Schauffele_app2': 'Xander Schauffele — powerful drive, balanced finish',
}

# Feature indices — matches reprocess_pros_to_36.py exactly
# kpts[1:] normalized by img_w/img_h, then 4 joint angles
# (kpt[i] skips nose, so kpt[j] in original = index (j-1)*2 in features)
KEY_FEATURES = {
    'l_shoulder_x': 8,   # kpts[5].x / img_w
    'r_shoulder_x': 10,  # kpts[6].x / img_w
    'l_hip_x':      20,  # kpts[11].x / img_w
    'r_hip_x':      22,  # kpts[12].x / img_w
    'l_wrist_y':    17,  # kpts[9].y / img_h
    'r_wrist_y':    19,  # kpts[10].y / img_h
    'l_elbow_angle':32,  # angle(L_SHLDR, L_ELBOW, L_WRIST)
    'r_elbow_angle':33,  # angle(R_SHLDR, R_ELBOW, R_WRIST)
    'l_knee_angle': 34,  # angle(L_HIP, L_KNEE, L_ANKLE)
    'r_knee_angle': 35,  # angle(R_HIP, R_KNEE, R_ANKLE)
}
FEAT_LABELS = {
    'l_shoulder_x':  'left shoulder position',
    'r_shoulder_x':  'right shoulder position',
    'l_hip_x':       'left hip rotation',
    'r_hip_x':       'right hip rotation',
    'l_wrist_y':     'left wrist height',
    'r_wrist_y':     'right wrist height',
    'l_elbow_angle': 'left elbow angle',
    'r_elbow_angle': 'right elbow angle',
    'l_knee_angle':  'left knee angle',
    'r_knee_angle':  'right knee angle',
}
PHASES = {
    'address':(0,15),'backswing':(15,50),'top':(50,60),
    'downswing':(60,80),'impact':(80,90),'follow_through':(90,120),
}

def compare_pros(X_user):
    results=[]
    if not os.path.exists(PROS_DIR): return results
    # Search recursively — NPZ files are in subfolders
    npz_files = glob.glob(os.path.join(PROS_DIR, '**', '*.npz'), recursive=True)
    # Also include flat old pros dir if it exists
    old_dir = PROS_DIR  # legacy path — adjust if needed
    if os.path.exists(old_dir):
        npz_files += glob.glob(os.path.join(old_dir, '*.npz'))
    seen = set()
    for fpath in sorted(npz_files):
        pro_id = os.path.splitext(os.path.basename(fpath))[0]
        if pro_id in seen: continue
        seen.add(pro_id)
        try:
            d=np.load(fpath,allow_pickle=True)
            X_pro=np.nan_to_num(d['X'].astype(np.float32))
            if X_pro.shape[0]<10: continue
            # Normalize angle features (last 4) to match user convention
            # User: actual joint angle /180 (1.0=straight, 0.5=90deg)
            # Pro NPZ may use: raw degrees, complement (0=straight), or already normalized
            ang = X_pro[:,32:].copy()
            if ang.max() > 1.5:
                # Raw degrees - normalize
                X_pro[:,32:] = ang / 180.0
            elif ang.mean() < 0.25:
                # Complement convention (0=straight, small values = straight joint)
                # Convert to supplement convention (1=straight)
                X_pro[:,32:] = 1.0 - ang
            # else: already normalized in same 0-1 supplement convention
            dist=dtw_distance(X_user,X_pro)
            score = dist  # raw; normalized relatively after all pros collected
            results.append({
                'pro_id':pro_id,'dist':round(dist,2),'score':round(score,4),
                'desc':PRO_DESC.get(pro_id, pro_id.replace('_',' '))
            })
        except Exception as e:
            print(f"Warning: {fpath}: {e}")
    results.sort(key=lambda x:x['dist'])
    # Relative scoring: best=100%, others proportional
    if results:
        min_d = results[0]['dist']
        for r in results:
            r['score'] = round(min(1.0, min_d / max(r['dist'], 1e-8)), 4)
    return results

def compute_deltas(X_user, pro_id):
    # Search recursively for the npz
    matches = glob.glob(os.path.join(PROS_DIR,'**',pro_id+'.npz'),recursive=True)
    if not matches:
        old_dir = PROS_DIR  # legacy path — adjust if needed
        old_path = os.path.join(old_dir, pro_id+'.npz')
        if os.path.exists(old_path): matches=[old_path]
    if not matches: return []
    X_pro=np.nan_to_num(np.load(matches[0],allow_pickle=True)['X'].astype(np.float32))
    _ang=X_pro[:,32:].copy()
    if _ang.max()>1.5: X_pro[:,32:]=_ang/180.0
    elif _ang.mean()<0.25: X_pro[:,32:]=1.0-_ang
    ang=X_pro[:,32:].copy()
    if ang.max()>1.5: X_pro[:,32:]=ang/180.0
    elif ang.mean()<0.25: X_pro[:,32:]=1.0-ang
    deltas=[]
    for phase,(t0,t1) in PHASES.items():
        u=X_user[t0:t1].mean(axis=0); p=X_pro[t0:t1].mean(axis=0)
        for feat,idx in KEY_FEATURES.items():
            if idx>=X_user.shape[1]: continue
            delta=float(u[idx]-p[idx])
            # adaptive threshold: 0.02 for position (0-1 range), 5.0 for angles (0-180)
            thr = 5.0 if idx >= 32 else 0.02
            if abs(delta)>thr:
                deltas.append({
                    'phase':phase,'metric':feat,
                    'label':FEAT_LABELS.get(feat,feat),
                    'direction':'needs more' if delta<0 else 'needs less',
                    'delta':(round(delta*180.0,1) if feat in ('l_elbow_angle','r_elbow_angle','l_knee_angle','r_knee_angle') else round(delta*100.0,1))
                })
    deltas.sort(key=lambda d:abs(d['delta']),reverse=True)
    return deltas[:6]

def get_llm_advice(pro_id, pro_desc, similarity, deltas):
    system = """You are an experienced professional golf coach.
Give clear, concise, practical advice to help a golfer match a specific pro golfer's technique.
Focus on the specific differences. Give actionable corrections and simple drills.
Keep response under 200 words. Be encouraging. Do NOT mention numbers or data values."""
    issues=[{'phase':d['phase'],'aspect':d['label'],
              'direction':d['direction']} for d in deltas]
    payload={
        'target_pro':pro_id,
        'pro_description':pro_desc,
        'similarity':'good match' if similarity>0.001 else 'needs work',
        'differences':issues
    }
    user=(f"Help this golfer's swing match {pro_id}'s technique.\n\n"
          f"COMPARISON:\n{json.dumps(payload,indent=2)}")
    r=requests.post(f"{OLLAMA_URL}/api/chat",json={
        'model':MODEL,
        'messages':[{'role':'system','content':system},
                    {'role':'user','content':user}],
        'options':{'temperature':0.3,'num_predict':400},
        'stream':False
    },timeout=120)
    r.raise_for_status()
    return r.json()['message']['content']


# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', board_url=BOARD_URL)

@app.route('/api/board/health')
def board_health():
    try:
        r=requests.get(f"{BOARD_URL}/health",timeout=3)
        return jsonify(r.json())
    except:
        return jsonify({"status":"offline"}), 503

@app.route('/api/board/stream')
def board_stream():
    def gen():
        with requests.get(f"{BOARD_URL}/stream",
                          stream=True,timeout=30) as r:
            for chunk in r.iter_content(chunk_size=4096):
                yield chunk
    return Response(stream_with_context(gen()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/record/start', methods=['POST'])
def record_start():
    r=requests.post(f"{BOARD_URL}/record/start",timeout=5)
    return jsonify(r.json())

@app.route('/api/record/stop', methods=['POST'])
def record_stop():
    r=requests.post(f"{BOARD_URL}/record/stop",timeout=5)
    return jsonify(r.json())

@app.route('/api/process', methods=['POST'])
def process():
    data=request.get_json() or {}
    r=requests.post(f"{BOARD_URL}/process",json=data,timeout=10)
    return jsonify(r.json())

@app.route('/api/status')
def board_status():
    r=requests.get(f"{BOARD_URL}/status",timeout=3)
    return jsonify(r.json())

@app.route('/api/video/processed')
def proxy_video():
    try:
        # Forward Range header from browser to board for seek support
        headers = {}
        if 'Range' in request.headers:
            headers['Range'] = request.headers['Range']
        r = requests.get(f"{BOARD_URL}/video/processed",
                         headers=headers, stream=True, timeout=30)
        # Pass back status (206 for range, 200 for full)
        resp = Response(
            stream_with_context(r.iter_content(chunk_size=65536)),
            status=r.status_code,
            mimetype='video/mp4'
        )
        # Forward range response headers
        for h in ('Content-Range','Accept-Ranges','Content-Length','Content-Type'):
            if h in r.headers:
                resp.headers[h] = r.headers[h]
        resp.headers['Accept-Ranges'] = 'bytes'
        return resp
    except Exception as e:
        return jsonify({"error":str(e)}), 503

@app.route('/api/videos')
def list_videos():
    try:
        r=requests.get(f"{BOARD_URL}/list/videos",timeout=5)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error":str(e)}), 503

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error":"No video file"}), 400
    f=request.files['video']
    if f.filename=='':
        return jsonify({"error":"No filename"}), 400
    try:
        r=requests.post(
            f"{BOARD_URL}/upload/video",
            files={'video':(f.filename, f.stream, f.mimetype)},
            timeout=120
        )
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error":str(e)}), 503
@app.route('/api/pro/video/<pro_id>')
def serve_pro_video(pro_id):
    import subprocess as sp
    # Find pose video (_pose.mp4 preferred)
    matches = glob.glob(
        os.path.join(PRO_VIDEOS_DIR, '**', f'{pro_id}_pose.mp4'), recursive=True)
    if not matches:
        matches = glob.glob(
            os.path.join(PRO_VIDEOS_DIR, '**', f'*{pro_id}*.mp4'), recursive=True)
    if not matches:
        return jsonify({"error": f"No video for {pro_id}"}), 404

    src  = matches[0]
    # Convert to H264 for browser compatibility (mp4v codec not supported by browsers)
    h264 = src.replace('_pose.mp4','_pose_h264.mp4')
    if '_pose' not in h264:
        h264 = src.replace('.mp4','_h264.mp4')
    if not os.path.exists(h264):
        print(f"[Pro Video] Converting {os.path.basename(src)} to H264...")
        try:
            sp.run(['ffmpeg','-y','-i',src,'-vcodec','libx264','-preset','fast',
                    '-crf','23','-movflags','+faststart','-an',h264],
                   check=True, capture_output=True, timeout=300)
            print(f"[Pro Video] Done: {os.path.basename(h264)}")
        except Exception as e:
            print(f"[Pro Video] ffmpeg error: {e}, serving raw")
            h264 = src

    path      = h264
    file_size = os.path.getsize(path)
    range_hdr = request.headers.get('Range')

    with open(path, 'rb') as f:
        if range_hdr:
            parts      = range_hdr.replace('bytes=', '').split('-')
            byte_start = int(parts[0])
            byte_end   = int(parts[1]) if parts[1] else file_size - 1
            length     = byte_end - byte_start + 1
            f.seek(byte_start)
            data = f.read(length)
            rv   = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
            rv.headers['Content-Range']  = f'bytes {byte_start}-{byte_end}/{file_size}'
            rv.headers['Accept-Ranges']  = 'bytes'
            rv.headers['Content-Length'] = str(length)
        else:
            data = f.read()
            rv   = Response(data, 200, mimetype='video/mp4', direct_passthrough=True)
            rv.headers['Accept-Ranges']  = 'bytes'
            rv.headers['Content-Length'] = str(file_size)
    return rv

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data=request.get_json() or {}
    pro_id=data.get('pro_id')

    r=requests.get(f"{BOARD_URL}/files/json",timeout=10)
    if r.status_code!=200:
        return jsonify({"error":"No processed swing found"}), 400

    swing_data=r.json()
    frames=swing_data.get('frames',[])

    X_user=extract_features(frames)
    if X_user is None:
        return jsonify({"error":"Not enough keypoints in swing"}), 400

    results=compare_pros(X_user)
    if not results:
        return jsonify({"error":"No pro data found"}), 400

    if pro_id:
        selected=next((r for r in results if r['pro_id']==pro_id),results[0])
    else:
        selected=results[0]

    deltas=compute_deltas(X_user,selected['pro_id'])

    try:
        advice=get_llm_advice(selected['pro_id'],selected['desc'],
                               selected['score'],deltas)
    except Exception as e:
        advice=f"Could not get LLM advice: {e}"

    return jsonify({
        "ranking":results,
        "selected":selected,
        "deltas":deltas,
        "advice":advice,
    })


if __name__ == '__main__':
    print("Golf Analysis Web App")
    print(f"Board:  {BOARD_URL}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Pros:   {PROS_DIR}")
    print("Open:   http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)