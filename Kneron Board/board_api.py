#  ─────────────────────────────────────────────────────────────────
#  CONFIGURE PATHS BELOW — set these for your board environment
#  ─────────────────────────────────────────────────────────────────

#  Path to the combined NEF file (YOLOv7-tiny + HRNet merged)
#  Example: '/root/combined_golf.nef'
# YOLO_NEF = '/path/to/combined_golf.nef'
# HRNET_NEF = YOLO_NEF  # same file, different model id

#  Path to the KL730 firmware tar file
#  Example: '/root/kneopi-examples/ai_application/plus_python/res/firmware/KL730/kp_firmware.tar'
# FIRMWARE = '/path/to/kp_firmware.tar'

#  Directory to save recorded and processed videos
#  Example: '/root/outputs'
# OUTPUTS_DIR = '/path/to/outputs'

#  Path to YOLOX ONNX for CPU fallback (used when NPU is unavailable)
#  Example: '/root/yolox_det.onnx'
# YOLOX_CPU_MODEL = '/path/to/yolox_det.onnx'

#  Path to RTMPose ONNX for CPU pose fallback
# # Example: '/root/rtmpose_golf.onnx'
# RTMPOSE_CPU_MODEL = '/path/to/rtmpose_golf.onnx'

# ─────────────────────────────────────────────────────────────────
# board_api.py - Dual NPU: YOLOv7-tiny (NPU ~10ms) + HRNet Golf (NPU ~64ms)
# Run: python3 board_api.py  (from the board/ directory)

import os, sys, json, time, threading, subprocess
import cv2, numpy as np
from flask import Flask, Response, request, jsonify, send_file

os.environ.setdefault("DISPLAY", ":0")
# sys.path is configured automatically when running from the board directory

app = Flask(__name__)

# ── PATHS ─────────────────────────────────────────────────────────────
YOLO_NEF   = YOLO_NEF      # set in CONFIG section above
HRNET_NEF  = HRNET_NEF     # set in CONFIG section above
FIRMWARE   = FIRMWARE       # set in CONFIG section above

# ── CONSTANTS ─────────────────────────────────────────────────────────
SCORE_THR    = 0.1
SCALE_FACTOR = 1000.0
SLOWDOWN     = 1.5
DET_EVERY    = 3
YOLO_CONF    = 0.35
YOLO_IOU     = 0.45
PERSON_CLASS = 0
YOLO_INPUT_W = 640
YOLO_INPUT_H = 640
# YOLOv7-tiny anchors 640x640
ANCHORS = [
    [[12,16],  [19,36],   [40,28]],
    [[36,75],  [76,55],   [72,146]],
    [[142,110],[192,243], [459,401]],
]
STRIDES = [8, 16, 32]

# ── GLOBALS ───────────────────────────────────────────────────────────
_lock          = threading.Lock()
_camera        = None
_recording     = False
_writer        = None
_rec_path      = None
_rec_frames    = 0
_rec_start     = 0.0
_status        = {"state":"idle","message":"Ready","progress":0}
_latest_frame  = None
_device_group  = None
_yolo_model    = None
_hrnet_model   = None
_npu_ready     = False
_det_npu_ready = False

FLIP_PAIRS = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

_calib_swap=None; _calib_count=0; _calib_votes=[]; CALIB_FRAMES=15

def reset_lr_calibration():
    global _calib_swap,_calib_count,_calib_votes
    _calib_swap=None; _calib_count=0; _calib_votes=[]

def fix_lr_swap(kpts, scores, score_thr=0.1, frame_w=640):
    global _calib_swap,_calib_count,_calib_votes
    kpts=kpts.copy(); scores=scores.copy()
    if _calib_swap is None:
        if scores[3]>=score_thr*2 and scores[4]>=score_thr*2:
            ear_lr=(kpts[3][0]>kpts[4][0]); sho_lr=(kpts[5][0]>kpts[6][0])
            _calib_votes.append(ear_lr!=sho_lr); _calib_count+=1
        if _calib_count>=CALIB_FRAMES:
            _calib_swap=(sum(_calib_votes)>len(_calib_votes)/2)
            print(f"[LR-Calib] swap={_calib_swap} ({sum(_calib_votes)}/{len(_calib_votes)})")
        if scores[3]>=score_thr and scores[4]>=score_thr:
            if (kpts[3][0]>kpts[4][0])!=(kpts[5][0]>kpts[6][0]):
                for l,r in FLIP_PAIRS:
                    kpts[l],kpts[r]=kpts[r].copy(),kpts[l].copy()
                    scores[l],scores[r]=scores[r],scores[l]
        return kpts,scores
    if _calib_swap:
        for l,r in FLIP_PAIRS:
            kpts[l],kpts[r]=kpts[r].copy(),kpts[l].copy()
            scores[l],scores[r]=scores[r],scores[l]
    return kpts,scores

# ── DUAL NPU INIT ─────────────────────────────────────────────────────
def init_npu():
    global _device_group, _yolo_model, _hrnet_model, _npu_ready, _det_npu_ready
    try:
        import kp
        print("[NPU] Connecting KL730...")
        dg = kp.core.connect_devices(usb_port_ids=[0])
        kp.core.load_firmware_from_file(device_group=dg,
            scpu_fw_path=FIRMWARE, ncpu_fw_path='')
        kp.core.set_timeout(device_group=dg, milliseconds=30000)

        print("[NPU] Loading combined NEF (YOLOv7-tiny + HRNet)...")
        nef = kp.core.load_model_from_file(device_group=dg, file_path=YOLO_NEF)
        print(f"[NPU] {len(nef.models)} models loaded from combined NEF")
        for m in nef.models:
            if m.id == 11111:
                _yolo_model = m
                print(f"[NPU] YOLOv7-tiny  id={m.id}")
            else:
                _hrnet_model = m
                print(f"[NPU] HRNet-Golf   id={m.id}")

        _device_group = dg

        # Warmup YOLO
        dummy565 = cv2.cvtColor(np.zeros((640,640,3),dtype=np.uint8), cv2.COLOR_BGR2BGR565)
        for _ in range(2):
            inp = kp.GenericInputNodeImage(image=dummy565,
                resize_mode=kp.ResizeMode.KP_RESIZE_DISABLE,
                padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON,
                image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)
            desc = kp.GenericImageInferenceDescriptor(
                model_id=_yolo_model.id, inference_number=0,
                input_node_image_list=[inp])
            kp.inference.generic_image_inference_send(device_group=dg,
                generic_inference_input_descriptor=desc)
            kp.inference.generic_image_inference_receive(device_group=dg)
        _det_npu_ready = True
        print("[NPU] YOLOv7 warmup OK")

        # Warmup HRNet
        dummy_hr565 = cv2.cvtColor(np.zeros((256,192,3),dtype=np.uint8), cv2.COLOR_BGR2BGR565)
        for _ in range(2):
            inp = kp.GenericInputNodeImage(image=dummy_hr565,
                resize_mode=kp.ResizeMode.KP_RESIZE_DISABLE,
                padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON,
                image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)
            desc = kp.GenericImageInferenceDescriptor(
                model_id=_hrnet_model.id, inference_number=0,
                input_node_image_list=[inp])
            kp.inference.generic_image_inference_send(device_group=dg,
                generic_inference_input_descriptor=desc)
            kp.inference.generic_image_inference_receive(device_group=dg)
        _npu_ready = True
        print("[NPU] HRNet warmup OK")
        print("[NPU] Dual-NPU ready: YOLO~10ms + HRNet~64ms")
        return True
    except Exception as e:
        print(f"[NPU] Init failed: {e}")
        import traceback; traceback.print_exc()
        return False

# ── YOLOV7 DECODE & NMS ───────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def _decode_yolov7_head(raw_node, stride, anchors, orig_h, orig_w):
    raw = np.array(raw_node.ndarray, dtype=np.float32)
    # raw shape from NPU: [1,3,gh,gw,85] or flat
    if raw.ndim != 5:
        gh = gw = int(round(YOLO_INPUT_W / stride))
        raw = raw.reshape(1, 3, gh, gw, 85)
    _, na, gh, gw, _ = raw.shape
    pred = _sigmoid(raw[0])  # [3,gh,gw,85]
    gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
    grid = np.stack([gx, gy], axis=-1).astype(np.float32)
    boxes = []
    for a in range(na):
        pw, ph = anchors[a]
        p    = pred[a]
        bxy  = (p[..., :2]*2 - 0.5 + grid) * stride
        bwh  = (p[..., 2:4]*2)**2 * np.array([pw, ph])
        conf = p[..., 4] * p[..., 5+PERSON_CLASS]
        mask = conf > YOLO_CONF
        if not mask.any(): continue
        sx = orig_w / YOLO_INPUT_W; sy = orig_h / YOLO_INPUT_H
        cx = bxy[mask,0]; cy = bxy[mask,1]
        bw = bwh[mask,0]; bh = bwh[mask,1]; c = conf[mask]
        x1 = np.clip((cx-bw/2)*sx, 0, orig_w)
        y1 = np.clip((cy-bh/2)*sy, 0, orig_h)
        x2 = np.clip((cx+bw/2)*sx, 0, orig_w)
        y2 = np.clip((cy+bh/2)*sy, 0, orig_h)
        for i in range(len(c)):
            boxes.append([x1[i],y1[i],x2[i],y2[i],float(c[i])])
    return boxes

def _nms(boxes):
    if not boxes: return []
    b  = np.array(boxes)
    x1,y1,x2,y2,sc = b[:,0],b[:,1],b[:,2],b[:,3],b[:,4]
    areas = (x2-x1)*(y2-y1)
    order = sc.argsort()[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-8)
        order=order[1:][iou<=YOLO_IOU]
    return [boxes[k][:4] for k in keep]

def run_yolo_npu(img_bgr):
    import kp
    orig_h, orig_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr,(YOLO_INPUT_W,YOLO_INPUT_H),interpolation=cv2.INTER_LINEAR)
    img565  = cv2.cvtColor(resized, cv2.COLOR_BGR2BGR565)
    inp = kp.GenericInputNodeImage(image=img565,
        resize_mode=kp.ResizeMode.KP_RESIZE_DISABLE,
        padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
        normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON,
        image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)
    desc = kp.GenericImageInferenceDescriptor(
        model_id=_yolo_model.id, inference_number=0,
        input_node_image_list=[inp])
    kp.inference.generic_image_inference_send(device_group=_device_group,
        generic_inference_input_descriptor=desc)
    raw = kp.inference.generic_image_inference_receive(device_group=_device_group)
    all_boxes = []
    for i in range(3):
        out = kp.inference.generic_inference_retrieve_float_node(
            node_idx=i, generic_raw_result=raw,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
        all_boxes.extend(_decode_yolov7_head(out, STRIDES[i], ANCHORS[i], orig_h, orig_w))
    return _nms(all_boxes)

# ── HRNET POSE ────────────────────────────────────────────────────────
def decode_heatmap(heatmaps, inv_trans):
    hm = heatmaps[0] / SCALE_FACTOR if heatmaps.max() > 100 else heatmaps[0]
    kpts=[]; scores=[]
    for i in range(17):
        h=hm[i]; idx=np.argmax(h)
        py,px=np.unravel_index(idx,h.shape)
        score=float(h[py,px])
        if 1<=px<h.shape[1]-1 and 1<=py<h.shape[0]-1:
            dx=0.5*(h[py,px+1]-h[py,px-1])/(abs(h[py,px+1]+h[py,px-1]-2*h[py,px])+1e-8)
            dy=0.5*(h[py+1,px]-h[py-1,px])/(abs(h[py+1,px]+h[py-1,px]-2*h[py,px])+1e-8)
            dx=np.clip(dx,-0.5,0.5); dy=np.clip(dy,-0.5,0.5)
            px_sub=(px+dx)*(192/48); py_sub=(py+dy)*(256/64)
        else:
            px_sub=px*(192/48); py_sub=py*(256/64)
        kpts.append([px_sub,py_sub]); scores.append(score)
    kpts=np.array(kpts); scores=np.array(scores)
    return np.concatenate([kpts,np.ones((17,1))],axis=1)@inv_trans.T, scores

def run_hrnet_npu(img_bgr, bbox):
    import kp
    from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, get_warp_matrix
    fw=img_bgr.shape[1]
    center,scale=bbox_xyxy2cs(bbox,padding=1.25)
    warp_mat=get_warp_matrix(center,scale,0,output_size=(192,256))
    inv_trans=cv2.invertAffineTransform(warp_mat)
    crop=cv2.warpAffine(img_bgr,warp_mat,(192,256),flags=cv2.INTER_LINEAR,borderValue=0)
    img565=cv2.cvtColor(crop,cv2.COLOR_BGR2BGR565)
    inp=kp.GenericInputNodeImage(image=img565,
        resize_mode=kp.ResizeMode.KP_RESIZE_DISABLE,
        padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
        normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON,
        image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)
    desc=kp.GenericImageInferenceDescriptor(
        model_id=_hrnet_model.id,inference_number=0,input_node_image_list=[inp])
    kp.inference.generic_image_inference_send(device_group=_device_group,
        generic_inference_input_descriptor=desc)
    raw=kp.inference.generic_image_inference_receive(device_group=_device_group)
    out=kp.inference.generic_inference_retrieve_float_node(
        node_idx=0,generic_raw_result=raw,
        channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT)
    hm=np.array(out.ndarray,dtype=np.float32)
    kpts,scores=decode_heatmap(hm,inv_trans)
    return fix_lr_swap(kpts,scores,SCORE_THR,fw)

# ── CPU FALLBACK ──────────────────────────────────────────────────────
_det_cpu = None
def detect_persons(frame):
    global _det_cpu
    if _det_npu_ready:
        try:
            return run_yolo_npu(frame)
        except Exception as e:
            print(f"[Det-NPU err] {e}")
    if _det_cpu is None:
        from rtmlib import YOLOX
        _det_cpu = YOLOX(onnx_model=YOLOX_CPU_MODEL,
                         model_input_size=(640,640),
                         backend='onnxruntime', device='cpu')
        print("[API] YOLOX CPU fallback loaded")
    bboxes = _det_cpu(frame)
    return [b[:4] for b in bboxes] if len(bboxes) > 0 else []

# ── VIDEO SLOWDOWN ────────────────────────────────────────────────────
def slowdown_video(video_path, factor=3.0):
    slow_path = video_path.replace('.mp4','_slow.mp4')
    if os.path.exists(slow_path): os.remove(slow_path)
    try:
        subprocess.run(['ffmpeg','-y','-i',video_path,
            '-vf',f'setpts={factor}*PTS','-r','30','-an',slow_path],
            check=True, capture_output=True, timeout=120)
        cap=cv2.VideoCapture(slow_path)
        nf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        print(f"[Process] Slow video: {nf} frames")
        return slow_path
    except Exception as e:
        print(f"[Process] Slowdown failed: {e}"); return video_path

# ── CAMERA ────────────────────────────────────────────────────────────
def get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        for b in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cam=cv2.VideoCapture(0,b)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                _camera=cam; break
    return _camera

def camera_thread():
    global _latest_frame, _recording, _writer, _rec_frames
    while True:
        cap=get_camera(); ret,frame=cap.read()
        if not ret: time.sleep(0.01); continue
        with _lock:
            _latest_frame=frame.copy()
            if _recording and _writer:
                _writer.write(frame); _rec_frames+=1
        time.sleep(0.001)

def generate_stream():
    while True:
        with _lock: frame=_latest_frame
        if frame is None: time.sleep(0.05); continue
        vis=frame.copy()
        with _lock: is_rec=_recording; nf=_rec_frames
        if is_rec:
            cv2.circle(vis,(30,30),12,(0,0,255),-1)
            cv2.putText(vis,f"REC {nf}f",(50,38),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        else:
            cv2.circle(vis,(30,30),10,(0,200,0),-1)
            cv2.putText(vis,"LIVE",(50,38),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,0),2)
        d="YOLO-NPU" if _det_npu_ready else "YOLO-CPU"
        p="HRNet-NPU" if _npu_ready else "RTMPose-CPU"
        cv2.putText(vis,f"{d}+{p}",(vis.shape[1]-280,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,0),2)
        _,jpg=cv2.imencode('.jpg',vis,[cv2.IMWRITE_JPEG_QUALITY,70])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n'
        time.sleep(0.033)

# ── ROUTES ────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({"status":"ok","board":"kneo-pi","npu":_npu_ready,
                    "det_npu":_det_npu_ready,
                    "mode":f"{'YOLO-NPU' if _det_npu_ready else 'YOLO-CPU'}+{'HRNet-NPU' if _npu_ready else 'RTMPose-CPU'}"})

@app.route('/stream')
def stream():
    return Response(generate_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with _lock:
        s=dict(_status); s['recording']=_recording; s['rec_frames']=_rec_frames
    s['npu']=_npu_ready; s['det_npu']=_det_npu_ready; return jsonify(s)

@app.route('/record/start', methods=['POST'])
def record_start():
    global _recording,_writer,_rec_path,_rec_frames,_rec_start
    with _lock:
        if _recording: return jsonify({"error":"Already recording"}),400
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        ts=time.strftime("%Y%m%d_%H%M%S")
        _rec_path=os.path.join(OUTPUTS_DIR, f"swing_{ts}.mp4")
        _writer=cv2.VideoWriter(_rec_path,cv2.VideoWriter_fourcc(*"mp4v"),30.0,(640,480))
        _recording=True; _rec_frames=0; _rec_start=time.time()
    return jsonify({"status":"recording","path":_rec_path})

@app.route('/record/stop', methods=['POST'])
def record_stop():
    global _recording,_writer,_rec_frames
    with _lock:
        if not _recording: return jsonify({"error":"Not recording"}),400
        _recording=False
        if _writer: _writer.release(); _writer=None
        path=_rec_path; nf=_rec_frames; el=time.time()-_rec_start
    return jsonify({"status":"saved","path":path,"frames":nf,"duration":round(el,1)})

# ── VIDEO PROCESSING ──────────────────────────────────────────────────
def process_video_task(video_path, output_video, output_json):
    global _status
    try:
        from process_golf_pose import SwingPhaseDetector, KeypointSmoother, draw_hud, draw_phase_legend
        from rtmlib import draw_skeleton
        d="YOLO-NPU" if _det_npu_ready else "YOLO-CPU"
        p="HRNet-NPU" if _npu_ready else "RTMPose-CPU"
        mode=f"{d}+{p}"
        print(f"[Process] Starting ({mode})")

        with _lock: _status={"state":"processing","message":"Slowing video...","progress":2}
        proc_path=slowdown_video(video_path,factor=SLOWDOWN)
        with _lock: _status={"state":"processing","message":f"Processing ({mode})...","progress":5}

        cap=cv2.VideoCapture(proc_path)
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in=cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer=cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc(*"mp4v"),fps_in,(fw,fh))
        detector=SwingPhaseDetector(window=12)
        smoother=KeypointSmoother(alpha=0.35,window=6)
        frame_id=0; all_frames=[]; t_start=time.time()
        t_det=0.0; t_pose=0.0; _last_bbox=None

        while True:
            ok,frame=cap.read()
            if not ok: break

            if frame_id % DET_EVERY == 0:
                t0=time.time()
                bboxes=detect_persons(frame)
                t_det+=time.time()-t0
                if bboxes:
                    arr=np.array(bboxes)
                    areas=[(b[2]-b[0])*(b[3]-b[1]) for b in arr]
                    _last_bbox=arr[int(np.argmax(areas))]

            main_bbox=_last_bbox
            kpts_raw=None; conf_raw=None
            if main_bbox is not None:
                t0=time.time()
                try:
                    if _npu_ready:
                        kpts_raw,conf_raw=run_hrnet_npu(frame,main_bbox)
                    else:
                        from rtmlib import RTMPose
                        if not hasattr(process_video_task,'_pose_cpu'):
                            process_video_task._pose_cpu=RTMPose(
                                onnx_model=RTMPOSE_CPU_MODEL,
                                model_input_size=(192,256),
                                backend='onnxruntime',device='cpu')
                        kp_arr,sc_arr=process_video_task._pose_cpu(frame,main_bbox[np.newaxis])
                        kpts_raw=kp_arr[0]; conf_raw=sc_arr[0]
                except Exception as e:
                    print(f"Pose err frame {frame_id}: {e}")
                t_pose+=time.time()-t0

            if kpts_raw is not None:
                kpts,conf=smoother.update(kpts_raw,conf_raw,SCORE_THR)
            else:
                if smoother.smooth is not None:
                    kpts=smoother.smooth.copy(); conf=smoother.s_conf.copy()*0.7
                else:
                    kpts=None; conf=None

            phase='Address'
            if kpts is not None and conf is not None:
                phase=detector.update(frame_id,kpts,conf,SCORE_THR,fh)

            if kpts is not None and conf is not None:
                vis=draw_skeleton(frame.copy(),kpts[np.newaxis],conf[np.newaxis],kpt_thr=SCORE_THR)
                detected=int(np.sum(conf>=SCORE_THR))
            else:
                vis=frame.copy(); detected=0

            avg_fps=frame_id/max(time.time()-t_start,1e-6)
            vis=draw_hud(vis,phase,frame_id,total,avg_fps,detected)
            vis=draw_phase_legend(vis,phase)
            writer.write(vis)

            fkpts=[]
            if kpts is not None and conf is not None:
                for i in range(17):
                    fkpts.append({"x":round(float(kpts[i,0]),2),"y":round(float(kpts[i,1]),2),
                                  "conf":round(float(conf[i]),4),"valid":bool(conf[i]>=SCORE_THR)})
            all_frames.append({"frame_id":frame_id,"phase":phase,"keypoints":fkpts})
            frame_id+=1

            if frame_id % 10 == 0:
                ad=t_det/max(frame_id//DET_EVERY,1)*1000
                ap=t_pose/max(frame_id,1)*1000
                with _lock:
                    _status={"state":"processing",
                             "message":f"Frame {frame_id}/{total} | Det:{ad:.0f}ms Pose:{ap:.0f}ms",
                             "progress":int(5+90*frame_id/max(total,1))}

        cap.release(); writer.release()
        if proc_path!=video_path and os.path.exists(proc_path): os.remove(proc_path)

        n_with_kpts=sum(1 for f in all_frames if len(f['keypoints'])==17)
        fps_avg=frame_id/max(time.time()-t_start,1e-6)
        print(f"[Process] Done! {n_with_kpts}/{frame_id} frames | {fps_avg:.1f}FPS")

        with open(output_json,'w') as f:
            json.dump({"total_frames":frame_id,"frames":all_frames,"mode":mode},f)

        with _lock:
            _status={"state":"done",
                     "message":f"Done! {n_with_kpts}/{frame_id} frames | {fps_avg:.1f}FPS",
                     "progress":100,"video":output_video,"json":output_json,
                     "video_filename":os.path.basename(output_video)}

    except Exception as e:
        import traceback; traceback.print_exc()
        with _lock: _status={"state":"error","message":str(e),"progress":0}

@app.route('/process',methods=['POST'])
def process():
    global _status
    data=request.get_json() or {}
    vp=data.get('video_path',_rec_path)
    if not vp or not os.path.exists(vp):
        return jsonify({"error":f"Video not found: {vp}"}),400
    ts=time.strftime("%Y%m%d_%H%M%S")
    ov=os.path.join(OUTPUTS_DIR, f"pose_{ts}.mp4"); oj=os.path.join(OUTPUTS_DIR, f"phases_{ts}.json")
    with _lock: _status={"state":"starting","message":"Starting...","progress":0}
    threading.Thread(target=process_video_task,args=(vp,ov,oj),daemon=True).start()
    return jsonify({"status":"started","output_video":ov,"output_json":oj})

@app.route('/files/json')
def get_json():
    with _lock: path=_status.get('json')
    if not path or not os.path.exists(path): return jsonify({"error":"No JSON ready"}),404
    with open(path) as f: return jsonify(json.load(f))

@app.route('/upload/video',methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({"error":"No video"}),400
    f=request.files['video']; os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ext=os.path.splitext(f.filename)[1] or '.mp4'
    path=os.path.join(OUTPUTS_DIR, f"upload_{time.strftime('%Y%m%d_%H%M%S')}{ext}")
    f.save(path)
    return jsonify({"status":"saved","path":path,"size_mb":round(os.path.getsize(path)/1024/1024,1)})

def _serve_h264(path):
    h264=path.replace('.mp4','_h264.mp4')
    if not os.path.exists(h264):
        try:
            subprocess.run(['ffmpeg','-y','-i',path,'-vcodec','libx264',
                '-preset','ultrafast','-crf','28','-movflags','+faststart','-an',h264],
                check=True,capture_output=True,timeout=300)
        except Exception as e:
            print(f"ffmpeg: {e}"); return send_file(path,mimetype='video/mp4')
    return send_file(h264,mimetype='video/mp4')

@app.route('/video/processed')
def serve_processed_video():
    with _lock: path=_status.get('video')
    if not path or not os.path.exists(path): return jsonify({"error":"No video ready"}),404
    return _serve_h264(path)

@app.route('/video/file/<filename>')
def serve_video_file(filename):
    path=os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path): return jsonify({"error":"Not found"}),404
    return _serve_h264(path)

@app.route('/list/videos')
def list_videos():
    import glob
    vs=sorted(glob.glob(os.path.join(OUTPUTS_DIR, '*.mp4')),reverse=True)
    vs=[v for v in vs if '_h264' not in v and '_slow' not in v]
    return jsonify([{"name":os.path.basename(v),"path":v,
                     "size_mb":round(os.path.getsize(v)/1024/1024,1)} for v in vs])

if __name__=='__main__':
    print("="*60)
    print("  Board API — Dual NPU: YOLOv7-tiny + HRNet Golf")
    print(f"  Slowdown:{SLOWDOWN}x | ScoreThr:{SCORE_THR} | DetEvery:{DET_EVERY}")
    print("="*60)
    init_npu()
    threading.Thread(target=camera_thread,daemon=True).start()
    time.sleep(0.5)
    print(f"\n[Board API] Ready on port 5000")
    print(f"[Board API] Det:  {'YOLO-NPU (~10ms)' if _det_npu_ready else 'YOLO-CPU fallback'}")
    print(f"[Board API] Pose: {'HRNet-NPU (~64ms)' if _npu_ready else 'RTMPose-CPU fallback'}")
    app.run(host='0.0.0.0',port=5000,threaded=True)
