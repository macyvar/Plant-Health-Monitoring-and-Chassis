#!/usr/bin/env python3
"""
Two-Stage Leaf Classifier (Raspberry Pi friendly)
-------------------------------------------------
Stage A (Gate):   Leaf vs Not-Leaf
  - Option 1: One-class (ONLY leaf images) -> OneClassSVM  (leaf_gate_oneclass.pkl)
  - Option 2: Two-class  (leaf vs not-leaf) -> RandomForest (leaf_gate.pkl)

Stage B (Health): Healthy vs Diseased -> RandomForest (model.pkl)

Features:
- HSV color histograms
- green_ratio (fraction of green-ish pixels)
- edge_ratio  (Canny edges per pixel)

CLI:
  # Prepare PlantVillage into leaflab/{healthy,diseased,leaves_all}
  python leaf_pipeline.py --prepare-plantvillage \
    --pv-root  /path/to/PlantVillage \
    --out-root /path/to/leaflab [--limit-per-class N]

  # Train one-class gate (leaf only)
  python leaf_pipeline.py --train-gate-oneclass \
    --gate-leaf-dir /path/to/leaflab/leaves_all \
    --gate-nu 0.05 --gate-gamma scale

  # Train health classifier
  python leaf_pipeline.py --train-health \
    --healthy-dir  /path/to/leaflab/healthy \
    --diseased-dir /path/to/leaflab/diseased

  # Classify an existing image
  python leaf_pipeline.py --image /path/to/img.jpg

  # Run camera once (Pi)
  python leaf_pipeline.py
"""

import os, time, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# one-class
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# Config / Paths

CAPTURE_SAVE_DIR = Path.home() / "leaf_captures"
CAPTURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

GATE_MODEL_PATH        = "leaf_gate.pkl"            # 2-class gate
ONECLASS_GATE_PATH     = "leaf_gate_oneclass.pkl"   # 1-class gate
HEALTH_MODEL_PATH      = "model.pkl"                # healthy/diseased


# Features

def extract_hsv_hist(bgr):
    """50/60/60 bins for H/S/V, L1-normalized."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten().astype(np.float32)
    s = hist.sum()
    if s > 0: hist /= s
    return hist

def green_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([30, 40, 30]),  np.array([85, 255, 255])
    lower2, upper2 = np.array([25, 20, 20]),  np.array([100, 255, 255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    return mask

def green_ratio(bgr):
    m = green_mask(bgr)
    return float(cv2.countNonZero(m)) / float(bgr.shape[0] * bgr.shape[1])

def edge_ratio(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5,5), 0), 50, 150)
    return float(cv2.countNonZero(edges)) / float(bgr.shape[0] * bgr.shape[1])

def make_feature_vector(bgr):
    hist = extract_hsv_hist(bgr)
    g = np.array([green_ratio(bgr)], dtype=np.float32)
    e = np.array([edge_ratio(bgr)], dtype=np.float32)
    return np.concatenate([hist, g, e], axis=0)


# Data loading

def load_images_from_folder(folder, max_per_class=None, recursive=True):
    """Return image paths from a folder; recursively by default."""
    paths = []
    if not folder or not os.path.isdir(folder):
        return paths
    exts = (".jpg", ".jpeg", ".png")
    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(exts):
                    paths.append(os.path.join(root, name))
                    if max_per_class and len(paths) >= max_per_class:
                        return paths
    else:
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if os.path.isfile(p) and name.lower().endswith(exts):
                paths.append(p)
                if max_per_class and len(paths) >= max_per_class:
                    break
    return paths

def load_xy_from_dirs(pos_dir, neg_dir, max_per_class=None):
    X, y = [], []
    for p in load_images_from_folder(pos_dir, max_per_class=max_per_class):
        img = cv2.imread(p);  
        if img is None: continue
        X.append(make_feature_vector(img)); y.append(1)
    for p in load_images_from_folder(neg_dir, max_per_class=max_per_class):
        img = cv2.imread(p);  
        if img is None: continue
        X.append(make_feature_vector(img)); y.append(0)
    return np.array(X), np.array(y)

def load_xy_health(healthy_dir, diseased_dir, max_per_class=None):
    X, y = [], []
    for p in load_images_from_folder(healthy_dir, max_per_class=max_per_class):
        img = cv2.imread(p);  
        if img is None: continue
        X.append(make_feature_vector(img)); y.append(0)  # Healthy=0
    for p in load_images_from_folder(diseased_dir, max_per_class=max_per_class):
        img = cv2.imread(p);  
        if img is None: continue
        X.append(make_feature_vector(img)); y.append(1)  # Diseased=1
    return np.array(X), np.array(y)


# Prepare PlantVillage

def prepare_plantvillage(pv_root, out_root, limit=None):
    pv_root = os.path.expanduser(pv_root)
    out_root = Path(os.path.expanduser(out_root))
    healthy_dir = out_root / "healthy"
    diseased_dir = out_root / "diseased"
    leaves_all = out_root / "leaves_all"
    for p in (healthy_dir, diseased_dir, leaves_all):
        if p.exists(): 
            # keep previous copies if present; remove to refresh
            pass
        p.mkdir(parents=True, exist_ok=True)

    healthy_count = diseased_count = 0
    for root, _, files in os.walk(pv_root):
        rp = Path(root)
        is_healthy = "healthy" in rp.as_posix().lower()
        for fn in files:
            if not fn.lower().endswith(('.jpg','.jpeg','.png')): 
                continue
            src = rp / fn
            rel = src.relative_to(pv_root)
            # copy into class tree
            cls_base = healthy_dir if is_healthy else diseased_dir
            (cls_base / rel.parent).mkdir(parents=True, exist_ok=True)
            dest1 = cls_base / rel
            if not dest1.exists():
                dest1.parent.mkdir(parents=True, exist_ok=True)
                try: 
                    import shutil; shutil.copy2(src, dest1)
                except Exception: pass
            # also into leaves_all
            (leaves_all / rel.parent).mkdir(parents=True, exist_ok=True)
            dest2 = leaves_all / rel
            if not dest2.exists():
                try: 
                    import shutil; shutil.copy2(src, dest2)
                except Exception: pass
            if is_healthy: healthy_count += 1
            else: diseased_count += 1
            if limit and (healthy_count + diseased_count) >= limit:
                break

    print("Prepared PlantVillage into:")
    print(f"  Healthy:  {healthy_dir} ({healthy_count} imgs)")
    print(f"  Diseased: {diseased_dir} ({diseased_count} imgs)")
    print(f"  Leaves:   {leaves_all} (sum)")


# Training

def train_gate(leaf_dir, notleaf_dir, out=GATE_MODEL_PATH):
    print(f"Loading gate data...\n  leaf_dir={leaf_dir}\n  notleaf_dir={notleaf_dir}")
    X, y = load_xy_from_dirs(leaf_dir, notleaf_dir)
    if len(X) == 0:
        print("No training images found. Check your paths."); return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    print("Gate Accuracy:", accuracy_score(yte, preds))
    print(classification_report(yte, preds, target_names=["NotLeaf","Leaf"]))
    joblib.dump(clf, out)
    print(f" Saved gate model -> {out}")

def train_gate_oneclass(leaf_dir, out=ONECLASS_GATE_PATH, nu=0.05, gamma="scale"):
    print(" Training ONE-CLASS gate")
    print(f"  leaf_dir={leaf_dir}\n  nu={nu} gamma={gamma}")
    X = []
    for p in load_images_from_folder(leaf_dir):
        img = cv2.imread(p)
        if img is None: continue
        X.append(make_feature_vector(img))
    X = np.array(X)
    if X.size == 0:
        print("No leaf images found."); return
    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                         OneClassSVM(kernel="rbf", nu=nu, gamma=gamma))
    pipe.fit(X)
    joblib.dump(pipe, out)
    print(f"Saved one-class gate -> {out}")

def train_health(healthy_dir, diseased_dir, out=HEALTH_MODEL_PATH):
    print(f"🌿 Loading health data...\n  healthy_dir={healthy_dir}\n  diseased_dir={diseased_dir}")
    X, y = load_xy_health(healthy_dir, diseased_dir)
    if len(X) == 0:
        print("No training images found. Check your paths."); return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    print("Health Accuracy:", accuracy_score(yte, preds))
    print(classification_report(yte, preds, target_names=["Healthy","Diseased"]))
    joblib.dump(clf, out)
    print(f"Saved health model -> {out}")


# Camera capture (Pi)

def get_frame_from_picamera2():
    try:
        from picamera2 import Picamera2
    except Exception:
        return None
    try:
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration())
        cam.start()
        time.sleep(1.5)
        rgb = cam.capture_array()
        cam.stop(); cam.close()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Picamera2 error: {e}")
        return None

def get_frame_from_opencv(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print("OpenCV camera could not open."); return None
    for _ in range(5):
        cap.read(); time.sleep(0.03)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# Inference

def classify_image(bgr, gate_path=GATE_MODEL_PATH, health_path=HEALTH_MODEL_PATH, save=True, show=True):
    # Prefer one-class gate if available
    if os.path.exists(ONECLASS_GATE_PATH):
        gate_path = ONECLASS_GATE_PATH

    if not os.path.exists(gate_path):
        print(f"Missing gate model: {gate_path}"); return None
    if not os.path.exists(health_path):
        print(f"Missing health model: {health_path}"); return None

    gate = joblib.load(gate_path)
    health = joblib.load(health_path)

    x = make_feature_vector(bgr).reshape(1, -1)
    try:
        gate_pred = int(gate.predict(x)[0])  # +1 inlier, -1 outlier
        is_leaf = 1 if gate_pred == 1 else 0
    except Exception:
        is_leaf = int(gate.predict(x)[0])    # two-class fallback

    if not is_leaf:
        label, color = "Not a leaf", (0, 0, 255)
    else:
        pred = int(health.predict(x)[0])  # 0 Healthy, 1 Diseased
        label = "Diseased" if pred else "Healthy"
        color = (0, 0, 255) if pred else (0, 200, 0)

    annotated = bgr.copy()
    cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = CAPTURE_SAVE_DIR / f"classified_{label.replace(' ', '_')}_{ts}.jpg"
    if save:
        cv2.imwrite(str(out_path), annotated)
        print(f"Imaged saved: {out_path}")
    print(f"Result: {label}")

    if show:
        try:
            cv2.imshow("Leaf Two-Stage Classifier", annotated)
            cv2.waitKey(2500); cv2.destroyAllWindows()
        except Exception:
            pass
    return label

def run_camera_once():
    frame = get_frame_from_picamera2()
    if frame is None:
        print("Frame falling back to OpenCV VideoCapture(0)")
        frame = get_frame_from_opencv(0)
    if frame is None:
        print("Could not get a frame from any camera."); return
    classify_image(frame)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Two-stage leaf pipeline (gate + health)")
    # prepare
    p.add_argument("--prepare-plantvillage", action="store_true", help="Prepare PlantVillage into leaflab layout")
    p.add_argument("--pv-root", type=str, help="Path to PlantVillage (folder containing class folders)")
    p.add_argument("--out-root", type=str, help="Output root for leaflab folders")
    p.add_argument("--limit-per-class", type=int, default=None, help="Optional cap per class during prepare")

    # gates
    p.add_argument("--train-gate", action="store_true", help="Train 2-class leaf vs not-leaf gate")
    p.add_argument("--leaf-dir", type=str, help="Leaf images (positive) for 2-class gate")
    p.add_argument("--notleaf-dir", type=str, help="Not-leaf images (negative) for 2-class gate")

    p.add_argument("--train-gate-oneclass", action="store_true", help="Train ONE-CLASS gate (leaf only)")
    p.add_argument("--gate-leaf-dir", type=str, help="Leaf images root for 1-class gate")
    p.add_argument("--gate-nu", type=float, default=0.05, help="nu for OneClassSVM (outlier fraction)")
    p.add_argument("--gate-gamma", type=str, default="scale", help="gamma for OneClassSVM (e.g., 'scale' or float)")

    # health
    p.add_argument("--train-health", action="store_true", help="Train Healthy vs Diseased model")
    p.add_argument("--healthy-dir", type=str, help="Folder with healthy images")
    p.add_argument("--diseased-dir", type=str, help="Folder with diseased images")

    # inference
    p.add_argument("--image", type=str, help="Classify a specific image")
    p.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    p.add_argument("--no-save", action="store_true", help="Do not save annotated output")
    return p.parse_args()

def main():
    args = parse_args()

    if args.prepare_plantvillage:
        if not args.pv_root or not args.out_root:
            print("No: --prepare-plantvillage requires --pv-root and --out-root"); return
        prepare_plantvillage(args.pv_root, args.out_root, args.limit_per_class); return

    if args.train_gate_oneclass:
        if not args.gate_leaf_dir:
            print("No: --train-gate-oneclass requires --gate-leaf-dir"); return
        nu = float(args.gate_nu)
        gamma = args.gate_gamma if args.gate_gamma == "scale" else float(args.gate_gamma)
        train_gate_oneclass(args.gate_leaf_dir, ONECLASS_GATE_PATH, nu=nu, gamma=gamma); return

    if args.train_gate:
        if not args.leaf_dir or not args.notleaf_dir:
            print("No: --train-gate requires --leaf-dir and --notleaf-dir"); return
        train_gate(args.leaf_dir, args.notleaf_dir); return

    if args.train_health:
        if not args.healthy_dir or not args.diseased_dir:
            print("No: --train-health requires --healthy-dir and --diseased-dir"); return
        train_health(args.healthy_dir, args.diseased_dir); return

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}"); return
        classify_image(img, save=not args.no_save, show=not args.no_show)
    else:
        run_camera_once()

if __name__ == "__main__":
    main()
