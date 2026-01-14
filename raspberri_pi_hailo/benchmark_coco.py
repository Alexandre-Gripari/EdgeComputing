#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import time
import glob
import json
import sys
import threading
from collections import deque
from functools import lru_cache
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HEF_FILE = "yolov11n.hef"
IMG_DIR = "coco128/images/train2017/"
LABEL_DIR = "coco128/labels/train2017/"
GT_JSON = "coco128_ground_truth.json"
PRED_JSON = "coco128_predictions.json"

YOLO_TO_COCO_MAPPING = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


@lru_cache(maxsize=None)
def get_img_size(path):
    with Image.open(path) as im:
        return im.size


def load_image_metadata():
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    metadata = []
    for idx, img_path in enumerate(img_files, 1):
        w, h = get_img_size(img_path)
        metadata.append({
            "path": img_path,
            "id": idx,
            "width": w,
            "height": h
        })
    return metadata


def generate_ground_truth(image_meta):
    print(f"[1/3] Generating Ground Truth JSON...")
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in YOLO_TO_COCO_MAPPING]
    }
    ann_id = 0

    for meta in image_meta:
        filename = os.path.basename(meta["path"])
        dataset["images"].append({
            "id": meta["id"], "file_name": filename, "width": meta["width"], "height": meta["height"]
        })

        txt_name = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(LABEL_DIR, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_idx = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])

                    x = (cx - bw / 2) * meta["width"]
                    y = (cy - bh / 2) * meta["height"]
                    abs_w = bw * meta["width"]
                    abs_h = bh * meta["height"]

                    if 0 <= cls_idx < len(YOLO_TO_COCO_MAPPING):
                        dataset["annotations"].append({
                            "id": ann_id,
                            "image_id": meta["id"],
                            "category_id": YOLO_TO_COCO_MAPPING[cls_idx],
                            "bbox": [x, y, abs_w, abs_h],
                            "area": abs_w * abs_h,
                            "iscrowd": 0
                        })
                        ann_id += 1
                        
    with open(GT_JSON, 'w') as f:
        json.dump(dataset, f)
image_queue = deque()
results = []


def feeder_thread(appsrc, image_meta):
    print(f"   -> Feeder started. Processing {len(image_meta)} images...")

    for meta in image_meta:
        with open(meta["path"], 'rb') as f:
            data = f.read()

        image_queue.append((meta["id"], meta["width"], meta["height"]))

        buf = Gst.Buffer.new_wrapped(data)
        appsrc.emit("push-buffer", buf)

    print("   -> All images pushed. Sending EOS...")
    appsrc.emit("end-of-stream")

def probe_cb(pad, info):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    try:
        img_id, w_orig, h_orig = image_queue.popleft()
    except IndexError:
        print("Error: Received more frames than sent!")
        return Gst.PadProbeReturn.OK

    import hailo
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    for det in detections:
        bbox = det.get_bbox()
        cls_id_raw = det.get_class_id()
        
        coco_id = 1
        if 1 <= cls_id_raw <= 80:
            idx = cls_id_raw - 1
            if idx < len(YOLO_TO_COCO_MAPPING):
                coco_id = YOLO_TO_COCO_MAPPING[idx]
        elif cls_id_raw == 0:
            coco_id = 1 

        results.append({
            "image_id": img_id,
            "category_id": coco_id,
            "bbox": [
                bbox.xmin() * w_orig,
                bbox.ymin() * h_orig,
                (bbox.xmax() - bbox.xmin()) * w_orig,
                (bbox.ymax() - bbox.ymin()) * h_orig
            ],
            "score": det.get_confidence()
        })
        
    return Gst.PadProbeReturn.OK


def run_inference_optimized(image_meta):
    print(f"[2/3] Running Inference (Optimized Batch Mode)...")
    start_time = time.monotonic()
    results.clear()
    image_queue.clear()
    
    Gst.init(None)

    pipeline_str = (
        "appsrc name=source emit-signals=True format=time ! "
        "jpegdec ! " 
        "videoscale ! video/x-raw, width=640, height=640 ! "
        "videoconvert ! video/x-raw, format=RGB ! "
        f"hailonet hef-path={HEF_FILE} batch-size=1 "
        "nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
        "hailofilter so-path=/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so qos=false ! "
        "fakesink name=sink"
    )
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"Pipeline Error: {e}")
        sys.exit(1)

    sink = pipeline.get_by_name("sink")
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe_cb)
    
    appsrc = pipeline.get_by_name("source")
    
    pipeline.set_state(Gst.State.PLAYING)
    
    t = threading.Thread(target=feeder_thread, args=(appsrc, image_meta))
    t.start()
    
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    
    if msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f"Error: {err}, {debug}")
        pipeline.set_state(Gst.State.NULL)
        t.join()
        return
    
    pipeline.set_state(Gst.State.NULL)
    t.join()

    end_time = time.monotonic()
    duration = end_time - start_time  # (Il faut définir start_time au début de la fonction)
    fps_app = len(results) / duration
    print(f"   -> Temps Total : {duration:.2f}s | FPS Applicatif : {fps_app:.2f} FPS")
    
    with open(PRED_JSON, 'w') as f:
        json.dump(results, f, separators=(",", ":"))
    print(f"   -> Inference done. {len(results)} detections saved.")

def evaluate():
    print("[3/3] Calculating mAP...")
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        cocoGt = COCO(GT_JSON)
        cocoDt = cocoGt.loadRes(PRED_JSON)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        sys.stdout = old_stdout
        cocoEval.summarize()
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Evaluation Error: {e}")

def main():
    image_meta = load_image_metadata()
    generate_ground_truth(image_meta)
    run_inference_optimized(image_meta)
    evaluate()


if __name__ == "__main__":
    main()