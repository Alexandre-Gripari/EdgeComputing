import cv2
import numpy as np
import tensorflow as tf
import os
import glob
import csv
from tqdm import tqdm

MODELS = {
    "Original": "model/person_det_160x128.tflite",
    "Tiny": "model/student_tiny_quant.tflite",
    "Medium": "model/student_medium_quant.tflite",
    "Large": "model/student_large_quant.tflite",
}

VAL_IMG_DIR = "./coco_person_mini/images/train"
VAL_LABEL_DIR = "./coco_person_mini/labels/train"
CSV_OUTPUT_FILE = "data/resultats_models.csv"

# Model parameters
INPUT_WIDTH = 160
INPUT_HEIGHT = 128
CONF_THRESHOLD = 0.3
IOU_NMS_THRESHOLD = 0.3

# Output indices
IDX_OBJ = 0
IDX_TX = 1
IDX_TY = 2
IDX_TW = 3
IDX_TH = 4


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area


def load_ground_truth(txt_path, img_w, img_h):
    gt_boxes = []
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                nx, ny, nw, nh = map(float, parts[1:5])

                w = nw * img_w
                h = nh * img_h
                x_center = nx * img_w
                y_center = ny * img_h

                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2

                gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes


def run_inference(interpreter, image_path):
    original_image = cv2.imread(image_path)
    if original_image is None:
        return [], 0, 0

    orig_h, orig_w = original_image.shape[:2]
    resized_image = cv2.resize(original_image, (INPUT_WIDTH, INPUT_HEIGHT))
    input_data = np.expand_dims(resized_image, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    scale, zero_point = output_details[0]["quantization"]
    if scale == 0.0:
        scale, zero_point = 1.0, 0

    if input_details[0]["dtype"] == np.float32:
        input_data = np.float32(input_data)
    else:
        input_data = np.int8(input_data - 128)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])[0]
    output_float = (raw_output.astype(np.float32) - zero_point) * scale

    rows, cols, _ = output_float.shape
    boxes = []
    confidences = []

    for y in range(rows):
        for x in range(cols):
            score = sigmoid(output_float[y, x, IDX_OBJ])
            if score > CONF_THRESHOLD:
                bcx = (np.tanh(output_float[y, x, IDX_TX]) + x) / cols
                bcy = (np.tanh(output_float[y, x, IDX_TY]) + y) / rows
                bw = sigmoid(output_float[y, x, IDX_TW])
                bh = sigmoid(output_float[y, x, IDX_TH])

                center_x = bcx * orig_w
                center_y = bcy * orig_h
                width = bw * orig_w
                height = bh * orig_h

                x_min = int(center_x - width / 2)
                y_min = int(center_y - height / 2)

                boxes.append([x_min, y_min, int(width), int(height)])
                confidences.append(float(score))

    final_dets = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONF_THRESHOLD, IOU_NMS_THRESHOLD
        )
        for i in indices:
            idx = i if isinstance(i, (int, np.integer)) else i[0]
            box = boxes[idx]
            x, y, w, h = box
            final_dets.append([x, y, x + w, y + h, confidences[idx]])

    return final_dets, orig_w, orig_h


def calculate_ap(all_gt, all_preds, iou_threshold=0.5):
    true_positives = []
    scores = []
    num_gt_total = 0

    for img_id in all_gt:
        gts = all_gt[img_id]
        preds = all_preds.get(img_id, [])

        num_gt_total += len(gts)
        preds.sort(key=lambda x: x[4], reverse=True)

        gt_matched = [False] * len(gts)

        for p in preds:
            p_box = p[:4]
            p_score = p[4]
            scores.append(p_score)

            best_iou = 0
            best_gt_idx = -1

            for i, gt_box in enumerate(gts):
                iou = compute_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    true_positives.append(1)
                    gt_matched[best_gt_idx] = True
                else:
                    true_positives.append(0)
            else:
                true_positives.append(0)

    if num_gt_total == 0:
        return 0.0

    combined = list(zip(scores, true_positives))
    combined.sort(key=lambda x: x[0], reverse=True)

    tp_cumsum = np.cumsum([c[1] for c in combined])
    fp_cumsum = np.cumsum([1 - c[1] for c in combined])

    recalls = tp_cumsum / num_gt_total
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


print(f"Dataset images: {VAL_IMG_DIR}")
print(f"Dataset labels: {VAL_LABEL_DIR}")
img_files = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))
print(f"Number of images found: {len(img_files)}")

results = {}

for model_name, model_path in MODELS.items():
    print(f"\n--- Evaluating {model_name} ---")

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        continue

    all_ground_truths = {}
    all_predictions = {}

    for img_path in tqdm(img_files):

        preds, w, h = run_inference(interpreter, img_path)
        all_predictions[img_path] = preds

        basename = os.path.basename(img_path)
        txt_name = os.path.splitext(basename)[0] + ".txt"
        txt_path = os.path.join(VAL_LABEL_DIR, txt_name)

        gts = load_ground_truth(txt_path, w, h)
        all_ground_truths[img_path] = gts

    map_50 = calculate_ap(all_ground_truths, all_predictions, iou_threshold=0.5)

    aps = []
    print("Calculating mAP 50-95...")
    for iou in np.arange(0.5, 0.96, 0.05):
        ap = calculate_ap(all_ground_truths, all_predictions, iou_threshold=iou)
        aps.append(ap)
    map_50_95 = np.mean(aps)

    results[model_name] = {"mAP-50": map_50, "mAP-50-95": map_50_95}

print("\n" + "=" * 45)
print(f"{'MODEL':<15} | {'mAP@50':<10} | {'mAP@50:95':<10}")
print("-" * 45)

csv_data = []

for name, metrics in results.items():
    print(f"{name:<15} | {metrics['mAP-50']:.4f}     | {metrics['mAP-50-95']:.4f}")

    csv_data.append([name, metrics["mAP-50"], metrics["mAP-50-95"]])

print("=" * 45)

try:
    with open(CSV_OUTPUT_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "mAP@50", "mAP@50-95"])
        writer.writerows(csv_data)
    print(f"\n[SUCCESS] Results have been saved to '{CSV_OUTPUT_FILE}'")
except Exception as e:
    print(f"\n[ERROR] Unable to save CSV: {e}")
