import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

CLASSES_TO_TRACK = {"person"}  # person only for clean single box result

class Track:
    _next_id = 0
    R_SENSOR1 = np.array([[10, 0], [0, 10]])
    R_SENSOR2 = np.array([[80, 0], [0, 80]])

    def __init__(self, center_x, center_y, w, h, label):
        self.id = Track._next_id
        Track._next_id += 1

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.kf.P *= 1000
        self.kf.Q = np.eye(4) * 0.01
        self.kf.x = np.array([center_x, center_y, 0, 0])

        self.w = w
        self.h = h
        self.label = label
        self.missed_frames = 0
        self.hits = 1
        self.confirmed = False
        self.last_sensor2_point = (center_x, center_y)

    def predict(self):
        self.kf.predict()

    def update(self, center_x, center_y, w, h, label):
        self.kf.R = Track.R_SENSOR1
        self.kf.update(np.array([center_x, center_y]))

        noise_x = np.random.randn() * 15
        noise_y = np.random.randn() * 15
        sensor2_x = center_x + noise_x
        sensor2_y = center_y + noise_y
        self.last_sensor2_point = (int(sensor2_x), int(sensor2_y))

        self.kf.R = Track.R_SENSOR2
        self.kf.update(np.array([sensor2_x, sensor2_y]))

        self.w = w
        self.h = h
        self.label = label
        self.missed_frames = 0
        self.hits += 1

    def get_box(self):
        cx, cy = self.kf.x[0], self.kf.x[1]
        x = int(cx - self.w / 2)
        y = int(cy - self.h / 2)
        return (x, y, int(self.w), int(self.h))


def compute_iou(boxA, boxB):
    ax1, ay1, aw, ah = boxA
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = boxB
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = aw * ah
    areaB = bw * bh
    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


# settings
MAX_MISSED_FRAMES = 5
IOU_MATCH_THRESHOLD = 0.2
CONFIRM_HITS = 2             
YOLO_CONFIDENCE_THRESHOLD = 0.5  

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (0, 128, 255)
]

# setup 
cap = cv2.VideoCapture(0)
model = YOLO("yolov8s.pt")   

tracks = []

print("Running YOLO multi-object Kalman tracking. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect using YOLO
    results = model(frame, verbose=False, iou=0.4, conf=0.5)[0]  # NMS tuning

    detections = []
    for box in results.boxes:
        confidence = float(box.conf[0])
        if confidence < YOLO_CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        label = model.names[class_id]

        if label not in CLASSES_TO_TRACK:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        detections.append((x, y, w, h, label))

    # predict
    for track in tracks:
        track.predict()

    # match
    if len(tracks) > 0 and len(detections) > 0:
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(tracks):
            predicted_box = track.get_box()
            for d_idx, det in enumerate(detections):
                det_box = det[:4]
                iou = compute_iou(predicted_box, det_box)
                cost_matrix[t_idx, d_idx] = 1 - iou

        track_indices, det_indices = linear_sum_assignment(cost_matrix)
    else:
        track_indices, det_indices = np.array([], dtype=int), np.array([], dtype=int)

    matched_tracks = set()
    matched_detections = set()

    for t_idx, d_idx in zip(track_indices, det_indices):
        iou = 1 - cost_matrix[t_idx, d_idx]
        if iou >= IOU_MATCH_THRESHOLD:
            x, y, w, h, label = detections[d_idx]
            center_x = x + w // 2
            center_y = y + h // 2
            tracks[t_idx].update(center_x, center_y, w, h, label)
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)

    # unmatched tracks
    for t_idx, track in enumerate(tracks):
        if t_idx not in matched_tracks:
            track.missed_frames += 1

    # remove stale tracks
    tracks = [t for t in tracks if t.missed_frames <= MAX_MISSED_FRAMES]

    # new tracks for unmatched detections
    for d_idx, det in enumerate(detections):
        if d_idx not in matched_detections:
            x, y, w, h, label = det
            center_x = x + w // 2
            center_y = y + h // 2
            tracks.append(Track(center_x, center_y, w, h, label))

    # confirm tracks
    for track in tracks:
        if not track.confirmed and track.hits >= CONFIRM_HITS:
            track.confirmed = True

    # draw
    confirmed_count = 0
    for track in tracks:
        if not track.confirmed:
            continue

        confirmed_count += 1
        x, y, w, h = track.get_box()
        color = COLORS[track.id % len(COLORS)]

        thickness = 2 if track.missed_frames == 0 else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(frame, track.last_sensor2_point, 4, (0, 165, 255), -1)

        label_text = f"ID {track.id} {track.label}"
        if track.missed_frames > 0:
            label_text += f" (coasting {track.missed_frames})"
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Active Tracks: {confirmed_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Orange dot = simulated Sensor 2 reading", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    cv2.imshow("YOLO Multi-Object Kalman Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
