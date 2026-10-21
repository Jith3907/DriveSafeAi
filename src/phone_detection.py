"""
Phone Detection Module for DriveSafe AI.
Uses pretrained Ultralytics YOLOv8n to identify handheld mobile devices in real time.

Part C fixes:
 - Voting window: keep last N YOLO results, trigger when >= votes_needed are True
 - Hold time: keep alert True for PHONE_HOLD_TIME_SEC after last positive detection
 - Lower confidence to 0.28
 - Run YOLO every 2nd frame (YOLO_FRAME_INTERVAL)
 - Ignore detections smaller than 1% of frame area
 - Rich debug info returned
"""

import time
from collections import deque
from ultralytics import YOLO
from src import config


class PhoneDetector:
    def __init__(self):
        # Load lightweight pretrained COCO weights
        self.model = YOLO(config.YOLO_MODEL_NAME)

        # Voting window (last N YOLO results)
        voting_window = int(getattr(config, "PHONE_VOTING_WINDOW", 6))
        self.yolo_results_window: deque = deque(maxlen=voting_window)

        # Hold time
        self.last_positive_time: float = 0.0
        self.phone_in_use: bool = False

        # Frame skipping
        self._frame_counter: int = 0
        self._last_yolo_found: bool = False
        self._last_confidence: float = 0.0
        self._last_box_area_ratio: float = 0.0

        # For YOLO run tracking
        self._yolo_run_count: int = 0
        self._avg_fps_tracker: list = []
        self._fps_start_time = time.time()

    def detect(self, frame):
        """
        Run phone detection on the given frame.

        Returns a dict with detection status, debug info, etc.
        """
        now = time.time()
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        min_area_ratio = float(getattr(config, "PHONE_MIN_AREA_RATIO", 0.01))

        yolo_frame_interval = int(getattr(config, "YOLO_FRAME_INTERVAL", 2))
        voting_window = int(getattr(config, "PHONE_VOTING_WINDOW", 6))
        votes_needed = int(getattr(config, "PHONE_VOTES_NEEDED", 3))
        hold_time = float(getattr(config, "PHONE_HOLD_TIME_SEC", 1.5))
        conf_thresh = float(getattr(config, "PHONE_CONF_THRESHOLD", 0.28))
        imgsz = int(getattr(config, "PHONE_IMGSZ", 480))

        # Resize voting window if config changed
        if self.yolo_results_window.maxlen != voting_window:
            old = list(self.yolo_results_window)
            self.yolo_results_window = deque(old, maxlen=voting_window)

        self._frame_counter += 1
        phone_found_this_frame = False
        detections = []
        alert_reason = ""

        run_yolo_now = (self._frame_counter % yolo_frame_interval == 0)

        if run_yolo_now:
            self._yolo_run_count += 1

            results = self.model.predict(
                source=frame,
                classes=[config.PHONE_CLASS_ID],
                conf=conf_thresh,
                imgsz=imgsz,
                verbose=False
            )

            self._last_confidence = 0.0
            self._last_box_area_ratio = 0.0
            phone_in_frame = False

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    c = float(box.conf[0].cpu().numpy())

                    x1, y1, x2, y2 = xyxy
                    box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
                    area_ratio = box_area / frame_area if frame_area > 0 else 0.0

                    # Filter: ignore tiny boxes (< 1% of frame area)
                    if area_ratio < min_area_ratio:
                        continue

                    detections.append({
                        "bbox": xyxy,
                        "confidence": c,
                        "area_ratio": area_ratio
                    })
                    if c > self._last_confidence:
                        self._last_confidence = c
                        self._last_box_area_ratio = area_ratio
                    phone_in_frame = True

            phone_found_this_frame = phone_in_frame
            self._last_yolo_found = phone_in_frame

            # Update voting window
            self.yolo_results_window.append(phone_in_frame)

            if phone_in_frame:
                self.last_positive_time = now
        else:
            # Use last YOLO result (no new inference)
            phone_found_this_frame = self._last_yolo_found

        # === Voting decision ===
        true_votes = sum(1 for v in self.yolo_results_window if v)
        vote_str = f"{true_votes}/{len(self.yolo_results_window)}"

        votes_met = (true_votes >= votes_needed and len(self.yolo_results_window) >= 1)

        # === Hold time logic ===
        hold_remaining = 0.0
        if votes_met:
            self.phone_in_use = True
            alert_reason = f"Vote threshold met ({vote_str} >= {votes_needed}/{voting_window})"
        elif self.phone_in_use:
            # Check hold timer
            elapsed_since_last_positive = now - self.last_positive_time
            hold_remaining = max(0.0, hold_time - elapsed_since_last_positive)
            if elapsed_since_last_positive < hold_time:
                # Still within hold window
                alert_reason = f"Hold active ({hold_remaining:.1f}s remaining after last positive)"
            else:
                self.phone_in_use = False
                alert_reason = f"Hold expired ({elapsed_since_last_positive:.1f}s > {hold_time:.1f}s): alert cleared"
        else:
            alert_reason = f"No phone detected (votes {vote_str} < {votes_needed})"

        severity = config.SEVERITY_CRITICAL if self.phone_in_use else config.SEVERITY_NORMAL
        alert_msg = "CRITICAL: PHONE USAGE DETECTED!" if self.phone_in_use else ""

        return {
            "detected": phone_found_this_frame,
            "detections": detections,
            "phone_in_use": self.phone_in_use,
            # backward compat
            "consec_frames": true_votes,
            "severity": severity,
            "alert": alert_msg,
            # Debug info
            "last_confidence": round(self._last_confidence, 3),
            "last_box_area_ratio": round(self._last_box_area_ratio, 4),
            "vote_str": vote_str,
            "true_votes": true_votes,
            "votes_needed": votes_needed,
            "voting_window": voting_window,
            "hold_remaining": round(hold_remaining, 2),
            "hold_time": hold_time,
            "alert_reason": alert_reason,
            "yolo_run_this_frame": run_yolo_now,
            "yolo_run_count": self._yolo_run_count,
            "frame_counter": self._frame_counter,
        }
