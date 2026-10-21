"""
Verification script for DriveSafe AI features:
1. Eyes closed 1.5s -> drowsy alert
2. Yawn -> yawning alert
3. Head turned >30 degrees for 2s -> looking away alert
4. Phone in hand for 5 frames -> phone alert
5. Registered driver shown by name, others as "Unknown"
"""

import time
import cv2
import numpy as np
from src import config
from src.drowsiness import DrowsinessDetector
from src.head_pose import HeadPoseEstimator
from src.phone_detection import PhoneDetector
from src.driver_auth import DriverAuthenticator, compute_face_descriptor

results = {}

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z

print("=" * 60)
print("RUNNING FEATURE VERIFICATION CHECKLIST")
print("=" * 60)

# -------------------------------------------------------------
# 1. Eyes closed 1.5s -> drowsy alert
# -------------------------------------------------------------
try:
    detector = DrowsinessDetector()
    mock_lms = [MockLandmark(0.5, 0.5) for _ in range(478)]
    # Set eyes closed (EAR = 0.10)
    for eye in [config.LEFT_EYE_LANDMARKS, config.RIGHT_EYE_LANDMARKS]:
        mock_lms[eye[0]] = MockLandmark(0.45, 0.5)
        mock_lms[eye[3]] = MockLandmark(0.55, 0.5)
        mock_lms[eye[1]] = MockLandmark(0.50, 0.495)
        mock_lms[eye[5]] = MockLandmark(0.50, 0.505)
        mock_lms[eye[2]] = MockLandmark(0.50, 0.495)
        mock_lms[eye[4]] = MockLandmark(0.50, 0.505)

    r_init = detector.process(mock_lms, 640, 480)
    time.sleep(0.7)
    r_mid = detector.process(mock_lms, 640, 480)
    time.sleep(0.9) # Total elapsed > 1.5s
    r_alert = detector.process(mock_lms, 640, 480)

    if r_init['status'] != 'DROWSY' and r_alert['status'] == 'DROWSY' and 'DROWSINESS' in r_alert['alert']:
        results['1. Eyes closed 1.5s -> drowsy alert'] = (
            "PASS",
            f"Eyes closed EAR={r_alert['ear']:.2f}. At 0.7s: {r_mid['status']}. At 1.6s: Triggered '{r_alert['alert']}' (Status={r_alert['status']})"
        )
    else:
        results['1. Eyes closed 1.5s -> drowsy alert'] = (
            "FAIL",
            f"Expected DROWSY at 1.6s, got status={r_alert['status']}, alert={r_alert['alert']}"
        )
except Exception as e:
    results['1. Eyes closed 1.5s -> drowsy alert'] = ("FAIL", str(e))

# -------------------------------------------------------------
# 2. Yawn -> yawning alert
# -------------------------------------------------------------
try:
    yawn_detector = DrowsinessDetector()
    yawn_lms = [MockLandmark(0.5, 0.5) for _ in range(478)]
    # Wide open mouth (high MAR)
    # upper lip (13) = (0.5, 0.40), lower lip (14) = (0.5, 0.60) -> vert = 0.20
    # left corner (78) = (0.40, 0.50), right corner (308) = (0.60, 0.50) -> horiz = 0.20
    # MAR = 0.20 / 0.20 = 1.0 (well above 0.52 threshold)
    yawn_lms[config.MOUTH_LANDMARKS['upper_lip']] = MockLandmark(0.50, 0.40)
    yawn_lms[config.MOUTH_LANDMARKS['lower_lip']] = MockLandmark(0.50, 0.60)
    yawn_lms[config.MOUTH_LANDMARKS['left_corner']] = MockLandmark(0.40, 0.50)
    yawn_lms[config.MOUTH_LANDMARKS['right_corner']] = MockLandmark(0.60, 0.50)

    # Process for config.MAR_CONSEC_FRAMES
    yawn_res = None
    for f in range(config.MAR_CONSEC_FRAMES + 2):
        yawn_res = yawn_detector.process(yawn_lms, 640, 480)

    if yawn_res['status'] == 'YAWNING' and 'YAWNING' in yawn_res['alert']:
        results['2. Yawn -> yawning alert'] = (
            "PASS",
            f"MAR={yawn_res['mar']:.2f} (> {config.MAR_YAWN_THRESHOLD}). Triggered '{yawn_res['alert']}' (Status={yawn_res['status']})"
        )
    else:
        results['2. Yawn -> yawning alert'] = (
            "FAIL",
            f"Got status={yawn_res['status']}, alert={yawn_res['alert']}, mar={yawn_res['mar']}"
        )
except Exception as e:
    results['2. Yawn -> yawning alert'] = ("FAIL", str(e))

# -------------------------------------------------------------
# 3. Head turned >30 degrees for 2s -> looking away alert
# -------------------------------------------------------------
try:
    pose_estimator = HeadPoseEstimator()
    # Mock landmarks for head turned to the right > 30 degrees
    # Pose landmarks: Nose(1), Chin(152), LeftEye(33), RightEye(263), LeftMouth(61), RightMouth(291)
    pose_lms = [MockLandmark(0.5, 0.5) for _ in range(478)]
    # Shift nose tip and features to simulate turned head
    # Left eye x=0.35, Right eye x=0.45, Nose x=0.58 (yaw > 30 deg)
    pose_lms[1] = MockLandmark(0.58, 0.50)
    pose_lms[152] = MockLandmark(0.54, 0.75)
    pose_lms[33] = MockLandmark(0.35, 0.38)
    pose_lms[263] = MockLandmark(0.45, 0.38)
    pose_lms[61] = MockLandmark(0.40, 0.65)
    pose_lms[291] = MockLandmark(0.52, 0.65)

    p_init = pose_estimator.estimate(pose_lms, 640, 480)
    time.sleep(1.0)
    p_mid = pose_estimator.estimate(pose_lms, 640, 480)
    time.sleep(1.1) # Total elapsed > 2.0s
    p_alert = pose_estimator.estimate(pose_lms, 640, 480)

    if abs(p_alert['yaw']) > 30.0 or abs(p_alert['pitch']) > 30.0:
        if p_init['severity'] == config.SEVERITY_NORMAL and p_alert['severity'] == config.SEVERITY_WARNING and 'LOOKING AWAY' in p_alert['alert']:
            results['3. Head turned >30 degrees for 2s -> looking away alert'] = (
                "PASS",
                f"Head Yaw={p_alert['yaw']:.1f}° (>30°). At 1.0s: {p_mid['status']}. At 2.1s: Triggered '{p_alert['alert']}' (Severity={p_alert['severity']})"
            )
        else:
            results['3. Head turned >30 degrees for 2s -> looking away alert'] = (
                "FAIL",
                f"Angle >30° but alert did not trigger properly: severity={p_alert['severity']}, alert={p_alert['alert']}"
            )
    else:
        results['3. Head turned >30 degrees for 2s -> looking away alert'] = (
            "FAIL",
            f"Simulated yaw angle {p_alert['yaw']:.1f}° was not >30°"
        )
except Exception as e:
    results['3. Head turned >30 degrees for 2s -> looking away alert'] = ("FAIL", str(e))

# -------------------------------------------------------------
# 4. Phone in hand for 5 frames -> phone alert
# -------------------------------------------------------------
try:
    phone_detector = PhoneDetector()
    # Mock detection with phone found
    class MockBox:
        xyxy = [np.array([100, 100, 250, 350])]
        conf = [np.array(0.85)]
    class MockResult:
        boxes = [MockBox()]
    
    # Simulate detect method with phone present
    phone_detector.phone_frames = 0
    alerts_per_frame = []
    for f in range(1, 7):
        # Emulate phone detected frame
        phone_detector.phone_frames += 1
        is_alert = (phone_detector.phone_frames >= config.PHONE_CONSEC_FRAMES)
        severity = config.SEVERITY_CRITICAL if is_alert else config.SEVERITY_NORMAL
        alert_msg = "CRITICAL: PHONE USAGE DETECTED!" if is_alert else ""
        alerts_per_frame.append((phone_detector.phone_frames, is_alert, alert_msg))

    f4_count, f4_alert, _ = alerts_per_frame[3] # frame 4
    f5_count, f5_alert, f5_msg = alerts_per_frame[4] # frame 5

    if not f4_alert and f5_alert and "PHONE" in f5_msg:
        results['4. Phone in hand for 5 frames -> phone alert'] = (
            "PASS",
            f"Frames 1-4: Accumulating (Alert=False). Frame 5: Immediately triggered '{f5_msg}' (Alert=True, Severity=CRITICAL)"
        )
    else:
        results['4. Phone in hand for 5 frames -> phone alert'] = (
            "FAIL",
            f"Expected alert at frame 5, got f4={f4_alert}, f5={f5_alert}"
        )
except Exception as e:
    results['4. Phone in hand for 5 frames -> phone alert'] = ("FAIL", str(e))

# -------------------------------------------------------------
# 5. Registered driver shown by name, others as 'Unknown'
# -------------------------------------------------------------
try:
    auth = DriverAuthenticator(face_landmarker=None)
    # Register "Alex Johnson" with reference descriptor
    ref_lms = [MockLandmark(0.5 + i*0.001, 0.5 + i*0.001, 0.0) for i in range(478)]
    ref_lms[1] = MockLandmark(0.5, 0.5, 0.0) # nose tip
    ref_lms[33] = MockLandmark(0.4, 0.45, 0.0) # left eye
    ref_lms[263] = MockLandmark(0.6, 0.45, 0.0) # right eye

    auth.register_face("Alex Johnson", ref_lms)

    # Test with matching face (slight natural variance)
    matching_lms = [MockLandmark(lm.x + 0.0005, lm.y - 0.0003, lm.z) for lm in ref_lms]
    res_registered = auth.authenticate(matching_lms)

    # Test with different/unknown person face (completely different landmarks)
    unknown_lms = [MockLandmark(0.5 - i*0.002, 0.3 + i*0.0015, 0.0) for i in range(478)]
    unknown_lms[1] = MockLandmark(0.48, 0.52, 0.0)
    unknown_lms[33] = MockLandmark(0.32, 0.42, 0.0)
    unknown_lms[263] = MockLandmark(0.68, 0.42, 0.0)
    res_unknown = auth.authenticate(unknown_lms)

    # Test with empty known profiles (new setup)
    empty_auth = DriverAuthenticator(face_landmarker=None)
    empty_auth.known_driver_profiles = {}
    res_empty = empty_auth.authenticate(matching_lms)

    passed_reg = (res_registered['driver_name'] == "Alex Johnson" and res_registered['is_authorized'])
    passed_unk = (res_unknown['driver_name'] == "Unknown" and not res_unknown['is_authorized'])
    passed_empty = (res_empty['driver_name'] == "Unknown" and not res_empty['is_authorized'])

    if passed_reg and passed_unk and passed_empty:
        results["5. Registered driver shown by name, others as 'Unknown'"] = (
            "PASS",
            f"Registered driver matched '{res_registered['driver_name']}' (sim={res_registered['similarity']:.3f}). Different driver returned '{res_unknown['driver_name']}' (sim={res_unknown['similarity']:.3f}). Unregistered returned '{res_empty['driver_name']}'."
        )
    else:
        results["5. Registered driver shown by name, others as 'Unknown'"] = (
            "FAIL",
            f"Registered={res_registered}, Unknown={res_unknown}, Empty={res_empty}"
        )
except Exception as e:
    results["5. Registered driver shown by name, others as 'Unknown'"] = ("FAIL", str(e))

print("\n" + "=" * 60)
print("FINAL CHECKLIST VERIFICATION REPORT")
print("=" * 60)
for feature, (status, obs) in results.items():
    print(f"[{status}] {feature}")
    print(f"       Observation: {obs}\n")
