from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from src import config

def compute_face_descriptor(landmarks):
    """
    Computes a scale and translation normalized 3D facial geometry descriptor
    from 478 face mesh landmarks.
    """
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float64)
    # Center at nose tip (landmark 1)
    center = pts[1].copy()
    pts_centered = pts - center
    # Scale by interocular distance (landmarks 33 and 263)
    iod = np.linalg.norm(pts[33] - pts[263])
    if iod <= 1e-6:
        iod = 1.0
    normalized = pts_centered / iod
    return normalized.flatten()

class DriverAuthenticator:
    def __init__(self, face_landmarker=None):
        self.known_driver_profiles = {}  # {name: descriptor}
        self.face_landmarker = face_landmarker
        self._init_landmarker_if_needed()
        self._load_known_drivers()

    def _init_landmarker_if_needed(self):
        if self.face_landmarker is None and config.FACE_LANDMARKER_MODEL_PATH.exists():
            base_options = mp_python.BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL_PATH))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def _load_known_drivers(self):
        """Loads and extracts facial descriptors for all registered driver images."""
        if not config.KNOWN_DRIVERS_DIR.exists() or self.face_landmarker is None:
            return

        image_extensions = (".jpg", ".jpeg", ".png", ".webp")
        for img_path in config.KNOWN_DRIVERS_DIR.iterdir():
            if img_path.suffix.lower() in image_extensions:
                driver_name = img_path.stem.replace("_", " ").title()
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                res = self.face_landmarker.detect(mp_img)
                if res.face_landmarks:
                    desc = compute_face_descriptor(res.face_landmarks[0])
                    self.known_driver_profiles[driver_name] = desc
                    print(f"[DriverAuth] Registered driver profile loaded: {driver_name}")

    def register_face(self, driver_name: str, landmarks, frame=None):
        """Register a driver directly from live landmarks and optionally save reference photo."""
        desc = compute_face_descriptor(landmarks)
        clean_name = driver_name.strip().title()
        self.known_driver_profiles[clean_name] = desc
        if frame is not None:
            save_path = config.KNOWN_DRIVERS_DIR / f"{driver_name.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(str(save_path), frame)
        print(f"[DriverAuth] Driver '{clean_name}' successfully registered.")

    def authenticate(self, landmarks, frame=None):
        """
        Compares live facial landmarks with known driver profiles.
        Returns registered driver's name if similarity >= threshold, otherwise 'Unknown'.
        """
        if not self.known_driver_profiles:
            return {
                "driver_name": "Unknown",
                "is_authorized": False,
                "status": "UNKNOWN DRIVER",
                "similarity": 0.0
            }

        live_desc = compute_face_descriptor(landmarks)
        live_norm = np.linalg.norm(live_desc)

        best_name = "Unknown"
        best_sim = -1.0

        for name, ref_desc in self.known_driver_profiles.items():
            ref_norm = np.linalg.norm(ref_desc)
            if ref_norm == 0 or live_norm == 0:
                continue
            sim = float(np.dot(ref_desc, live_desc) / (ref_norm * live_norm))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        is_authorized = (best_sim >= config.FACE_AUTH_SIMILARITY_THRESHOLD)

        return {
            "driver_name": best_name if is_authorized else "Unknown",
            "is_authorized": is_authorized,
            "status": "AUTHORIZED" if is_authorized else "UNKNOWN DRIVER",
            "similarity": best_sim
        }
