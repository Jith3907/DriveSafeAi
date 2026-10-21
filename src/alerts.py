"""
Alert Management System for DriveSafe AI.
Handles visual HUD banners, sound chime notifications, and alarm cooldowns.
"""

import time
import pygame
from src import config

class AlertSystem:
    def __init__(self):
        self.last_audio_play_time = 0.0
        self.audio_initialized = False
        self.sound = None
        self._init_audio()

    def _init_audio(self):
        if not config.AUDIO_ALERT_ENABLED:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            if config.ALERT_SOUND_PATH.exists():
                self.sound = pygame.mixer.Sound(str(config.ALERT_SOUND_PATH))
                self.audio_initialized = True
        except Exception as e:
            print(f"[AlertSystem] Warning: Audio init failed ({e}). Running in silent HUD mode.")
            self.audio_initialized = False

    def trigger_alert(self, message: str, severity: str = config.SEVERITY_WARNING, play_sound: bool = True):
        current_time = time.time()
        if (severity == config.SEVERITY_CRITICAL or severity == config.SEVERITY_WARNING) and play_sound:
            if self.audio_initialized and self.sound:
                if (current_time - self.last_audio_play_time) >= config.AUDIO_COOLDOWN_SECONDS:
                    try:
                        self.sound.play()
                        self.last_audio_play_time = current_time
                    except Exception:
                        pass
        return {
            "message": message,
            "severity": severity,
            "color": config.ALERT_COLORS.get(severity, config.ALERT_COLORS[config.SEVERITY_NORMAL])
        }
