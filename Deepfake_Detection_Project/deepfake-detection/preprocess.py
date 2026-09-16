"""
Preprocessing Module
Handles video validation, frame extraction, face detection, and face alignment.
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional

import config

logger = logging.getLogger(__name__)


def validate_video(video_path: str) -> Tuple[bool, dict]:
    """
    Validate the uploaded video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (is_valid, metadata_dict).
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False, {"error": "File not found"}

    ext = os.path.splitext(video_path)[1].lower().replace(".", "")
    if ext not in config.ALLOWED_EXTENSIONS:
        logger.error(f"Unsupported video format: {ext}")
        return False, {"error": f"Unsupported format: {ext}"}

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > config.MAX_VIDEO_SIZE_MB:
        logger.error(f"Video too large: {file_size_mb:.1f}MB")
        return False, {"error": f"File too large ({file_size_mb:.1f}MB)"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return False, {"error": "Cannot open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    if duration > config.MAX_VIDEO_DURATION_SEC:
        logger.error(f"Video too long: {duration:.1f}s")
        return False, {"error": f"Video too long ({duration:.1f}s)"}

    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
        "file_size_mb": file_size_mb,
        "format": ext,
    }

    logger.info(f"Video validated: {metadata}")
    return True, metadata


def extract_frames(video_path: str, interval_ms: int = None) -> Tuple[List[np.ndarray], dict]:
    """
    Extract frames from video at specified intervals using OpenCV.

    Args:
        video_path: Path to the video file.
        interval_ms: Interval between frames in milliseconds.

    Returns:
        Tuple of (list of frames, metadata).
    """
    if interval_ms is None:
        interval_ms = config.FRAME_EXTRACTION_INTERVAL_MS

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video for frame extraction")
        return [], {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps * interval_ms / 1000))
    frames = []
    frame_indices = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            frames.append(frame)
            frame_indices.append(idx)
        idx += 1

    cap.release()

    metadata = {
        "total_frames_extracted": len(frames),
        "total_frames_in_video": idx,
        "frame_indices": frame_indices,
        "interval_ms": interval_ms,
        "fps": fps,
    }

    logger.info(f"Extracted {len(frames)} frames from video")
    return frames, metadata


def detect_face_mtcnn(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect and crop face using MTCNN.
    Falls back to MediaPipe if MTCNN fails.

    Args:
        frame: Input BGR frame.

    Returns:
        Cropped and aligned face image (224x224) or None.
    """
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)

        if results:
            bbox = results[0]["box"]  # x, y, w, h
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            padding = int(max(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, config.TARGET_FACE_SIZE)
            return face
    except ImportError:
        logger.warning("MTCNN not available, falling back to MediaPipe")
    except Exception as e:
        logger.warning(f"MTCNN detection failed: {e}, falling back to MediaPipe")

    return detect_face_mediapipe(frame)


def detect_face_mediapipe(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect and crop face using MediaPipe Face Detection.

    Args:
        frame: Input BGR frame.

    Returns:
        Cropped and aligned face image (224x224) or None.
    """
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]
            bbox_c = detection.location_data.relative_bounding_box
            h_frame, w_frame = frame.shape[:2]
            x = int(bbox_c.xmin * w_frame)
            y = int(bbox_c.ymin * h_frame)
            w = int(bbox_c.width * w_frame)
            h = int(bbox_c.height * h_frame)

            padding = int(max(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w_frame - x, w + 2 * padding)
            h = min(h_frame - y, h + 2 * padding)
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, config.TARGET_FACE_SIZE)
            face_detection.close()
            return face

        face_detection.close()
    except Exception as e:
        logger.error(f"MediaPipe face detection failed: {e}")

    return None


def align_face(face: np.ndarray) -> np.ndarray:
    """
    Align face using eye positions detected by MediaPipe Face Mesh.

    Args:
        face: Cropped face image.

    Returns:
    Aligned face image.
    """
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_face)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = face.shape[:2]

            left_eye = landmarks[33]
            right_eye = landmarks[263]

            left_eye_pos = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_pos = (int(right_eye.x * w), int(right_eye.y * h))

            dx = right_eye_pos[0] - left_eye_pos[0]
            dy = right_eye_pos[1] - left_eye_pos[1]
            angle = np.degrees(np.arctan2(dy, dx))

            if abs(angle) > 0.5:
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                face = cv2.warpAffine(face, rotation_matrix, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

            face_mesh.close()
            return face

        face_mesh.close()
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}")

    return face


def preprocess_face(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Complete face preprocessing pipeline:
    detect -> crop -> align -> resize -> normalize.

    Args:
        frame: Input BGR frame.

    Returns:
        Preprocessed face image or None.
    """
    face = detect_face_mtcnn(frame)
    if face is None:
        return None

    face = align_face(face)
    face = cv2.resize(face, config.TARGET_FACE_SIZE)
    face_normalized = face.astype(np.float32) / 255.0

    return face_normalized


def preprocess_frames(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    """
    Preprocess a list of frames, extracting and aligning faces.

    Args:
        frames: List of video frames.

    Returns:
        Tuple of (list of preprocessed faces, successful count).
    """
    faces = []
    success_count = 0

    for i, frame in enumerate(frames):
        face = preprocess_face(frame)
        if face is not None:
            faces.append(face)
            success_count += 1
        else:
            logger.warning(f"No face detected in frame {i}")

    logger.info(f"Preprocessed {success_count}/{len(frames)} frames successfully")
    return faces, success_count
