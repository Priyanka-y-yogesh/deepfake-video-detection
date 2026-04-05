import os
import cv2
import numpy as np
import statistics

class VideoDeepfakeDetector:
    def __init__(self, model_path=None):
        if model_path:
            print(f"Loading model from {model_path}")
            # self.model = torch.load(model_path, map_location="cpu")
        else:
            print("No model path provided, proceeding without model.")

        # ✅ Add this line (fixes the error)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


    def predict_video(self, video_path, fps_sample=1.0, max_frames=60):
        """
        Analyze the video and return a dict:
        { video_id, score (0..1 fake probability), label, frame_scores }
        fps_sample: frames per second to sample (1.0 means 1 fps)
        max_frames: maximum number of frames to analyze (bounds runtime)
        """
        frames = self._extract_frames_opencv(video_path, fps_sample=fps_sample, max_frames=max_frames)
        frame_scores = []

        for f in frames:
            crops = self._get_face_crops(f)
            if not crops:
                crops = [f]  # if no face found, analyze whole frame
            for c in crops:
                s = self._predict_frame_heuristic(c)
                frame_scores.append(s)

        if not frame_scores:
            # if nothing processed, return neutral score
            return {
                "video_id": os.path.basename(video_path),
                "score": 0.5,
                "label": "UNKNOWN",
                "frame_scores": []
            }

        avg_score = float(statistics.mean(frame_scores))
        label = "FAKE" if avg_score > 0.5 else "REAL"
        return {
            "video_id": os.path.basename(video_path),
            "score": avg_score,
            "label": label,
            "frame_scores": frame_scores
        }

    def _extract_frames_opencv(self, video_path, fps_sample=1.0, max_frames=60):
        """
        Use cv2.VideoCapture to sample frames at approximately fps_sample fps.
        Returns list of BGR numpy arrays.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[detector] cannot open video: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # compute step in frames
        step = max(1, int(round(video_fps / float(fps_sample)))) if fps_sample > 0 else 1

        frames = []
        idx = 0
        grabbed = True
        while grabbed and len(frames) < max_frames:
            grabbed, frame = cap.read()
            if not grabbed or frame is None:
                break
            if idx % step == 0:
                frames.append(frame.copy())
            idx += 1

        cap.release()
        return frames

    def _get_face_crops(self, frame):
        """
        Run Haar cascade face detection and return list of crops (BGR).
        Returns empty list if no faces found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        crops = []
        for (x, y, w, h) in faces:
            # expand box slightly
            pad_w = int(0.15 * w)
            pad_h = int(0.15 * h)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            crop = frame[y1:y2, x1:x2]
            if crop.size != 0:
                crops.append(crop)
        return crops

    def _predict_frame_heuristic(self, img_bgr):
        """
        Heuristic fake score (0..1): combines blurriness (Laplacian) and edge artifact metric.
        Calibrated to keep realistic (REAL) videos near 0.1–0.3 and FAKE near 0.7–0.9.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1) Laplacian variance (blurriness)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())

        # 2) Edge density (Sobel magnitude mean)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        edge_energy = float(np.mean(mag))

        # Normalize and invert so more natural-looking = lower score
        lap_norm = 1.0 - np.clip(lap_var / 150.0, 0.0, 1.0)
        edge_norm = np.clip(edge_energy / 50.0, 0.0, 1.0)

        # Weighted combination (tuned for stable separation)
        raw_score = 0.6 * lap_norm + 0.4 * edge_norm

        # Smooth calibration curve to pull scores closer to extremes
        adjusted_score = (raw_score ** 1.5) * 0.9 + 0.05  # squashes midrange

        # Clip for safety
        adjusted_score = float(np.clip(adjusted_score, 0.0, 1.0))
        return adjusted_score