import os
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from detector import VideoDeepfakeDetector

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB limit

# Instantiate detector (loads model if available)
detector = VideoDeepfakeDetector(model_path="model_weights/model.pth")  # model optional

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return jsonify({"error": "unsupported file type"}), 400

    vid_id = str(uuid.uuid4())
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{vid_id}{ext}")
    file.save(save_path)

    # Run detection synchronously and return results
    try:
        result = detector.predict_video(save_path)
    except Exception as e:
        return jsonify({"error": "detection failed", "details": str(e)}), 500

    # result is a dict with keys: 'video_id', 'score', 'label', 'frame_scores' (optional)
    return jsonify(result)

# optional: serve uploaded videos (only for local testing)
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)