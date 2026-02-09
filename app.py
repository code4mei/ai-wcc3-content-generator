from flask import Flask, request, jsonify, render_template,send_from_directory
import os
from video_analyzer import find_highlights
from effects_engine import cut_highlight_clips,merge_clips
from ai_text import add_ai_text_to_clips

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Step 1 → Receive video
    Step 2 → Detect highlights
    Step 3 → Cut clips
    Step 4 → Add AI text
    Step 5 → Return output paths
    """

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    print("Video uploaded:", video_path)
    #auto-increment run folder
    existing_runs = [
    d for d in os.listdir("outputs")
    if d.startswith("run_")
    ]
    run_id = len(existing_runs) + 1
    run_folder = f"outputs/run_{run_id}"
    os.makedirs(run_folder)
    print(f"📁 Created run folder: {run_folder}", flush=True)
    # 🔹 Step 1: Detect highlights
    highlight_times = find_highlights(video_path)

    if not highlight_times:
        return jsonify({"message": "No highlights detected"}), 200

    # 🔹 Step 2: Cut highlight clips
    clips = cut_highlight_clips(video_path,highlight_times,run_folder)

    # 🔹 Step 3: Add AI text overlays
    
    final_videos = add_ai_text_to_clips(clips,run_folder)
    
    format_type = request.form.get("format", "reel")
    print(f"Requested format: {format_type}", flush=True)
    final_output = os.path.join(run_folder, "final_output.mp4")
    if format_type == "reel":
        target_size = (1080, 1920)   # 9:16
        final_output = f"{run_folder}/final_reel.mp4"
    else:
        target_size = (1920, 1080)   # 16:9
        final_output = f"{run_folder}/final_youtube_video.mp4"
    print("Final videos:", final_videos, flush=True)
    try:
      merge_clips(
          clip_paths=final_videos,
          output_path=final_output,
          target_size=target_size
      )
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "message": "Highlight generation failed"
        }), 400
    return jsonify({
        "message": "Video generated successfully!",
        "final_output": final_output,
        "video_url": f"/outputs/{os.path.basename(run_folder)}/{os.path.basename(final_output)}"
    })

@app.route("/outputs/<path:filename>")
def serve_video(filename):
    return send_from_directory("outputs", filename)

if __name__ == "__main__":
    app.run(debug=True)
