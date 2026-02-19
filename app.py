from flask import Flask, request, jsonify, render_template,send_from_directory
import os
from video_analyzer import find_highlights, transcribe_audio
from effects_engine import cut_highlight_clips,create_reel_json,render_with_remotion,copy_to_public_folder
from ai_text import process_clips
import json

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
    highlight_times,layout_mode = find_highlights(video_path)

    if not highlight_times:
        return jsonify({"message": "No highlights detected"}), 200
    print("🧠 Generating transcript...")
    transcript_segments = transcribe_audio(video_path)

    # Save transcript to run folder
    transcript_path = os.path.join(run_folder, "transcript.json")
    with open(transcript_path, "w") as f:
        json.dump(transcript_segments, f, indent=4)
    print(f"Transcript saved at {transcript_path}")
    # 🔹 Step 2: Cut highlight clips
    clips = cut_highlight_clips(video_path,highlight_times,run_folder)
    # 🔹 Step 3: Process clips (slow-mo effects)
    final_videos = process_clips(clips, run_folder,transcript_segments)
    #Create reel.json with clip metadata for Remotion
    create_reel_json(run_folder, clip_metadata=final_videos, layout_mode=layout_mode)
    #copy to remotion folder
    # remotion_public_folder = "C:\Users\1102\remotion-project\public\clips"
    copy_to_public_folder(run_folder)
    #render only once
    render_with_remotion(run_folder)

    final_output = os.path.join(run_folder, "final_reel_remotion.mp4")
    print("Final videos:", final_videos, flush=True)

    return jsonify({
        "message": "Video generated successfully!",
        "final_output": final_output,
        "video_url": f"/outputs/{os.path.basename(run_folder)}/final_reel_remotion.mp4"
    })

@app.route("/outputs/<path:filename>")
def serve_video(filename):
    return send_from_directory("outputs", filename)

if __name__ == "__main__":
    app.run(debug=True)
