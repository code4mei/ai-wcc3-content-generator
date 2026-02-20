from flask import Flask, request, render_template, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename
import os
import time
import traceback
import subprocess
from video_analyzer import find_highlights, detect_clip_layout
from effects_engine import cut_highlight_clips, create_reel_json, render_with_remotion, copy_to_public_folder
from ai_text import process_clips
import json
import config

app = Flask(__name__)

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

# Pipeline steps: (weight, display text)
STEPS = [
    (5,  "Saving video..."),
    (25, "Analyzing video for highlights..."),
    (5,  "Saving transcript..."),
    (20, "Cutting highlight clips..."),
    (15, "Processing clips with effects..."),
    (5,  "Detecting clip layouts..."),
    (5,  "Preparing reel configuration..."),
    (20, "Rendering final video..."),
]


def _make_error(message):
    """Helper to yield a JSON error event."""
    return json.dumps({"type": "error", "message": message}) + "\n"


def _validate_upload(video_file):
    """Validate the uploaded file. Returns (is_valid, error_message)."""
    if not video_file or not video_file.filename:
        return False, "No video file selected"

    _, ext = os.path.splitext(video_file.filename)
    if ext.lower() not in config.ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        )

    video_file.seek(0, os.SEEK_END)
    size_bytes = video_file.tell()
    video_file.seek(0)
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > config.MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.0f} MB). Maximum is {config.MAX_FILE_SIZE_MB} MB."

    if size_bytes == 0:
        return False, "Uploaded file is empty"

    return True, None


def _next_run_id():
    """Generate next run ID from existing folders."""
    existing = [
        int(d.split("_")[1])
        for d in os.listdir(config.OUTPUT_FOLDER)
        if d.startswith("run_") and d.split("_")[1].isdigit()
    ]
    return max(existing, default=0) + 1


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return Response(_make_error("No video uploaded"), mimetype="text/event-stream")

    video_file = request.files["video"]

    is_valid, error_msg = _validate_upload(video_file)
    if not is_valid:
        return Response(_make_error(error_msg), mimetype="text/event-stream")

    safe_name = secure_filename(video_file.filename)
    if not safe_name:
        safe_name = "upload.mp4"
    video_path = os.path.join(config.UPLOAD_FOLDER, safe_name)
    video_file.save(video_path)

    format_type = request.form.get("format", "reel")
    print(f"Video uploaded: {video_path} (format: {format_type})")

    def generate():
        start_time = time.time()

        def send_progress(step_index):
            elapsed = time.time() - start_time
            completed_weight = sum(w for w, _ in STEPS[:step_index])
            if completed_weight > 0:
                est_remaining = (elapsed / completed_weight) * (100 - completed_weight)
            else:
                est_remaining = 0
            return json.dumps({
                "type": "progress",
                "step": step_index + 1,
                "totalSteps": len(STEPS),
                "percent": completed_weight,
                "stepText": STEPS[step_index][1],
                "elapsed": round(elapsed, 1),
                "estimatedRemaining": round(est_remaining, 1),
            }) + "\n"

        try:
            # Step 0: Save video (already done)
            yield send_progress(0)

            run_id = _next_run_id()
            run_folder = os.path.join(config.OUTPUT_FOLDER, f"run_{run_id}")
            os.makedirs(run_folder, exist_ok=True)
            print(f"Created run folder: {run_folder}", flush=True)

            # Step 1: Detect highlights (also returns transcript)
            yield send_progress(1)
            highlight_times, _global_layout, transcript_segments = find_highlights(
                video_path, format_type=format_type
            )

            if not highlight_times:
                yield _make_error("No highlights detected in the video. Try a video with more action.")
                return

            # Step 2: Save transcript
            yield send_progress(2)
            transcript_path = os.path.join(run_folder, "transcript.json")
            with open(transcript_path, "w") as f:
                json.dump(transcript_segments, f, indent=4)
            print(f"Transcript saved at {transcript_path}")

            # Step 3: Cut highlight clips
            yield send_progress(3)
            clips = cut_highlight_clips(video_path, highlight_times, run_folder)

            if not clips:
                yield _make_error("Failed to extract any highlight clips from the video.")
                return

            # Step 4: Process clips (slow-mo effects + captions)
            yield send_progress(4)
            final_videos = process_clips(clips, run_folder, transcript_segments)

            if not final_videos:
                yield _make_error("Failed to process clips with effects.")
                return

            # Step 5: Detect clip layouts
            yield send_progress(5)
            clip_layouts = {}
            for clip_path, event_type, caption in final_videos:
                try:
                    clip_layouts[clip_path] = detect_clip_layout(clip_path)
                except Exception as e:
                    print(f"  Layout detection failed for {clip_path}: {e}")
                    clip_layouts[clip_path] = "verticalCrop"
                print(f"  {os.path.basename(clip_path)}: layoutMode={clip_layouts[clip_path]}")

            # Step 6: Create reel JSON + copy to remotion
            yield send_progress(6)
            create_reel_json(run_folder, clip_metadata=final_videos,
                             clip_layouts=clip_layouts, format_type=format_type)
            copy_to_public_folder(run_folder)

            # Step 7: Remotion render
            yield send_progress(7)
            render_with_remotion(run_folder, format_type=format_type)

            # Done
            elapsed = time.time() - start_time
            video_url = f"/outputs/run_{run_id}/final_reel_remotion.mp4"
            print(f"Pipeline complete in {elapsed:.1f}s", flush=True)

            yield json.dumps({
                "type": "complete",
                "percent": 100,
                "stepText": "Done!",
                "elapsed": round(elapsed, 1),
                "estimatedRemaining": 0,
                "videoUrl": video_url,
                "message": "Video generated successfully!",
            }) + "\n"

        except FileNotFoundError as e:
            print(f"Missing dependency: {e}", flush=True)
            yield _make_error(f"Missing dependency: {e}")
        except subprocess.CalledProcessError as e:
            print(f"Render failed: {e}", flush=True)
            yield _make_error("Video rendering failed. Check that Node.js and Remotion are installed.")
        except Exception as e:
            traceback.print_exc()
            yield _make_error(f"Unexpected error: {e}")
        finally:
            # Clean up temp audio file created by Whisper transcription
            base, _ = os.path.splitext(video_path)
            temp_audio = base + "_temp_audio.wav"
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                    print(f"Cleaned up temp file: {temp_audio}")
                except OSError:
                    pass

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/outputs/<path:filename>")
def serve_video(filename):
    return send_from_directory("outputs", filename)


if __name__ == "__main__":
    app.run(debug=True)
