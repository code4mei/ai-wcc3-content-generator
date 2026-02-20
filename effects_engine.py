from moviepy import VideoFileClip, concatenate_videoclips
import os
import cv2
import numpy as np
import subprocess
import json
import shutil
import config


def detect_scene_changes(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_frame = None
    frame_no = 0
    highlight_times = []

    print("  Detecting scene changes...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            frame_no += 1
            continue

        frame_diff = cv2.absdiff(prev_frame, gray)
        score = np.sum(frame_diff) / 1000000
        current_time = frame_no / fps

        if score > config.SCENE_CHANGE_THRESHOLD:
            start = max(0, current_time - 2)
            end = current_time + 3
            if not highlight_times or current_time - highlight_times[-1][0] > config.SCENE_CHANGE_COOLDOWN:
                highlight_times.append((start, end, "scene"))

        prev_frame = gray
        frame_no += 1

    cap.release()
    print(f"  Found {len(highlight_times)} scene-based highlights")
    return highlight_times


def cut_highlight_clips(video_path, highlight_times, run_folder):
    scene_times = detect_scene_changes(video_path)
    if scene_times:
        highlight_times.extend(scene_times)

    os.makedirs(run_folder, exist_ok=True)

    video = VideoFileClip(video_path)
    output_files = []

    for i, (start, end, label) in enumerate(highlight_times):
        duration = end - start
        if duration < config.MIN_CLIP_DURATION:
            print(f"  Skipping clip {i}: too short ({duration:.1f}s)")
            continue
        start = max(0, start)
        end = min(video.duration, end)
        if end - start < 1:
            continue
        if end - start > config.MAX_CLIP_DURATION:
            end = start + config.MAX_CLIP_DURATION

        print(f"  Clip {i + 1}: {start:.2f}s - {end:.2f}s [{label}]")
        small_clip = video.subclipped(start, end)
        output_name = os.path.join(run_folder, f"highlight_{i + 1}.mp4")
        small_clip.write_videofile(output_name, codec="libx264", audio_codec="aac", logger=None,
                                   ffmpeg_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"])
        output_files.append((output_name, label, start, end))

    video.close()
    return output_files


def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffprobe failed: {result.stderr}")
    data = json.loads(result.stdout)
    if "format" not in data:
        raise Exception(f"Invalid ffprobe output: {result.stdout}")
    return float(data["format"]["duration"])


def create_reel_json(run_folder, fps=None, transition_duration_frames=None,
                     clip_metadata=None, clip_layouts=None, format_type="reel"):
    fps = fps or config.REEL_FPS
    transition_duration_frames = transition_duration_frames or config.TRANSITION_DURATION_FRAMES
    clips_data = []

    print("Creating reel.json...")

    if clip_metadata:
        for i, (clip_path, event_type, caption) in enumerate(clip_metadata):
            file = os.path.basename(clip_path)
            full_path = os.path.abspath(clip_path)

            duration_seconds = get_video_duration(full_path)
            duration_frames = int(duration_seconds * fps)
            remotion_path = f"clips/{file}"

            layout_mode = clip_layouts.get(clip_path, "verticalCrop") if clip_layouts else "verticalCrop"

            clips_data.append({
                "path": remotion_path,
                "durationFrames": duration_frames,
                "caption": caption,
                "layoutMode": layout_mode,
                "eventType": event_type,
            })

            print(f"  Added {file} | {duration_seconds:.2f}s | {event_type} | {caption}")
    else:
        files = sorted([
            f for f in os.listdir(run_folder)
            if f.endswith(".mp4") and "clip_" in f
        ])
        if not files:
            print("No clips found in run folder.")
            return

        for file in files:
            full_path = os.path.abspath(os.path.join(run_folder, file))
            duration_seconds = get_video_duration(full_path)
            duration_frames = int(duration_seconds * fps)
            remotion_path = f"clips/{file}"

            clips_data.append({
                "path": remotion_path,
                "durationFrames": duration_frames,
                "caption": {"text": "INTENSE MOMENT", "highlights": ["INTENSE"]},
                "layoutMode": "verticalCrop",
                "eventType": "default",
            })
            print(f"  Added {file} | {duration_seconds:.2f}s")

    reel_json = {
        "clips": clips_data,
        "transitionDurationFrames": transition_duration_frames,
        "format": format_type,
    }

    json_path = os.path.join(run_folder, "reel.json")
    with open(json_path, "w") as f:
        json.dump(reel_json, f, indent=2)

    print("reel.json created successfully")


def render_with_remotion(run_folder, format_type="reel"):
    with open(os.path.join(run_folder, "reel.json")) as f:
        reel_data = json.load(f)

    output_path = os.path.abspath(os.path.join(run_folder, "final_reel_remotion.mp4"))

    transition_duration = reel_data.get("transitionDurationFrames", config.TRANSITION_DURATION_FRAMES)
    clips = reel_data["clips"]

    # Choose composition based on format
    composition_id = "WCC3REEL" if format_type == "reel" else "WCC3YOUTUBE"

    props = {
        "clips": clips,
        "transitionDurationFrames": transition_duration,
    }

    if not config.NPX_CMD:
        raise FileNotFoundError("npx not found. Install Node.js or set NPX_CMD env var.")

    command = [
        config.NPX_CMD,
        "remotion",
        "render",
        "src/index.ts",
        composition_id,
        output_path,
        "--props=" + json.dumps(props),
    ]

    result = subprocess.run(
        command,
        cwd=config.REMOTION_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Remotion STDOUT:", result.stdout)
        print("Remotion STDERR:", result.stderr)
        result.check_returncode()

    destination_path = os.path.join(config.REMOTION_PUBLIC_DIR, "final.mp4")
    shutil.copy2(output_path, destination_path)

    print("Final video copied to Remotion public as final.mp4")
    print("Remotion render complete")


def copy_to_public_folder(run_folder):
    clips_folder = os.path.join(config.REMOTION_PUBLIC_DIR, "clips")
    if os.path.exists(clips_folder):
        for f in os.listdir(clips_folder):
            os.remove(os.path.join(clips_folder, f))
    os.makedirs(clips_folder, exist_ok=True)

    for file in os.listdir(run_folder):
        if file.endswith(".mp4") and file.startswith("clip_"):
            src_path = os.path.join(run_folder, file)
            dest_path = os.path.join(clips_folder, file)
            shutil.copy2(src_path, dest_path)
            print(f"  Copied {file} to Remotion public")
