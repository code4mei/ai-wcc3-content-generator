from moviepy import VideoFileClip,concatenate_videoclips
import os
import cv2
import numpy as np
import subprocess
import json
import shutil
from ai_text import generate_caption_from_text

def detect_scene_changes(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) #frames per second
    prev_frame = None
    frame_no = 0
    highlight_times = []
    cooldown = 4

    print(" Detecting scene changes...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_no % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
        #     frames.append(frame)
        # frame_no += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            frame_no += 1
            continue

        frame_diff = cv2.absdiff(prev_frame, gray)
        score = np.sum(frame_diff)/1000000  # normalize score
        current_time = frame_no / fps

        if score > 60:   # <-- IMPORTANT threshold
            start = max(0, current_time - 2)
            end = current_time + 3
            if not highlight_times or current_time - highlight_times[-1][0] > cooldown:
                highlight_times.append((start, end, "scene"))

        prev_frame = gray
        frame_no += 1

    cap.release()
    print(" Found", len(highlight_times), "scene based highlights")
    return highlight_times

def cut_highlight_clips(video_path, highlight_times,run_folder):
    scene_times = detect_scene_changes(video_path)
    if scene_times:
        highlight_times.extend(scene_times)
    
    # This function cuts small video clips around the highlight times
    # Parameters:
    #     video_path (str) → path of the original video
    #     highlight_times (list) → seconds where highlights happen
    # Returns:
    #     list of output video file names
    # Create outputs folder if it does not exist
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Load the main video
    video = VideoFileClip(video_path)
    print(f"DEBUG: video.duration={video.duration}")

    output_files = []

    print("🚨 FINAL highlight_times used for clipping:")
    for h in highlight_times:
        print(h)
    # Loop through each highlight time
    for i, (start, end, label) in enumerate(highlight_times):
        duration = end - start
        print(f"✂️ Attempting clip {i}: {start} → {end} ({duration:.2f}s)")
        # print(f"DEBUG: start={start}, end={end}, label={label}")
        if duration < 2:
            print("⚠️ Skipping clip: too short")
            continue
        start = max(0, start)
        end = min(video.duration, end)
        if end-start < 1:
          print(f"Skipping bad clip: start={start} to {end}")
          continue
        # Limit clip length (avoid crazy long cuts)
        if end-start > 6:
            end = start + 6
        print(f"Creating clip {i+1}: {start:.2f}s to {end:.2f}s | label={label}")

        # Cut the clip
        small_clip = video.subclipped(start, end)

        # Output file name
        output_name = os.path.join(run_folder, f"highlight_{i+1}.mp4")

        # Save the clip
        small_clip.write_videofile(output_name, codec="libx264", audio_codec="aac", logger=None,
                                   ffmpeg_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"])

        output_files.append((output_name, label, start, end))
    
    # Close video file
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
    return float(json.loads(result.stdout)["format"]["duration"])
# def merge_clips(clip_paths, output_path, target_size):
#     clips = []
#     for c in clip_paths:
#         if os.path.exists(c) and os.path.getsize(c) > 0:
#             try:
#                 clip = VideoFileClip(c).resized(target_size)
#                 clips.append(clip)
#                 print(f"✅ Saved clip: {c}")
#             except Exception as e:
#                 print(f" Failed to load clip {c}: {e}")
#     if not clips:
#         raise ValueError("No valid clips to merge after loading")
#     final = concatenate_videoclips(clips, method="compose")

#     final.write_videofile(
#         output_path,
#         codec="libx264",
#         audio_codec="aac",
#         fps=24,
#         threads=4,
#         preset="veryfast", 
#     )
#     print("[INFO] Merging clips...", flush=True)
#     for c in clips:
#         c.close()
#     final.close()
def create_reel_json(run_folder, fps=30, transition_duration_frames=20, clip_metadata=None, layout_mode="contain"):
    clips_data = []

    print("Creating reel.json...")

    if clip_metadata:
        # Use metadata from process_clips: (path, label, text)
        for clip_path, label, text in clip_metadata:
            file = os.path.basename(clip_path)
            full_path = os.path.abspath(clip_path)

            duration_seconds = get_video_duration(full_path)
            duration_frames = int(duration_seconds * fps)

            remotion_path = f"clips/{file}"

            caption_data = text
            clips_data.append({
                "path": remotion_path,
                "durationFrames": duration_frames,
                "caption": caption_data,
                "layoutMode": layout_mode
            })

            print(f" Added {file} | {duration_seconds:.2f}s | {label} | {text}")
    else:
        # Fallback: scan folder for clip files
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
                "text": "INTENSE MOMENT",
                "highlights": ["INTENSE"],
                "layoutMode": layout_mode
            })

            print(f" Added {file} | {duration_seconds:.2f}s")

    reel_json = {
        "clips": clips_data,
        "transitionDurationFrames": transition_duration_frames
    }

    json_path = os.path.join(run_folder, "reel.json")
    with open(json_path, "w") as f:
        json.dump(reel_json, f, indent=2)

    print("reel.json created successfully")

def render_with_remotion(run_folder):
    with open(os.path.join(run_folder, "reel.json")) as f:
        reel_data = json.load(f)

    output_path = os.path.abspath(os.path.join(run_folder, "final_reel_remotion.mp4"))

    transition_duration = reel_data.get("transitionDurationFrames", 20)
    clips = reel_data["clips"]
    intro_frames = 75  # matches INTRO_DURATION_FRAMES in IntroCard.tsx

    # Total = intro + sum of clip durations - overlap from transitions
    # len(clips) transitions: 1 intro→first clip + (len(clips) - 1) between clips
    sum_of_durations = sum(clip["durationFrames"] for clip in clips)
    num_transitions = len(clips)
    duration_frames = intro_frames + sum_of_durations - (num_transitions * transition_duration)

    props = {
        "clips": clips,
        "transitionDurationFrames": transition_duration
    }
    command = [
        r"C:\Program Files\nodejs\npx.cmd",
        "remotion",
        "render",
        "src/index.ts",
        "WCC3REEL",
        output_path,
        "--props=" + json.dumps(props),
    ]

    result = subprocess.run(
        command,
        cwd=r"C:\Users\1102\ai-wcc3-content-generator\remotion-renderer",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Remotion STDOUT:", result.stdout)
        print("Remotion STDERR:", result.stderr)
        result.check_returncode()
    remotion_public_clips = r"C:\Users\1102\ai-wcc3-content-generator\remotion-renderer\public"
    destination_path = os.path.join(remotion_public_clips, "final.mp4")

    shutil.copy2(output_path, destination_path)

    print("Final video copied to Remotion public as final.mp4")
    print("Remotion render complete")
def copy_to_public_folder(run_folder):
    remotion_public_clips = r"C:\Users\1102\ai-wcc3-content-generator\remotion-renderer\public"
    print("Run folder:", run_folder)
    print("Remotion public folder:", remotion_public_clips)
    print("Does public exist?", os.path.exists(remotion_public_clips))
    clips_folder = os.path.join(remotion_public_clips,"clips")
    if os.path.exists(clips_folder):
        for f in os.listdir(clips_folder):
            os.remove(os.path.join(clips_folder, f))
    #create clips folder inside public if not exists
    os.makedirs(clips_folder, exist_ok=True)
    os.makedirs(remotion_public_clips, exist_ok=True)
    print("Clips folder created at:", clips_folder)

    for file in os.listdir(run_folder):
        if file.endswith(".mp4") and file.startswith("clip_"):
            src_path = os.path.join(run_folder,file)
            dest_path = os.path.join(clips_folder, file)
            print("Copying from:", src_path)
            print("Copying to:", dest_path)
            shutil.copy2(src_path, dest_path)
            print(f"Copied {file} to Remotion Public Folder")
    print("Final files in remotion public:")
    print(os.listdir(clips_folder))