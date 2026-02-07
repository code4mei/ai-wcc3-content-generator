from moviepy import VideoFileClip,concatenate_videoclips
import os
import cv2
import numpy as np
import subprocess
import json

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

def cut_highlight_clips(video_path, highlight_times):
    highlight_times = detect_scene_changes(video_path)
    
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
        print(f"DEBUG: start={start}, end={end}, label={label}")

        # # We take 3 seconds before and after highlight
        # start_time = highlight_time - 3
        # end_time = highlight_time + 3

        # Make sure times are inside video duration
        # if start_time < 0:
        #     start_time = 0
        # if end_time > video.duration:
        #     end_time = video.duration
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
        output_name = f"outputs/highlight_{i+1}.mp4"

        # Save the clip
        small_clip.write_videofile(output_name, codec="libx264", audio_codec="aac", logger=None)

        output_files.append((output_name, label))
    
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
    return float(json.loads(result.stdout)["format"]["duration"])
def merge_clips(clip_paths, output_path, target_size):
    clips = []
    for c in clip_paths:
        clip = VideoFileClip(c).resized(target_size)
        clips.append(clip)
    final = concatenate_videoclips(clips, method="compose")

    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        threads=4,
        preset="veryfast", 
    )
    print("[INFO] Merging clips...", flush=True)
    for c in clips:
        c.close()
    final.close()