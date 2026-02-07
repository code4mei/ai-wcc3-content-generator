import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip    
import numpy as np
import pytesseract
import re
from flask import request

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def parse_score(score_text):
    """
    Parse score string like '125/3' into runs and wickets
    """
    match = re.match(r'(\d+)/(\d+)', score_text)
    if match:
        runs = int(match.group(1))
        wickets = int(match.group(2))
        return runs, wickets
    return None, None

'''
# def find_highlights(video_path):
    """
    Detect cricket highlights based on scoreboard OCR + optional audio/motion filters

    Returns:
        list of highlights → [(start_time, end_time, label), ...]
    """

    print("Analyzing video for highlights...")

    video = VideoFileClip(video_path)
    # audio = video.audio
    duration = int(video.duration)
    cool_down = 0
    # previous_volume = 0

    highlight_times = []
    cap = cv2.VideoCapture(video_path)
    previous_frame = None  #compare frames to measure motion
    prev_scoreboard = None
    # frame_count = 0
    # fps = cap.get(cv2.CAP_PROP_FPS)


    for t in range(0, duration, 1): # check every second
        if cool_down > 0:
            cool_down -= 1
            # previous_volume = volume if 'volume' in locals() else previous_volume
            continue

        #video frame
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)  # go to second t
        success, frame = cap.read()
        if not success:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0
        # # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # # lower_green = np.array([35, 40, 40])
        # # upper_green = np.array([85, 255, 255])
        # # mask = cv2.inRange(hsv, lower_green, upper_green)
        # # green_ratio = np.sum(mask>0) / (frame.shape[0] * frame.shape[1])
        if previous_frame is not None:
            diff = cv2.absdiff(gray, previous_frame)
            motion_score = diff.sum() / (frame.shape[0] * frame.shape[1])  # normalize
        previous_frame = gray

        
        h, w, _ = frame.shape
        scoreboard = frame[int(0.75*h):h, int(0.2*w):int(0.8*w)]
        gray_scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray_scoreboard, (3,3), 0)
        _, thresh = cv2.threshold(gray_scoreboard, 150, 255, cv2.THRESH_BINARY)
        # score_change = 0
        # OCR to detect score
        score_text = pytesseract.image_to_string(thresh, config='--psm 7')
        score_text = score_text.strip().replace(" ", "").replace("\n", "")
        nums = re.findall(r'\d+', score_text)
        runs, wickets = None, None
        if len(nums) >= 2:
            runs = int(nums[0])
            wickets = int(nums[1])
        else:
            runs,wickets = None, None
        label = None
        if prev_scoreboard and runs is not None and wickets is not None:
            prev_runs, prev_wickets = prev_scoreboard
            run_diff = (runs - prev_runs) if runs is not None else 0
            wicket_diff = (wickets - prev_wickets) if wickets is not None else 0

            if wicket_diff > 0:
                label = f"WICKET! ({prev_wickets}→{wickets})"
            elif run_diff >= 6:
                label = f"SIX! (+{run_diff})"
            elif run_diff == 4:
                label = f"BOUNDARY! (+{run_diff})"
            elif run_diff > 0:
                label = f"RUN (+{run_diff})"

            if motion_score > 15 and label is None:  # adjust threshold if too sensitive
                label = "Exciting Moment"
            if label:
                start = max(t - 2, 0)
                end = min(t + 5, duration)
                highlight_times.append((start, end, label))
                cool_down = 3  # avoid double counting nearby frames
        if runs is not None and wickets is not None:
            prev_scoreboard = (runs, wickets)

        # prev_score = (runs, wickets)
        # if volume_change > 0.05 and motion_score > 4 and score_change > 8:  # threshold for high energy, motion, and score change
        #         highlight_times.append((t, min(t+5, duration), volume))
        #         cool_down = 8  # 8 seconds cool-down to avoid multiple detections
    cap.release()
    video.close()  # middle of the chunk
    print(f"Found {len(highlight_times)} highlights.")
    for h in highlight_times:
        print(h)
    # print(f"Time: {t}s | Volume: {round(volume,3)} | Motion: {round(motion_score,2)}")
    return highlight_times '''

def get_audio_peaks(video_path, window_size=0.5, threshold=0.05):
    """
    Detect times in the video where audio amplitude spikes (crowd cheers / celebrations)
    Returns a list of times in seconds
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    duration = video.duration
    times = [] 
    # step = window_size
    t = 0
    while t < duration:
        segment = audio.subclipped(t, min(t+window_size, duration))
        samples = segment.to_soundarray(fps=44100)
        amplitude = np.mean(np.abs(samples))
        if amplitude > threshold:
            times.append(t)
        t += window_size
    print(f"DEBUG: max audio_peak={max(times) if times else 'none'}")
    video.close()
    return times


def get_visual_peaks(video_path, threshold=25):
    """
    Detect frames where visual changes occur (big six text, celebration flashes)
    Returns list of frame indices and FPS
    """
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    highlights = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop area where big six / celebration text appears (tune for your video)
        h, w = gray.shape
        roi = gray[int(0.75*h):h, int(0.2*w):int(0.8*w)]

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, roi)
            score = np.sum(diff) / (roi.shape[0] * roi.shape[1])  # normalize
            if score > threshold:
                highlights.append(frame_idx)

        prev_frame = roi
        frame_idx += 1

    cap.release()
    return highlights, fps

def get_ocr_peaks(video_path):
    """
    Detect highlights from scoreboard changes using OCR
    Returns list of timestamps in seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    prev_score = None
    peaks = []

    for t in range(0, int(duration), 1):  # every 1 sec
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape
        scoreboard = frame[int(0.75*h):h, int(0.2*w):int(0.8*w)]
        gray_scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_scoreboard, 150, 255, cv2.THRESH_BINARY)
        score_text = pytesseract.image_to_string(thresh, config='--psm 7')
        score_text = score_text.strip().replace(" ", "").replace("\n", "")
        runs, wickets = parse_score(score_text)
        if runs is None:
            continue

        if prev_score:
            prev_runs, prev_wickets = prev_score
            run_diff = runs - prev_runs
            wicket_diff = wickets - prev_wickets
            if run_diff >= 4 or wicket_diff > 0:
                peaks.append(t)  # timestamp in seconds
        prev_score = (runs, wickets)

    cap.release()
    return peaks

def merge_peaks(audio_times, visual_times, ocr_times, video_duration):
    """
    Merge audio and visual peaks into final highlight timestamps
    Returns list of tuples: (start_time, end_time)
    """

    # for t in strong_times:
    #     if t > video_duration + 5:
    #         print(f"❌ BAD TIME DETECTED: {t} (probably frame number)")
    #         continue
    # ---------- STEP 1: Combine detections with score ----------
    candidates = []

    for t in audio_times:
        candidates.append((t, 1.0, "crowd"))   # crowd cheer = strong

    for t in visual_times:
        candidates.append((t, 0.9, "motion"))   # motion = medium

    for t in ocr_times:
        candidates.append((t, 0.95, "score_change"))   # score change = VERY strong
    # Fallback so system never returns empty
    if not candidates and audio_times:
        print(" No strong detections — using audio fallback")
        candidates = [(t, 2) for t in audio_times]

    if not candidates:
        print(" No highlights possible")
        return []

    # ---------- STEP 2: Sort by time ----------
    candidates.sort(key=lambda x: x[0])

    PRE_ROLL = 2     # seconds before action (buildup)
    POST_ROLL = 3     # after action (reaction)
    # MIN_GAP = 8       # minimum gap between highlights
    highlights = []
    # last_added = -10  # to avoid duplicates within 10 seconds

    # ---------- STEP 3: Remove nearby duplicates ----------
    grouped_events = []
    current_group = [candidates[0]]

    for t, score, label in candidates[1:]:
        # If event within 4 seconds → same moment
        if t - current_group[-1][0] <= 2:
            current_group.append((t, score, label))
        else:
            grouped_events.append(current_group)
            current_group = [(t, score, label)]

    grouped_events.append(current_group)
    if label == "scene":
        score = 0.5

    # ---------- STEP 4: Keep only important events ----------
    
    events = []
    used_labels = set()
    balanced_times = []
    for group in grouped_events:
        best_event = max(group, key=lambda x: x[1])  # (t, score, label)
        t, score, label = best_event

        if label in used_labels:
            continue
        balanced_times.append(t)  # Store (time, score, label)
        used_labels.add(label)

    # sort by importance FIRST
    events.sort(key=lambda x: x[1], reverse=True)

    events = balanced_times[:5]  # keep strongest  # limit to top 6 highlights

    print("DEBUG candidates", candidates[:10])
    print("DEBUG events", type(events[0]))
    # ---------- STEP 5: Convert to clips ----------
    for t, score, label in candidates:
        start = max(0, t - PRE_ROLL)
        end = min( video_duration, t + POST_ROLL)
        highlights.append((start, end,label))

        # if highlights and start <= highlights[-1][1]:
        #     highlights[-1] = (highlights[-1][0], end)
        # else:
        #     highlights.append((start, end))
    filtered = []
    for h in highlights:
        if not filtered or h[0] > filtered[-1][1]:  # no overlap
            filtered.append(h)
    highlights = filtered
    # ---------- STEP 6: Limit reel duration ----------
    format_type = request.form.get("format", "reel")
    if format_type == "reel":
        highlights = highlights[:6]
    else:
        highlights = highlights[:20]
    max_clips = 6   # 6 clips × ~6 sec = 36 sec reel
    highlights = highlights[:max_clips]

    print(f"Final highlights selected: {len(highlights)}")
    return highlights



def find_highlights(video_path):
    """
    End-to-end highlight detection using audio + visual cues
    Returns: list of (start_time, end_time)
    """
    print("Analyzing video for highlights...")

    video = VideoFileClip(video_path)
    video_duration = video.duration
    print(f"DEBUG: video.duration={video_duration}")

    audio_peaks = get_audio_peaks(video_path)
    visual_frames, fps = get_visual_peaks(video_path)
    visual_times = [f / fps for f in visual_frames]
    ocr_times = get_ocr_peaks(video_path)
    print(f"DEBUG: max audio_peak={max(audio_peaks) if audio_peaks else 'none'}")
    print(f"DEBUG: max visual_time={max(visual_times) if visual_times else 'none'}")
    print(f"DEBUG: max ocr_time={max(ocr_times) if ocr_times else 'none'}")
    print(f"DEBUG: video.duration={video_duration}")
    highlight_times = merge_peaks(audio_peaks, visual_times, ocr_times, video_duration)

    print(f"Found {len(highlight_times)} highlights.")
    for idx, (start, end, label) in enumerate(highlight_times, 1):
        print(f"{idx}. Start: {start}s  End: {end}s  Type: {label}")
    print("🔥 RETURNING highlight_times from find_highlights():")
    for h in highlight_times[:10]:
        print(h)

    video.close()
    return highlight_times