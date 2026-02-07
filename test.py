import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip    
import numpy as np
import pytesseract
import re

# Path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def parse_score(score_text):
    """
    Parse score string like '4/0' into runs and wickets
    """
    match = re.match(r'(\d+)/(\d+)', score_text)
    if match:
        runs = int(match.group(1))
        wickets = int(match.group(2))
        return runs, wickets
    return None, None

def find_highlights(video_path):
    """
    Detect cricket highlights based on scoreboard OCR
    Returns: list of tuples -> (start_time_sec, end_time_sec, label)
    """
    print("Analyzing video for highlights...")

    video = VideoFileClip(video_path)
    duration = int(video.duration)
    highlight_times = []

    cap = cv2.VideoCapture(video_path)
    prev_score = None
    cool_down = 0

    for t in range(0, duration, 2):  # check every 2 seconds
        if cool_down > 0:
            cool_down -= 2
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape

        # --- Crop the bottom scoreboard (adjusted from your screenshot) ---
        scoreboard = frame[int(h*0.85):h, int(w*0.02):int(w*0.98)]

        # --- Preprocess for OCR ---
        gray = cv2.cvtColor(scoreboard, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # --- OCR ---
        score_text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
        score_text = score_text.strip().replace(" ", "").replace("\n", "")
        runs, wickets = parse_score(score_text)
        if runs is None:
            continue  # skip if OCR failed

        label = None
        if prev_score:
            prev_runs, prev_wickets = prev_score
            run_diff = runs - prev_runs
            wicket_diff = wickets - prev_wickets

            # --- Label highlight ---
            if wicket_diff > 0:
                label = f"WICKET! ({prev_wickets}→{wickets})"
            elif run_diff >= 6:
                label = f"SIX! (+{run_diff})"
            elif run_diff == 4:
                label = f"BOUNDARY! (+{run_diff})"
            elif run_diff > 0:
                label = f"RUN (+{run_diff})"

            if label:
                # Add highlight (2 sec padding before, 5 sec after)
                start = max(t - 2, 0)
                end = min(t + 5, duration)
                highlight_times.append((start, end, label))
                cool_down = 8  # avoid duplicate detections nearby

        prev_score = (runs, wickets)

    cap.release()
    video.close()

    print(f"Found {len(highlight_times)} highlights.")
    for h in highlight_times:
        print(h)

    return highlight_times