import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip    
import numpy as np
import pytesseract
import re
from flask import request
import librosa
from faster_whisper import WhisperModel
import mediapipe as mp

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def detect_scoreboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return False

    horizontal = 0
    vertical = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:
            horizontal += 1
        if abs(x1 - x2) < 10:
            vertical += 1

    # Threshold tuning
    if horizontal > 10 and vertical > 5:
        return True

    return False

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
# ----------------------------
# TRANSCRIPTION USING WHISPER
# ----------------------------
def transcribe_audio(video_path):
    audio_path = video_path.replace(".mp4", "_temp_audio.wav")

    # Extract clean mono audio
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le')
    video.close()

    model = WhisperModel("base", device="cpu",compute_type="float32")  # change to cuda if GPU
    segments, _ = model.transcribe(audio_path)

    transcript_segments = []
    for segment in segments:
        transcript_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "word_count": len(segment.text.split())
        })

    return transcript_segments

def get_speech_peaks(transcript_segments, window_size=4):
    """
    Detect excitement via speech speed increase
    Returns timestamps where speech is fast
    """
    speech_peaks = []

    for segment in transcript_segments:
        duration = segment["end"] - segment["start"]
        if duration <= 0:
            continue

        wps = segment["word_count"] / duration

        # If speech faster than threshold → possible excitement
        if wps > 3.5:   # tune this
            speech_peaks.append(segment["start"])

    return speech_peaks

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
    
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection()
    cap = cv2.VideoCapture("clip.mp4")

    ret, frame = cap.read()
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
    x = bbox.xmin
    y = bbox.ymin
    w = bbox.width
    h = bbox.height

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

def merge_peaks(audio_times, visual_times, ocr_times,speech_times, video_duration):
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
    for t in speech_times:
        candidates.append((t, 0.85, "speech_excitement"))
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

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scoreboard_frames = 0
    checked_frames = 0

    for i in range(0, frame_count, int(fps)):  # 1 frame per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        checked_frames += 1
        if detect_scoreboard(frame):
            scoreboard_frames += 1

    cap.release()

    scoreboard_ratio = scoreboard_frames / max(1, checked_frames)

    if scoreboard_ratio > 0.3:
        layout_mode = "contain"
    else:
        layout_mode = "verticalCrop"
    
    video = VideoFileClip(video_path)
    video_duration = video.duration
    print(f"DEBUG: video.duration={video_duration}")

    audio_peaks = get_audio_peaks(video_path)
    visual_frames, fps = get_visual_peaks(video_path)
    visual_times = [f / fps for f in visual_frames]
    ocr_times = get_ocr_peaks(video_path)
    # NEW AI TRANSCRIPTION
    print("Running Whisper transcription...")
    transcript_segments = transcribe_audio(video_path)

    # NEW speech excitement detection
    speech_peaks = get_speech_peaks(transcript_segments)
    print(f"Speech peaks detected: {len(speech_peaks)}")
    print(f"DEBUG: max audio_peak={max(audio_peaks) if audio_peaks else 'none'}")
    print(f"DEBUG: max visual_time={max(visual_times) if visual_times else 'none'}")
    print(f"DEBUG: max ocr_time={max(ocr_times) if ocr_times else 'none'}")
    print(f"DEBUG: video.duration={video_duration}")
    highlight_times = merge_peaks(audio_peaks, visual_times, ocr_times,speech_peaks, video_duration)

    print(f"Found {len(highlight_times)} highlights.")
    for idx, (start, end, label) in enumerate(highlight_times, 1):
        print(f"{idx}. Start: {start}s  End: {end}s  Type: {label}")
    print("🔥 RETURNING highlight_times from find_highlights():")
    for h in highlight_times[:10]:
        print(h)

    video.close()
    return highlight_times,layout_mode