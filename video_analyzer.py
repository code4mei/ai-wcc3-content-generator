import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import pytesseract
import re
import os
import librosa
from faster_whisper import WhisperModel
import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


# ──────────────────────────────────────────────
# Scoreboard / overlay detection
# ──────────────────────────────────────────────

def detect_scoreboard(frame):
    """
    Detect scoreboard / player-sheet overlay using three fast signals:
      1. Low green-field ratio  -> overlay covers the field
      2. High edge density in top or bottom band -> structured graphics / text
      3. Relaxed Hough-line check -> rectangular overlay borders
    Returns True if any signal fires.
    """
    h, w = frame.shape[:2]

    # Signal 1: green-field ratio (HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = np.sum(green_mask > 0) / (h * w)
    if green_ratio < config.SCOREBOARD_GREEN_RATIO:
        return True

    # Signal 2: edge density in top / bottom third
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for region in [gray[: h // 3, :], gray[2 * h // 3 :, :]]:
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > config.SCOREBOARD_EDGE_DENSITY:
            return True

    # Signal 3: Hough-line fallback
    full_edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(
        full_edges, 1, np.pi / 180,
        threshold=config.SCOREBOARD_HOUGH_THRESHOLD,
        minLineLength=config.SCOREBOARD_HOUGH_MIN_LENGTH,
        maxLineGap=config.SCOREBOARD_HOUGH_MAX_GAP,
    )
    if lines is not None:
        horiz = sum(1 for l in lines if abs(l[0][1] - l[0][3]) < 10)
        vert = sum(1 for l in lines if abs(l[0][0] - l[0][2]) < 10)
        if horiz > config.SCOREBOARD_HORIZ_LINES and vert > config.SCOREBOARD_VERT_LINES:
            return True

    return False


def detect_clip_layout(clip_path):
    """
    Check if a single clip contains scoreboard/player sheet.
    Samples several frames and returns "contain" if scoreboard
    is found in any significant portion, else "verticalCrop".
    """
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0 or fps == 0:
        cap.release()
        return "verticalCrop"

    step = max(1, frame_count // 5)
    scoreboard_hits = 0
    checked = 0

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        checked += 1
        if detect_scoreboard(frame):
            scoreboard_hits += 1

    cap.release()

    if checked > 0 and scoreboard_hits > 0:
        print(f"  Scoreboard detected in {clip_path}: {scoreboard_hits}/{checked} frames")
        return "contain"
    return "verticalCrop"


# ──────────────────────────────────────────────
# Score parsing & OCR
# ──────────────────────────────────────────────

def parse_score(score_text):
    """Parse score string like '125/3' into (runs, wickets)."""
    match = re.match(r'(\d+)/(\d+)', score_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


# ──────────────────────────────────────────────
# Transcription
# ──────────────────────────────────────────────

def transcribe_audio(video_path):
    base, _ = os.path.splitext(video_path)
    audio_path = base + "_temp_audio.wav"

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le')
    video.close()

    model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                         compute_type=config.WHISPER_COMPUTE_TYPE)
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


# ──────────────────────────────────────────────
# Event type classification
# ──────────────────────────────────────────────

def classify_event(label, run_diff=0, wicket_diff=0, transcript_text=""):
    """
    Classify a highlight into a cricket event type based on detection source,
    score changes, and transcript keywords.
    Returns one of: "six", "boundary", "wicket", "catch", "replay", or the original label.
    """
    text_lower = transcript_text.lower()

    # Score-change events are the most reliable
    if label == "score_change":
        if wicket_diff > 0:
            # Check if it's specifically a catch from transcript
            if any(kw in text_lower for kw in config.CATCH_KEYWORDS):
                return "catch"
            return "wicket"
        if run_diff >= config.OCR_RUN_DIFF_SIX:
            return "six"
        if run_diff >= config.OCR_RUN_DIFF_BOUNDARY:
            return "boundary"

    # Transcript keyword matching for other detection types
    if any(kw in text_lower for kw in config.REPLAY_KEYWORDS):
        return "replay"
    if any(kw in text_lower for kw in config.SIX_KEYWORDS):
        return "six"
    if any(kw in text_lower for kw in config.WICKET_KEYWORDS):
        return "wicket"
    if any(kw in text_lower for kw in config.CATCH_KEYWORDS):
        return "catch"
    if any(kw in text_lower for kw in config.BOUNDARY_KEYWORDS):
        return "boundary"

    return label  # fallback to original detection label


# ──────────────────────────────────────────────
# Replay detection
# ──────────────────────────────────────────────

def detect_replay_frames(video_path, sample_interval=2):
    """
    Detect replay sequences in WCC3 footage.
    Replays typically show:
      1. A brief dark flash / wipe transition before and after
      2. Camera angle change (wider shot vs close-up)
      3. Sometimes a 'REPLAY' text watermark

    Returns list of (start_time, end_time) for detected replay segments.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    replays = []
    prev_brightness = None
    in_replay = False
    replay_start = 0

    for t_sec in range(0, int(duration), sample_interval):
        frame_idx = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Detect dark flash (brightness drop below 30)
        if prev_brightness is not None:
            brightness_drop = prev_brightness - brightness

            if brightness_drop > 60 and not in_replay:
                # Sharp brightness drop = potential replay start
                in_replay = True
                replay_start = t_sec
            elif in_replay and brightness_drop < -40:
                # Brightness recovery = replay end
                replay_duration = t_sec - replay_start
                if 2 < replay_duration < 15:
                    replays.append((replay_start, t_sec))
                in_replay = False

        # Also check for "REPLAY" text via OCR on a small region
        h, w = frame.shape[:2]
        top_strip = frame[0:h // 8, w // 4: 3 * w // 4]
        try:
            text = pytesseract.image_to_string(top_strip, config='--psm 7').strip().upper()
            if "REPLAY" in text or "ACTION REPLAY" in text:
                if not in_replay:
                    in_replay = True
                    replay_start = t_sec
        except Exception:
            pass

        prev_brightness = brightness

    # Close any open replay at end
    if in_replay:
        replay_duration = duration - replay_start
        if 2 < replay_duration < 15:
            replays.append((replay_start, duration))

    cap.release()
    print(f"Replay detection: found {len(replays)} replay segments")
    return replays


# ──────────────────────────────────────────────
# Peak detection functions
# ──────────────────────────────────────────────

def get_speech_peaks(transcript_segments):
    """Detect excitement via speech speed increase."""
    speech_peaks = []
    for segment in transcript_segments:
        duration = segment["end"] - segment["start"]
        if duration <= 0:
            continue
        wps = segment["word_count"] / duration
        if wps > config.SPEECH_WPS_THRESHOLD:
            speech_peaks.append(segment["start"])
    return speech_peaks


def get_audio_peaks(video_path):
    """Detect times where audio amplitude spikes (crowd cheers / celebrations)."""
    video = VideoFileClip(video_path)
    audio = video.audio
    duration = video.duration
    times = []
    t = 0
    while t < duration:
        segment = audio.subclipped(t, min(t + config.AUDIO_WINDOW_SIZE, duration))
        samples = segment.to_soundarray(fps=44100)
        amplitude = np.mean(np.abs(samples))
        if amplitude > config.AUDIO_AMPLITUDE_THRESHOLD:
            times.append(t)
        t += config.AUDIO_WINDOW_SIZE
    video.close()
    return times


def get_visual_peaks(video_path):
    """Detect frames where visual changes occur (big six text, celebration flashes)."""
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
        h, w = gray.shape
        roi = gray[int(0.75 * h):h, int(0.2 * w):int(0.8 * w)]

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, roi)
            score = np.sum(diff) / (roi.shape[0] * roi.shape[1])
            if score > config.VISUAL_CHANGE_THRESHOLD:
                highlights.append(frame_idx)

        prev_frame = roi
        frame_idx += 1

    cap.release()
    return highlights, fps


def get_ocr_peaks(video_path):
    """
    Detect highlights from scoreboard changes using OCR.
    Returns list of (timestamp, run_diff, wicket_diff) tuples for event classification.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    prev_score = None
    peaks = []

    for t in range(0, int(duration), config.OCR_SAMPLE_INTERVAL):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape
        scoreboard = frame[int(0.75 * h):h, int(0.2 * w):int(0.8 * w)]
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
            if run_diff >= config.OCR_RUN_DIFF_BOUNDARY or wicket_diff > 0:
                peaks.append((t, run_diff, wicket_diff))
        prev_score = (runs, wickets)

    cap.release()
    return peaks


# ──────────────────────────────────────────────
# Merge peaks
# ──────────────────────────────────────────────

def merge_peaks(audio_times, visual_times, ocr_peaks, speech_times,
                video_duration, transcript_segments=None, format_type="reel"):
    """
    Merge all detection signals into final highlight timestamps.
    Returns list of tuples: (start_time, end_time, event_type)
    where event_type is a classified cricket event like "six", "wicket", etc.
    """
    candidates = []

    for t in audio_times:
        candidates.append((t, config.CONFIDENCE_CROWD, "crowd", 0, 0))

    for t in visual_times:
        candidates.append((t, config.CONFIDENCE_MOTION, "motion", 0, 0))

    # OCR peaks now carry run_diff and wicket_diff for classification
    for t, run_diff, wicket_diff in ocr_peaks:
        candidates.append((t, config.CONFIDENCE_SCORE_CHANGE, "score_change", run_diff, wicket_diff))

    for t in speech_times:
        candidates.append((t, config.CONFIDENCE_SPEECH, "speech_excitement", 0, 0))

    if not candidates:
        print("No highlights detected from any source")
        return []

    candidates.sort(key=lambda x: x[0])

    # Group nearby events
    grouped_events = []
    current_group = [candidates[0]]

    for entry in candidates[1:]:
        if entry[0] - current_group[-1][0] <= config.MERGE_GROUP_WINDOW:
            current_group.append(entry)
        else:
            grouped_events.append(current_group)
            current_group = [entry]
    grouped_events.append(current_group)

    # Pick best event from each group
    best_events = []
    for group in grouped_events:
        best = max(group, key=lambda x: x[1])
        best_events.append(best)

    # Sort by confidence, take top N
    best_events.sort(key=lambda x: x[1], reverse=True)
    max_clips = config.MAX_CLIPS_REEL if format_type == "reel" else config.MAX_CLIPS_YOUTUBE
    top_events = best_events[:max_clips]

    # Re-sort chronologically
    top_events.sort(key=lambda x: x[0])

    print(f"{len(candidates)} candidates -> {len(grouped_events)} groups -> {len(top_events)} selected")

    # Convert to clips with event classification
    highlights = []
    for t, score, label, run_diff, wicket_diff in top_events:
        # Get transcript text near this timestamp for keyword classification
        transcript_text = ""
        if transcript_segments:
            for seg in transcript_segments:
                if seg["start"] <= t + config.POST_ROLL and seg["end"] >= t - config.PRE_ROLL:
                    transcript_text += " " + seg["text"]

        event_type = classify_event(label, run_diff, wicket_diff, transcript_text)
        start = max(0, t - config.PRE_ROLL)
        end = min(video_duration, t + config.POST_ROLL)
        highlights.append((start, end, event_type))

    # Remove overlapping clips
    filtered = []
    for h in highlights:
        if not filtered or h[0] > filtered[-1][1]:
            filtered.append(h)

    print(f"Final highlights selected: {len(filtered)}")
    return filtered


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def find_highlights(video_path, format_type="reel"):
    """
    End-to-end highlight detection using audio + visual + OCR + speech cues.
    Returns: (highlight_times, layout_mode, transcript_segments)
    """
    print("Analyzing video for highlights...")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scoreboard_frames = 0
    checked_frames = 0

    for i in range(0, frame_count, max(1, int(fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        checked_frames += 1
        if detect_scoreboard(frame):
            scoreboard_frames += 1

    cap.release()

    scoreboard_ratio = scoreboard_frames / max(1, checked_frames)
    layout_mode = "contain" if scoreboard_ratio > config.SCOREBOARD_RATIO_THRESHOLD else "verticalCrop"

    video = VideoFileClip(video_path)
    video_duration = video.duration
    print(f"Video duration: {video_duration:.1f}s")

    audio_peaks = get_audio_peaks(video_path)
    visual_frames, vid_fps = get_visual_peaks(video_path)
    visual_times = [f / vid_fps for f in visual_frames] if vid_fps > 0 else []
    ocr_peaks = get_ocr_peaks(video_path)

    print("Running Whisper transcription...")
    transcript_segments = transcribe_audio(video_path)

    speech_peaks = get_speech_peaks(transcript_segments)

    # Detect replay segments
    replay_segments = detect_replay_frames(video_path)

    print(f"Peaks found - audio: {len(audio_peaks)}, visual: {len(visual_times)}, "
          f"ocr: {len(ocr_peaks)}, speech: {len(speech_peaks)}, replays: {len(replay_segments)}")

    highlight_times = merge_peaks(
        audio_peaks, visual_times, ocr_peaks, speech_peaks,
        video_duration, transcript_segments=transcript_segments,
        format_type=format_type
    )

    # Mark highlights that fall within replay segments
    for i, (start, end, event_type) in enumerate(highlight_times):
        for r_start, r_end in replay_segments:
            if start >= r_start and end <= r_end and event_type not in ("six", "wicket", "catch", "boundary"):
                highlight_times[i] = (start, end, "replay")
                break

    print(f"Found {len(highlight_times)} highlights:")
    for idx, (start, end, event_type) in enumerate(highlight_times, 1):
        print(f"  {idx}. {start:.1f}s - {end:.1f}s  [{event_type}]")

    video.close()
    return highlight_times, layout_mode, transcript_segments
