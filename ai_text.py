from moviepy import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import os
import random
import re
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "that",
    "this", "to", "of", "and", "it", "in", "on",
    "for", "with", "as", "at", "by", "from"
}

HIGH_ENERGY_WORDS = [
    "oh", "wow", "what", "unbelievable",
    "incredible", "no way", "insane",
    "massive", "huge", "gone", "finish",
    "knockout", "goal", "out", "score"
]

LOW_ENERGY_WORDS = [
    "build up", "defensive", "waiting",
    "holding", "passing", "strategy"
]
def detect_energy_level(text):
    text_lower = text.lower()

    high_hits = sum(word in text_lower for word in HIGH_ENERGY_WORDS)
    low_hits = sum(word in text_lower for word in LOW_ENERGY_WORDS)

    if high_hits >= 2:
        return "high"
    elif low_hits >= 2:
        return "low"
    else:
        return "medium"

def extract_transcript_for_clip(transcript_segments, start_time, end_time):
    """
    Extract transcript text within highlight time range
    """
    clip_text = []
    PRE_ROLL = 0.8   # capture buildup
    POST_ROLL = 0.2  # slight cushion

    adjusted_start = start_time - PRE_ROLL
    adjusted_end = end_time + POST_ROLL

    for segment in transcript_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]

        # If segment overlaps with highlight window
        if seg_start < adjusted_end and seg_end > adjusted_start:
            clip_text.append(segment["text"].strip())

    return " ".join(clip_text)

def generate_caption_from_text(text):
    """
    Simple smart caption generator
    """
    if not text:
        return None
    # Split into sentences
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return None

    # Score sentences
    sentence_scores = []
    for s in sentences:
        words = re.findall(r"\b\w+\b", s.lower())
        filtered = [w for w in words if w not in STOPWORDS]

        if not filtered:
            continue

        score = 0
        freq = Counter(filtered)

        for w in filtered:
            score += len(w) * 0.3
            score += freq[w] * 1.5

        sentence_scores.append((s, score))

    if not sentence_scores:
        return None

    # Pick highest scoring sentence
    best_sentence = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[0][0]

    # Trim to max 6 words for punchiness
    words = best_sentence.split()
    short_phrase = " ".join(words[:6]).upper()

    # Highlight 1–2 longest words
    cleaned_words = re.findall(r"\b\w+\b", best_sentence)
    longest = sorted(cleaned_words, key=len, reverse=True)[:2]
    highlights = [w.upper() for w in longest]

    return {
        "text": short_phrase,
        "highlights": highlights
    }
    
def process_clips(clips, run_folder, transcript_segments):
    """
    Apply cinematic effects (slow-mo) to clips.
    Generate AI captions from transcript.
    Returns list of (output_path, label, text) tuples.
    """
    processed = []

    for i, clip_data in enumerate(clips):

        # If clips include timing info
        if len(clip_data) == 4:
            clip_path, label, start_time, end_time = clip_data
        else:
            # fallback if timing not included
            clip_path, label = clip_data
            start_time = 0
            end_time = 0

        video = VideoFileClip(clip_path)

        # Extract transcript for this highlight
        clip_text = extract_transcript_for_clip(
            transcript_segments,
            start_time,
            end_time
        )

        # Generate smart caption
        text = generate_caption_from_text(clip_text)
        # Slow motion for big shots
        if label in ["six", "four"]:
            slow_part = video.subclipped(0, min(1.5, video.duration)).with_speed_scaled(0.7)
            normal_part = video.subclipped(min(1.5, video.duration), video.duration)
            video = concatenate_videoclips([slow_part, normal_part])

        video = CompositeVideoClip([video])

        output_path = os.path.join(run_folder, f"clip_{i+1}.mp4")
        video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None,
                              ffmpeg_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"])
        processed.append((output_path, label, text))

        video.close()

    return processed
