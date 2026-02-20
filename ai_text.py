from moviepy import VideoFileClip, concatenate_videoclips, CompositeVideoClip
import os
import random
import re
from collections import Counter
import config

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "that",
    "this", "to", "of", "and", "it", "in", "on",
    "for", "with", "as", "at", "by", "from"
}

# Flatten all cricket keywords for scoring boost
_ALL_CRICKET_KEYWORDS = set(
    kw for kwlist in [config.SIX_KEYWORDS, config.WICKET_KEYWORDS,
                      config.BOUNDARY_KEYWORDS, config.CATCH_KEYWORDS]
    for kw in kwlist
)


def detect_energy_level(text):
    text_lower = text.lower()
    high_hits = sum(
        word in text_lower
        for word in config.SIX_KEYWORDS + config.WICKET_KEYWORDS + config.CATCH_KEYWORDS
    )
    low_hits = sum(
        word in text_lower
        for word in ["build up", "defensive", "waiting", "holding", "passing", "strategy"]
    )
    if high_hits >= 2:
        return "high"
    elif low_hits >= 2:
        return "low"
    return "medium"


def extract_transcript_for_clip(transcript_segments, start_time, end_time):
    """Extract transcript text within highlight time range."""
    clip_text = []
    PRE_ROLL = 0.8
    POST_ROLL = 0.2

    adjusted_start = start_time - PRE_ROLL
    adjusted_end = end_time + POST_ROLL

    for segment in transcript_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        if seg_start < adjusted_end and seg_end > adjusted_start:
            clip_text.append(segment["text"].strip())

    return " ".join(clip_text)


def generate_caption_from_text(text, event_type="default", used_captions=None):
    """
    Generate a caption for a clip. Uses cricket-specific templates when
    the transcript text is too short or generic, otherwise extracts from transcript.
    """
    if used_captions is None:
        used_captions = set()

    # Try transcript-based caption first if we have enough text
    transcript_caption = _caption_from_transcript(text, used_captions)
    if transcript_caption:
        return transcript_caption

    # Fall back to event-type template
    return _caption_from_template(event_type, used_captions)


def _caption_from_transcript(text, used_captions):
    """Generate caption from transcript text. Returns None if text is insufficient."""
    if not text or len(text.strip()) < 10:
        return None

    sentences = re.split(r"[.!?,;]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return None

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
            # Boost cricket keywords
            if w in _ALL_CRICKET_KEYWORDS:
                score += 3.0

        sentence_scores.append((s, score))

    if not sentence_scores:
        return None

    ranked = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    for sentence, _score in ranked:
        phrase = " ".join(sentence.split()[:config.MAX_CAPTION_WORDS]).upper()
        if phrase not in used_captions:
            phrase_words = re.findall(r"\b\w+\b", phrase)
            longest = sorted(phrase_words, key=len, reverse=True)[:2]
            highlights = [w.upper() for w in longest]
            return {"text": phrase, "highlights": highlights}

    return None


def _caption_from_template(event_type, used_captions):
    """Generate caption from cricket-specific templates."""
    templates = config.CAPTION_TEMPLATES.get(event_type, config.CAPTION_TEMPLATES["default"])

    available = [t for t in templates if t not in used_captions]
    if not available:
        available = templates

    caption_text = random.choice(available)
    words = caption_text.split()
    longest = sorted(words, key=len, reverse=True)[:2]
    highlights = [w.upper().rstrip("!") for w in longest]

    return {"text": caption_text, "highlights": highlights}


def process_clips(clips, run_folder, transcript_segments):
    """
    Apply cinematic effects (slow-mo) to clips.
    Generate AI captions from transcript (with deduplication).
    Returns list of (output_path, event_type, caption_dict) tuples.
    """
    processed = []
    used_captions = set()

    for i, clip_data in enumerate(clips):
        if len(clip_data) == 4:
            clip_path, event_type, start_time, end_time = clip_data
        else:
            clip_path, event_type = clip_data
            start_time = 0
            end_time = 0

        video = VideoFileClip(clip_path)

        # Extract transcript for this highlight
        clip_text = extract_transcript_for_clip(
            transcript_segments, start_time, end_time
        )

        # Generate smart caption with event type awareness
        caption = generate_caption_from_text(clip_text, event_type=event_type,
                                             used_captions=used_captions)
        if caption and caption.get("text"):
            used_captions.add(caption["text"])

        # Slow motion for big shots (sixes and boundaries)
        if event_type in ("six", "boundary", "four"):
            slow_end = min(config.SLOW_MO_DURATION, video.duration)
            slow_part = video.subclipped(0, slow_end).with_speed_scaled(config.SLOW_MO_SPEED)
            if slow_end < video.duration:
                normal_part = video.subclipped(slow_end, video.duration)
                video = concatenate_videoclips([slow_part, normal_part])
            else:
                video = slow_part

        video = CompositeVideoClip([video])

        output_path = os.path.join(run_folder, f"clip_{i + 1}.mp4")
        video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None,
                              ffmpeg_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"])
        processed.append((output_path, event_type, caption))

        video.close()

    return processed
