import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
REMOTION_DIR = os.path.join(BASE_DIR, "remotion-renderer")
REMOTION_PUBLIC_DIR = os.path.join(REMOTION_DIR, "public")

# External tools — resolved via env var > PATH > Windows default
TESSERACT_CMD = (
    os.environ.get("TESSERACT_CMD")
    or shutil.which("tesseract")
    or (r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None)
)
NPX_CMD = (
    os.environ.get("NPX_CMD")
    or shutil.which("npx")
    or (r"C:\Program Files\nodejs\npx.cmd"
        if os.path.isfile(r"C:\Program Files\nodejs\npx.cmd") else None)
)

# ──────────────────────────────────────────────
# Upload validation
# ──────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
MAX_FILE_SIZE_MB = 500

# ──────────────────────────────────────────────
# Whisper transcription
# ──────────────────────────────────────────────
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"          # change to "cuda" for GPU
WHISPER_COMPUTE_TYPE = "float32"

# ──────────────────────────────────────────────
# Highlight detection thresholds
# ──────────────────────────────────────────────
# Audio peak detection
AUDIO_WINDOW_SIZE = 0.5         # seconds per window
AUDIO_AMPLITUDE_THRESHOLD = 0.05

# Visual peak detection
VISUAL_CHANGE_THRESHOLD = 25    # normalized frame diff

# OCR scoreboard detection
OCR_SAMPLE_INTERVAL = 1         # seconds between OCR checks
OCR_RUN_DIFF_BOUNDARY = 4      # runs gained to count as boundary
OCR_RUN_DIFF_SIX = 6           # runs gained to count as six

# Speech excitement detection
SPEECH_WPS_THRESHOLD = 3.5      # words per second = excitement

# Scoreboard overlay detection
SCOREBOARD_GREEN_RATIO = 0.10   # below this = overlay covering field
SCOREBOARD_EDGE_DENSITY = 0.15  # above this = dense text/graphics
SCOREBOARD_HOUGH_THRESHOLD = 80
SCOREBOARD_HOUGH_MIN_LENGTH = 80
SCOREBOARD_HOUGH_MAX_GAP = 15
SCOREBOARD_HORIZ_LINES = 5
SCOREBOARD_VERT_LINES = 3
SCOREBOARD_RATIO_THRESHOLD = 0.3  # ratio above which = "contain" layout

# Scene change detection
SCENE_CHANGE_THRESHOLD = 60     # normalized diff score
SCENE_CHANGE_COOLDOWN = 4       # seconds between detections

# ──────────────────────────────────────────────
# Highlight merging
# ──────────────────────────────────────────────
MERGE_GROUP_WINDOW = 2          # seconds — events within this are same moment
PRE_ROLL = 2                    # seconds before detected peak
POST_ROLL = 3                   # seconds after detected peak
MAX_CLIPS_REEL = 6              # max clips for Instagram reel
MAX_CLIPS_YOUTUBE = 20          # max clips for YouTube video

# Confidence scores per detection type
CONFIDENCE_CROWD = 1.0
CONFIDENCE_MOTION = 0.9
CONFIDENCE_SCORE_CHANGE = 0.95
CONFIDENCE_SPEECH = 0.85
CONFIDENCE_SCENE = 0.5

# ──────────────────────────────────────────────
# Clip processing
# ──────────────────────────────────────────────
MIN_CLIP_DURATION = 2           # seconds — skip shorter clips
MAX_CLIP_DURATION = 6           # seconds — cap clip length
SLOW_MO_DURATION = 1.5          # seconds of slow-mo at clip start
SLOW_MO_SPEED = 0.7             # playback speed for slow-mo section

# ──────────────────────────────────────────────
# Caption generation
# ──────────────────────────────────────────────
MAX_CAPTION_WORDS = 6           # max words in caption overlay

# ──────────────────────────────────────────────
# Remotion rendering
# ──────────────────────────────────────────────
REEL_WIDTH = 1080
REEL_HEIGHT = 1920
REEL_FPS = 30
YOUTUBE_WIDTH = 1920
YOUTUBE_HEIGHT = 1080
YOUTUBE_FPS = 30
TRANSITION_DURATION_FRAMES = 20
INTRO_DURATION_FRAMES = 75      # 2.5 seconds at 30fps
OUTRO_DURATION_FRAMES = 90      # 3 seconds at 30fps
BG_MUSIC_VOLUME = 0.3
BG_MUSIC_DUCK_VOLUME = 0.1      # volume during commentary

# ──────────────────────────────────────────────
# Event classification keywords (cricket-specific)
# ──────────────────────────────────────────────
SIX_KEYWORDS = ["six", "sixer", "massive", "huge", "out of the ground",
                 "into the stands", "over the rope", "maximum"]
BOUNDARY_KEYWORDS = ["four", "boundary", "cover drive", "cut shot",
                      "pull shot", "sweep", "through the gap"]
WICKET_KEYWORDS = ["wicket", "bowled", "caught", "out", "stumped",
                    "lbw", "run out", "clean bowled", "gone", "walks back"]
CATCH_KEYWORDS = ["catch", "caught", "what a catch", "stunning catch",
                   "diving catch", "one-handed"]
REPLAY_KEYWORDS = ["replay", "let's see that again", "action replay",
                    "watch it again", "look at this"]

# ──────────────────────────────────────────────
# Cricket caption templates (per event type)
# ──────────────────────────────────────────────
CAPTION_TEMPLATES = {
    "six": [
        "MASSIVE SIX!",
        "INTO THE STANDS!",
        "WHAT A HIT!",
        "OUT OF THE GROUND!",
        "SIX! MAXIMUM!",
        "SMASHED FOR SIX!",
    ],
    "boundary": [
        "FOUR RUNS!",
        "TO THE BOUNDARY!",
        "BEAUTIFUL SHOT!",
        "RACES TO THE FENCE!",
        "WHAT A DRIVE!",
        "PIERCES THE FIELD!",
    ],
    "wicket": [
        "WICKET! HE'S OUT!",
        "CLEAN BOWLED!",
        "GONE! BIG WICKET!",
        "BREAKTHROUGH!",
        "TIMBER! STUMPS FLYING!",
        "WHAT A DELIVERY!",
    ],
    "catch": [
        "CAUGHT! WHAT A CATCH!",
        "STUNNING CATCH!",
        "SAFE HANDS!",
        "TAKEN BRILLIANTLY!",
        "SPECTACULAR GRAB!",
        "FLYING CATCH!",
    ],
    "replay": [
        "LOOK AT THIS AGAIN!",
        "INCREDIBLE MOMENT!",
        "WHAT A PLAY!",
    ],
    "crowd": [
        "THE CROWD GOES WILD!",
        "LISTEN TO THAT ROAR!",
        "WHAT A MOMENT!",
        "INCREDIBLE SCENES!",
    ],
    "default": [
        "WHAT A MOMENT!",
        "BRILLIANT PLAY!",
        "INTENSE ACTION!",
        "GAME CHANGER!",
    ],
}
