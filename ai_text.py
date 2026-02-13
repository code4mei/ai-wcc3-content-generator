from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
# from moviepy.video.io.VideoFileClip import VideoFileClip
# from moviepy.video.VideoClip import TextClip
# from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
# from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
# import cv2
import numpy as np
import random
import os

# from numpy.core.fromnumeric import clip
 
def get_event_text(label):
    if label == "six":
        return "MASSIVE SIX!"
    elif label == "four":
        return "CRISP FOUR!"
    elif label == "wicket":
        return "WICKET DOWN!"
    elif label == "crowd":
        return "WHAT A MOMENT!"
    else:
        return "TENSE BUILD-UP!"
    # Volume = average audio loudness of the clip
    # if label == "score_change":
    #     return random.choice([
    #         "IT'S A BOUNDARY!",
    #         "FOUR RUNS!",
    #         "WHAT A SHOT!",
    #         "SMASHED!"
    #     ])

    # elif label == "crowd":
    #     return random.choice([
    #         "MASSIVE SIX!",
    #         "OUT OF THE PARK!",
    #         "UNBELIEVABLE SIX!",
    #         "MAXIMUM!"
    #     ])

    # else:
    #     return random.choice([
    #         "BIG MOMENT!",
    #         "GAME ON!",
    #         "PRESSURE BUILDING!"
    #     ])
def add_ai_text_to_clips(clips,run_folder):
    
    # Adds AI-style text overlays (supers) on highlight clips.

    # Parameters:
    #     clip_paths (list) → list of highlight clip file paths

    # Returns:
    #     list of final video paths
    # Possible cricket highlight texts

    final_video_paths = []

    # Loop through each highlight clip
    for i, (clip_path,label) in enumerate(clips):
        video = VideoFileClip(clip_path)
        # audio = video.audio
        # samples = audio.to_soundarray(fps=44100) #audio loudness analysis at 44.1kHz
        # volume = np.mean(np.abs(samples))  # get average volume of audio
        # print(f"DEBUG volume for clip {i+1}: {volume}")

        # Create text overlay
        text = get_event_text(label)

        #cinematic effects
        # Slow motion first 2 seconds
        if label in ["six", "four"]:
            slow_part = video.subclipped(0, min(1.5, video.duration)).with_speed_scaled(0.7)
        # Normal rest of clip
            normal_part = video.subclipped(min(1.5, video.duration), video.duration)
            video = concatenate_videoclips([slow_part, normal_part])

        # Smooth zoom-in effect
        # def zoom_in(frame, t):
        #     h, w = frame.shape[:2]
        #     zoom = 1 + 0.08 * t  # gradual zoom
        #     new_w, new_h = int(w/zoom), int(h/zoom)
        #     x1 = (w - new_w)//2
        #     y1 = (h - new_h)//2
        #     frame = frame[y1:y1+new_h, x1:x1+new_w]
        #     return cv2.resize(frame, (w, h))
        # video = video.with_frame_processor(zoom_in)
        # video = video.resized(lambda t: 1 + 0.05 * t)
        text_clip = TextClip(
            text=text,
            font='C:/Windows/Fonts/arialbd.ttf',                
            font_size=60,
            color='yellow',      
            stroke_color='black',
            stroke_width=3,
            method='label',         # better rendering
            size=video.size            # match video size
        )

        # Position text in center and match video duration
        text_clip = text_clip.with_position("center","bottom").with_duration(video.duration)

        # Combine video + text
        final_video = CompositeVideoClip([video, text_clip])

        output_path = os.path.join(run_folder, f"text_{i+1}.mp4")

        # Save final video
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        final_video_paths.append(output_path)

        video.close()
        final_video.close()

    return final_video_paths
