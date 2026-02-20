import React, { useMemo } from "react";
import {
  AbsoluteFill,
  Audio,
  OffthreadVideo,
  staticFile,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { IntroCard, INTRO_DURATION_FRAMES } from "./IntroCard";
import { OutroCard, OUTRO_DURATION_FRAMES } from "./OutroCard";
import { AnimatedText } from "./AnimatedText";

type CaptionData = {
  text: string;
  highlights: string[];
};

type Clip = {
  path: string;
  durationFrames: number;
  caption?: CaptionData;
  layoutMode: "contain" | "verticalCrop";
  eventType?: string;
  faceCenterX?: number;
  faceCenterY?: number;
};

export type WCC3REELProps = {
  clips: Clip[];
  transitionDurationFrames?: number;
};

const TRANSITIONS = [
  fade(),
  slide({ direction: "from-left" }),
  fade(),
  slide({ direction: "from-bottom" }),
];

// A single clip with zoom + animated text overlay
const ClipWithEffects: React.FC<{ clip: Clip }> = ({ clip }) => {
  const isContain = clip.layoutMode === "contain";
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const faceCenterX = clip.faceCenterX ?? 0.5;
  const faceCenterY = clip.faceCenterY ?? 0.5;

  const zoom = interpolate(frame, [0, durationInFrames], [1, 1.08], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ backgroundColor: "black", overflow: "hidden" }}>
      {/* Blurred background video for contain mode */}
      {isContain && (
        <OffthreadVideo
          src={staticFile(clip.path)}
          style={{
            position: "absolute",
            width: "100%",
            height: "100%",
            objectFit: "cover",
            filter: "blur(20px) brightness(0.5)",
            transform: "scale(1.1)",
          }}
        />
      )}
      {/* Main video */}
      <OffthreadVideo
        src={staticFile(clip.path)}
        style={{
          width: "100%",
          height: "100%",
          objectFit: isContain ? "contain" : "cover",
          transform: `scale(${zoom})`,
          transformOrigin: `${faceCenterX * 100}% ${faceCenterY * 100}%`,
        }}
      />
      {clip.caption ? <AnimatedText caption={clip.caption} /> : null}
    </AbsoluteFill>
  );
};

/**
 * Audio ducking: lower music volume during clip playback (commentary),
 * raise it during intro, outro, and transitions.
 */
const useMusicVolume = (
  clips: Clip[],
  transitionDurationFrames: number
): ((frame: number) => number) => {
  const FULL_VOLUME = 0.3;
  const DUCK_VOLUME = 0.1;
  const RAMP_FRAMES = 8; // smooth ramp between volumes

  // Pre-compute clip start frames
  const clipRanges = useMemo(() => {
    const ranges: Array<{ start: number; end: number }> = [];
    // Intro ends at INTRO_DURATION_FRAMES (minus transition overlap)
    let currentFrame = INTRO_DURATION_FRAMES - transitionDurationFrames;

    for (const clip of clips) {
      ranges.push({
        start: currentFrame,
        end: currentFrame + clip.durationFrames,
      });
      currentFrame += clip.durationFrames - transitionDurationFrames;
    }
    return ranges;
  }, [clips, transitionDurationFrames]);

  return (frame: number) => {
    // Check if we're in a clip (duck) or in intro/outro/transition (full)
    for (const range of clipRanges) {
      if (frame >= range.start + RAMP_FRAMES && frame <= range.end - RAMP_FRAMES) {
        return DUCK_VOLUME;
      }
      // Ramp down into clip
      if (frame >= range.start && frame < range.start + RAMP_FRAMES) {
        return interpolate(
          frame,
          [range.start, range.start + RAMP_FRAMES],
          [FULL_VOLUME, DUCK_VOLUME],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );
      }
      // Ramp up out of clip
      if (frame > range.end - RAMP_FRAMES && frame <= range.end) {
        return interpolate(
          frame,
          [range.end - RAMP_FRAMES, range.end],
          [DUCK_VOLUME, FULL_VOLUME],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );
      }
    }
    return FULL_VOLUME;
  };
};

export const WCC3REEL: React.FC<WCC3REELProps> = ({
  clips,
  transitionDurationFrames = 20,
}) => {
  const getMusicVolume = useMusicVolume(clips, transitionDurationFrames);

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      {/* Background music with ducking */}
      <Audio src={staticFile("bgmusic.mp3")} volume={getMusicVolume} />

      <TransitionSeries>
        {/* Intro card */}
        <TransitionSeries.Sequence
          key="intro"
          durationInFrames={INTRO_DURATION_FRAMES}
        >
          <IntroCard />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          key="intro-trans"
          presentation={fade()}
          timing={linearTiming({
            durationInFrames: transitionDurationFrames,
          })}
        />

        {/* Clips with transitions */}
        {clips.flatMap((clip, index) => {
          const elements: React.ReactNode[] = [];

          elements.push(
            <TransitionSeries.Sequence
              key={`clip-${index}`}
              durationInFrames={clip.durationFrames}
            >
              <ClipWithEffects clip={clip} />
            </TransitionSeries.Sequence>
          );

          if (
            index < clips.length - 1 &&
            clip.durationFrames > transitionDurationFrames &&
            clips[index + 1].durationFrames > transitionDurationFrames
          ) {
            elements.push(
              <TransitionSeries.Transition
                key={`trans-${index}`}
                presentation={TRANSITIONS[index % TRANSITIONS.length]}
                timing={linearTiming({
                  durationInFrames: transitionDurationFrames,
                })}
              />
            );
          }

          return elements;
        })}

        {/* Outro transition */}
        <TransitionSeries.Transition
          key="outro-trans"
          presentation={fade()}
          timing={linearTiming({
            durationInFrames: transitionDurationFrames,
          })}
        />

        {/* Outro card */}
        <TransitionSeries.Sequence
          key="outro"
          durationInFrames={OUTRO_DURATION_FRAMES}
        >
          <OutroCard />
        </TransitionSeries.Sequence>
      </TransitionSeries>
    </AbsoluteFill>
  );
};
