import React from "react";
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
  faceCenterX?: number;   // 0 → 1
  faceCenterY?: number;   // optional for future
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

// Ken Burns zoom effect wrapper
const ZoomClip: React.FC<{ src: string }> = ({ src }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  // Slow zoom from 1.0 to 1.12 over the clip duration
  const scale = interpolate(frame, [0, durationInFrames], [1, 1.12], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          transform: `scale(${scale})`,
          transformOrigin: "center center",
        }}
      >
        <OffthreadVideo
          src={staticFile(src)}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

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
      <OffthreadVideo
        src={staticFile(clip.path)}
        style={{
          width: isContain ? "100%" : "auto",
          height: isContain ? "auto" : "100%",
          objectFit: isContain ? "contain" : "cover",
          transform: `scale(${zoom})`,
          transformOrigin: `${faceCenterX * 100}% ${faceCenterY * 100}%`,
        }}
      />
      {clip.caption ? <AnimatedText caption={clip.caption} /> : null}
    </AbsoluteFill>
  );
};

export const WCC3REEL: React.FC<WCC3REELProps> = ({
  clips,
  transitionDurationFrames = 20,
}) => {
  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      {/* Background music - plays throughout the entire reel */}
      <Audio src={staticFile("bgmusic.mp3")} volume={0.3} />

      {/* Single TransitionSeries: intro card followed by clips */}
      <TransitionSeries>
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

          // Transition between clips (not after the last one)
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
      </TransitionSeries>
    </AbsoluteFill>
  );
};
