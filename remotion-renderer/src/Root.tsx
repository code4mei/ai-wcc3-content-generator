import { Composition } from "remotion";
import { WCC3REEL } from "./WCC3REEL";
import { INTRO_DURATION_FRAMES } from "./IntroCard";
import { OUTRO_DURATION_FRAMES } from "./OutroCard";

type Clip = {
  path: string;
  durationFrames: number;
  layoutMode: "contain" | "verticalCrop";
  label?: string;
};

const TRANSITION_DURATION_FRAMES = 20;

const calculateTotalFrames = (clips: Clip[], transitionDuration: number) => {
  if (clips.length === 0) return INTRO_DURATION_FRAMES + OUTRO_DURATION_FRAMES;
  const sumDurations = clips.reduce(
    (sum, clip) => sum + clip.durationFrames,
    0
  );
  // Transitions: 1 intro->first clip + (clips.length - 1) between clips + 1 last clip->outro
  const numTransitions = clips.length + 1;
  const totalDuration =
    INTRO_DURATION_FRAMES +
    sumDurations +
    OUTRO_DURATION_FRAMES -
    numTransitions * transitionDuration;
  return Math.max(1, totalDuration);
};

export const RemotionRoot: React.FC = () => {
  const defaultClips: Clip[] = [
    {
      path: "final.mp4",
      durationFrames: 900,
      layoutMode: "verticalCrop",
      label: "crowd",
    },
  ];

  return (
    <>
      {/* Instagram Reel — 9:16 portrait */}
      <Composition
        id="WCC3REEL"
        component={WCC3REEL}
        calculateMetadata={({ props }) => {
          const td =
            props.transitionDurationFrames ?? TRANSITION_DURATION_FRAMES;
          return {
            durationInFrames: calculateTotalFrames(props.clips, td),
          };
        }}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{
          clips: defaultClips,
          transitionDurationFrames: TRANSITION_DURATION_FRAMES,
        }}
      />

      {/* YouTube Video — 16:9 landscape */}
      <Composition
        id="WCC3YOUTUBE"
        component={WCC3REEL}
        calculateMetadata={({ props }) => {
          const td =
            props.transitionDurationFrames ?? TRANSITION_DURATION_FRAMES;
          return {
            durationInFrames: calculateTotalFrames(props.clips, td),
          };
        }}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{
          clips: defaultClips,
          transitionDurationFrames: TRANSITION_DURATION_FRAMES,
        }}
      />
    </>
  );
};
