import {Composition } from "remotion";
import {WCC3REEL} from "./WCC3REEL";
import {INTRO_DURATION_FRAMES} from "./IntroCard";

type Clip = {
  path: string;
  durationFrames: number;
  label?: string;
};

const TRANSITION_DURATION_FRAMES = 20;

const calculateTotalFrames = (clips: Clip[], transitionDuration: number) => {
  if (clips.length === 0) return INTRO_DURATION_FRAMES;
  const sumDurations = clips.reduce((sum, clip) => sum + clip.durationFrames, 0);
  // clips.length transitions: 1 intro→first clip + (clips.length - 1) between clips
  const numTransitions = clips.length;
  const totalDuration = INTRO_DURATION_FRAMES + sumDurations - numTransitions * transitionDuration;
  return Math.max(1, totalDuration);
};

export const RemotionRoot:React.FC=() => {
  const defaultClips: Clip[] = [
    {
      path: "final.mp4",
      durationFrames: 900,
      label: "crowd",
    },
  ];
  return (
    <Composition
      id="WCC3REEL"
      component={WCC3REEL}
      calculateMetadata={({props}) => {
        const td = props.transitionDurationFrames ?? TRANSITION_DURATION_FRAMES;
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
  );
};
