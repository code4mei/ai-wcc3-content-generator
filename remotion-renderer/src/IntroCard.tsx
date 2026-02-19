import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const INTRO_DURATION_FRAMES = 75; // 2.5 seconds at 30fps

export const IntroCard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Title spring animation
  const titleSpring = spring({
    frame: frame - 5,
    fps,
    config: { damping: 14, stiffness: 150, mass: 0.8 },
  });

  const titleScale = interpolate(titleSpring, [0, 1], [0, 1]);
  const titleY = interpolate(titleSpring, [0, 1], [60, 0]);

  // Subtitle appears after title
  const subtitleSpring = spring({
    frame: frame - 25,
    fps,
    config: { damping: 14, stiffness: 150, mass: 0.6 },
  });

  const subtitleOpacity = interpolate(subtitleSpring, [0, 1], [0, 1]);
  const subtitleY = interpolate(subtitleSpring, [0, 1], [20, 0]);

  // Decorative line grows
  const lineWidth = interpolate(
    frame,
    [15, 40],
    [0, 200],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Fade out at end
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 15, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#0a0a0a",
        justifyContent: "center",
        alignItems: "center",
        opacity: fadeOut,
      }}
    >
      {/* Title */}
      <div
        style={{
          transform: `scale(${titleScale}) translateY(${titleY}px)`,
          color: "#ffffff",
          fontSize: 80,
          fontWeight: 900,
          fontFamily: "Arial Black, Arial, sans-serif",
          textAlign: "center",
          letterSpacing: 6,
          textShadow: "0 0 30px #ffcc0066, 0 0 60px #ffcc0033",
        }}
      >
        WCC3
      </div>

      {/* Decorative line */}
      <div
        style={{
          width: lineWidth,
          height: 3,
          background: "linear-gradient(90deg, transparent, #ffcc00, transparent)",
          marginTop: 16,
          marginBottom: 16,
        }}
      />

      {/* Subtitle */}
      <div
        style={{
          transform: `translateY(${subtitleY}px)`,
          opacity: subtitleOpacity,
          color: "#ffcc00",
          fontSize: 32,
          fontWeight: 700,
          fontFamily: "Arial, sans-serif",
          textAlign: "center",
          letterSpacing: 8,
          textTransform: "uppercase",
        }}
      >
        Highlights
      </div>
    </AbsoluteFill>
  );
};
