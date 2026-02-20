import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const OUTRO_DURATION_FRAMES = 90; // 3 seconds at 30fps

export const OutroCard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Fade in
  const fadeIn = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Title spring animation
  const titleSpring = spring({
    frame: frame - 5,
    fps,
    config: { damping: 14, stiffness: 120, mass: 0.8 },
  });

  const titleScale = interpolate(titleSpring, [0, 1], [0.5, 1]);
  const titleY = interpolate(titleSpring, [0, 1], [40, 0]);

  // Subscribe CTA appears after title
  const ctaSpring = spring({
    frame: frame - 30,
    fps,
    config: { damping: 12, stiffness: 100, mass: 0.6 },
  });

  const ctaOpacity = interpolate(ctaSpring, [0, 1], [0, 1]);
  const ctaY = interpolate(ctaSpring, [0, 1], [20, 0]);

  // Decorative line grows
  const lineWidth = interpolate(frame, [10, 35], [0, 160], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

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
        opacity: fadeIn * fadeOut,
      }}
    >
      {/* Title */}
      <div
        style={{
          transform: `scale(${titleScale}) translateY(${titleY}px)`,
          color: "#ffffff",
          fontSize: 56,
          fontWeight: 900,
          fontFamily: "Arial Black, Arial, sans-serif",
          textAlign: "center",
          letterSpacing: 4,
          textShadow: "0 0 20px #ffcc0066, 0 0 40px #ffcc0033",
        }}
      >
        WCC3
      </div>

      {/* Decorative line */}
      <div
        style={{
          width: lineWidth,
          height: 2,
          background:
            "linear-gradient(90deg, transparent, #ffcc00, transparent)",
          marginTop: 12,
          marginBottom: 12,
        }}
      />

      {/* CTA text */}
      <div
        style={{
          transform: `translateY(${ctaY}px)`,
          opacity: ctaOpacity,
          color: "#ffcc00",
          fontSize: 24,
          fontWeight: 700,
          fontFamily: "Arial, sans-serif",
          textAlign: "center",
          letterSpacing: 6,
          textTransform: "uppercase",
        }}
      >
        Like & Subscribe
      </div>

      {/* Subtitle */}
      <div
        style={{
          marginTop: 16,
          opacity: ctaOpacity,
          color: "#888",
          fontSize: 16,
          fontWeight: 400,
          fontFamily: "Arial, sans-serif",
          textAlign: "center",
          letterSpacing: 2,
        }}
      >
        More highlights coming soon
      </div>
    </AbsoluteFill>
  );
};
