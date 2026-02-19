import React from "react";
import {
  interpolate,
  useCurrentFrame,
} from "remotion";

interface CaptionData {
  text: string;
  highlights: string[];
}

interface Props {
  caption: CaptionData;
}

const getFontSize = (text: string): number => {
  const len = text.length;
  if (len <= 15) return 72;
  if (len <= 25) return 62;
  if (len <= 35) return 52;
  return 44;
};

export const AnimatedText: React.FC<Props> = ({ caption }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(
    frame,
    [0, 15],
    [0, 1],
    { extrapolateLeft: "clamp" }
  );
  const translateY = interpolate(frame, [0, 15], [20,0],{
  extrapolateRight: "clamp",
  });
  const fontSize = getFontSize(caption.text);
  const blur = interpolate(frame, [0, 6], [8,0]);

  // Split long text into max 2 lines for balanced layout
  const words = caption.text.split(" ");
  let lines: string[];
  if (words.length > 3) {
    const mid = Math.ceil(words.length / 2);
    lines = [
      words.slice(0, mid).join(" "),
      words.slice(mid).join(" "),
    ];
  } else {
    lines = [caption.text];
  }

  return (
    <div
      style={{
        position: "absolute",
        bottom: 120,
        left: 0,
        right: 0,
        display: "flex",
        justifyContent: "center",
        textAlign: "center",
        padding: "0 48px",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          fontSize,
          fontWeight: 900,
          lineHeight: 1.2,
          opacity,
          maxWidth: "100%",
          overflowWrap: "break-word",
          wordBreak: "break-word",
          transform: `translateY(${translateY}px)`,
          filter: `blur(${blur}px)`
        }}
      >
        {lines.map((line, i) => (
          <div key={i}>
            {line.split(" ").map((word, index) => {
              const isHighlighted = caption.highlights.includes(word);
              return (
                <span
                  key={index}
                  style={{
                    marginRight: 12,
                    display: "inline-block",
                    color: isHighlighted ? "#00E5A8" : "white",
                    textShadow: isHighlighted
                      ? "0px 0px 12px rgba(0,229,168,0.6), 2px 2px 6px rgba(0,0,0,0.7)"
                      : "2px 2px 6px rgba(0,0,0,0.7)",
                  }}
                >
                  {word}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};
