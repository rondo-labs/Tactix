export interface TrackingPlayer {
  id: number;
  team: "A" | "B" | "REFEREE" | "GOALKEEPER" | "UNKNOWN";
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  speed?: number;
  jersey?: string;
  pressure?: number;
}

export interface TrackingBall {
  x: number;
  y: number;
}

export interface TrackingFrame {
  i: number;
  players: TrackingPlayer[];
  ball?: TrackingBall;
}

export interface TrackingEvent {
  frame: number;
  type: "shot" | "goal" | "corner" | "free_kick" | "foul" | "offside" | "substitution" | "card" | "other";
  team?: "A" | "B";
  player_id?: number;
  outcome?: string;
  x?: number;
  y?: number;
  label?: string;
}

export interface TrackingMeta {
  fps: number;
  total_frames: number;
  canvas: { width: number; height: number };
  pitch: { length: number; width: number };
}

export interface TrackingData {
  meta: TrackingMeta;
  frames: TrackingFrame[];
  events?: TrackingEvent[];
}
