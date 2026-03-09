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

export interface Formation {
  frame_start: number;
  frame_end: number;
  team: "A" | "B";
  name: string;
  lines: number[];
}

export interface Shot {
  frame: number;
  team: "A" | "B";
  player_id?: number;
  x: number;
  y: number;
  outcome: string;
  on_target?: boolean;
  xg?: number;
}

export interface PassSonarEntry {
  sectors: number[];
  x: number;
  y: number;
  team: "A" | "B";
}

export interface BuildupSequence {
  id: number;
  team: "A" | "B";
  frame_start: number;
  frame_end: number;
  passes: { from_id: number; to_id: number; frame: number; x: number; y: number; end_x: number; end_y: number }[];
  outcome?: string;
}

export interface DuelData {
  grid_a: number[][];
  grid_b: number[][];
}

export interface TransitionData {
  team: "A" | "B";
  frame: number;
  type: "attack_to_defense" | "defense_to_attack";
  x: number;
  y: number;
  duration_frames?: number;
  outcome?: string;
}

export interface SetPieceData {
  corners: { frame: number; team: "A" | "B"; outcome?: string }[];
  free_kicks: { frame: number; team: "A" | "B"; x: number; y: number; outcome?: string }[];
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
  formations?: Formation[];
  shots?: Shot[];
  pass_sonar?: Record<string, PassSonarEntry>;
  zone14?: Record<string, number[][]>;
  buildups?: BuildupSequence[];
  duels?: DuelData;
  transitions?: TransitionData[];
  set_pieces?: SetPieceData;
}
