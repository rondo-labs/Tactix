"""Generate a demo viewer_data.json for testing the Tactix web frontend.

Usage:
    python web/scripts/generate_demo_data.py [--frames 500] [--fps 25] [--out web/demo_data.json]
"""

import json
import math
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out", default="web/demo_data.json")
    args = parser.parse_args()

    random.seed(42)
    N = args.frames
    FPS = args.fps
    CANVAS_W, CANVAS_H = 1559, 1010

    # ── Player definitions ──
    team_a = [
        {"id": 1, "jersey": "1", "team": "GOALKEEPER", "home_x": 80, "home_y": 505},
        {"id": 2, "jersey": "2", "team": "A", "home_x": 250, "home_y": 150},
        {"id": 3, "jersey": "5", "team": "A", "home_x": 250, "home_y": 400},
        {"id": 4, "jersey": "6", "team": "A", "home_x": 250, "home_y": 610},
        {"id": 5, "jersey": "3", "team": "A", "home_x": 250, "home_y": 860},
        {"id": 6, "jersey": "8", "team": "A", "home_x": 480, "home_y": 300},
        {"id": 7, "jersey": "10", "team": "A", "home_x": 480, "home_y": 505},
        {"id": 8, "jersey": "4", "team": "A", "home_x": 480, "home_y": 710},
        {"id": 9, "jersey": "7", "team": "A", "home_x": 680, "home_y": 200},
        {"id": 10, "jersey": "9", "team": "A", "home_x": 680, "home_y": 505},
        {"id": 11, "jersey": "11", "team": "A", "home_x": 680, "home_y": 810},
    ]

    team_b = [
        {"id": 12, "jersey": "1", "team": "GOALKEEPER", "home_x": 1480, "home_y": 505},
        {"id": 13, "jersey": "2", "team": "B", "home_x": 1310, "home_y": 150},
        {"id": 14, "jersey": "5", "team": "B", "home_x": 1310, "home_y": 400},
        {"id": 15, "jersey": "6", "team": "B", "home_x": 1310, "home_y": 610},
        {"id": 16, "jersey": "3", "team": "B", "home_x": 1310, "home_y": 860},
        {"id": 17, "jersey": "8", "team": "B", "home_x": 1080, "home_y": 300},
        {"id": 18, "jersey": "10", "team": "B", "home_x": 1080, "home_y": 505},
        {"id": 19, "jersey": "4", "team": "B", "home_x": 1080, "home_y": 710},
        {"id": 20, "jersey": "7", "team": "B", "home_x": 880, "home_y": 200},
        {"id": 21, "jersey": "9", "team": "B", "home_x": 880, "home_y": 505},
        {"id": 22, "jersey": "11", "team": "B", "home_x": 880, "home_y": 810},
    ]

    referee = {"id": 23, "jersey": "R", "team": "REFEREE", "home_x": 780, "home_y": 505}

    all_players = team_a + team_b + [referee]

    # ── Generate frames ──
    positions = {p["id"]: [float(p["home_x"]), float(p["home_y"])] for p in all_players}
    velocities = {p["id"]: [0.0, 0.0] for p in all_players}

    ball_x, ball_y = 780.0, 505.0
    ball_vx, ball_vy = 0.0, 0.0

    frames = []
    for f in range(N):
        t = f / N  # normalized time

        # Move players with random walk + drift toward home
        for p in all_players:
            pid = p["id"]
            hx, hy = p["home_x"], p["home_y"]

            # Sinusoidal team shift (attack/defend phases)
            phase_shift = 150 * math.sin(2 * math.pi * t * 3)
            if p["team"] in ("A", "GOALKEEPER") and p in team_a:
                target_x = hx + phase_shift
            elif p["team"] in ("B", "GOALKEEPER") and p in team_b:
                target_x = hx + phase_shift
            else:
                target_x = hx + phase_shift * 0.5
            target_y = hy

            # Spring force toward home + random jitter
            cx, cy = positions[pid]
            dx = (target_x - cx) * 0.02 + random.gauss(0, 3)
            dy = (target_y - cy) * 0.02 + random.gauss(0, 3)

            velocities[pid] = [dx, dy]
            nx = max(20, min(CANVAS_W - 20, cx + dx))
            ny = max(20, min(CANVAS_H - 20, cy + dy))
            positions[pid] = [nx, ny]

        # Ball: follows a player roughly, with some randomness
        ball_carrier = all_players[(f // 30) % len(all_players)]
        bcx, bcy = positions[ball_carrier["id"]]
        ball_vx = (bcx - ball_x) * 0.1 + random.gauss(0, 5)
        ball_vy = (bcy - ball_y) * 0.1 + random.gauss(0, 5)
        ball_x = max(0, min(CANVAS_W, ball_x + ball_vx))
        ball_y = max(0, min(CANVAS_H, ball_y + ball_vy))

        players_data = []
        for p in all_players:
            pid = p["id"]
            vx, vy = velocities[pid]
            speed = math.sqrt(vx**2 + vy**2) * FPS / 100  # rough km/h approximation
            players_data.append({
                "id": pid,
                "team": p["team"],
                "x": round(positions[pid][0], 1),
                "y": round(positions[pid][1], 1),
                "vx": round(vx, 2),
                "vy": round(vy, 2),
                "speed": round(speed, 1),
                "jersey": p["jersey"],
            })

        frame_data = {
            "i": f,
            "players": players_data,
            "ball": {"x": round(ball_x, 1), "y": round(ball_y, 1)},
        }
        frames.append(frame_data)

    # ── Events ──
    events = [
        {"frame": 50, "type": "shot", "team": "A", "player_id": 10, "outcome": "saved", "x": 90.0, "y": 34.0},
        {"frame": 120, "type": "corner", "team": "A", "label": "Left side corner"},
        {"frame": 180, "type": "foul", "team": "B", "player_id": 18, "x": 60.0, "y": 40.0},
        {"frame": 200, "type": "free_kick", "team": "A", "x": 60.0, "y": 40.0},
        {"frame": 250, "type": "shot", "team": "A", "player_id": 9, "outcome": "goal", "x": 95.0, "y": 30.0},
        {"frame": 250, "type": "goal", "team": "A", "player_id": 9, "label": "Great strike!"},
        {"frame": 320, "type": "card", "team": "B", "player_id": 17, "outcome": "yellow", "label": "Yellow card"},
        {"frame": 380, "type": "shot", "team": "B", "player_id": 21, "outcome": "off_target", "x": 15.0, "y": 38.0},
        {"frame": 420, "type": "corner", "team": "B", "label": "Right side corner"},
        {"frame": 460, "type": "offside", "team": "A", "player_id": 10},
    ]

    # ── Formations ──
    formations = [
        {"frame_start": 0, "frame_end": 249, "team": "A", "name": "4-3-3", "lines": [4, 3, 3]},
        {"frame_start": 250, "frame_end": N - 1, "team": "A", "name": "4-4-2", "lines": [4, 4, 2]},
        {"frame_start": 0, "frame_end": N - 1, "team": "B", "name": "4-2-3-1", "lines": [4, 2, 3, 1]},
    ]

    # ── Shots ──
    shots = [
        {"frame": 50, "team": "A", "player_id": 10, "x": 90.0, "y": 34.0, "outcome": "saved", "on_target": True, "xg": 0.12},
        {"frame": 250, "team": "A", "player_id": 9, "x": 95.0, "y": 30.0, "outcome": "goal", "on_target": True, "xg": 0.35},
        {"frame": 380, "team": "B", "player_id": 21, "x": 15.0, "y": 38.0, "outcome": "off_target", "on_target": False, "xg": 0.08},
        {"frame": 450, "team": "B", "player_id": 20, "x": 20.0, "y": 25.0, "outcome": "saved", "on_target": True, "xg": 0.15},
    ]

    # ── Pass sonar (8 sectors per player) ──
    pass_sonar = {}
    for p in team_a + team_b:
        if p["team"] in ("A", "B"):
            pass_sonar[str(p["id"])] = {
                "sectors": [random.randint(2, 20) for _ in range(8)],
                "x": p["home_x"] / CANVAS_W * 105,
                "y": p["home_y"] / CANVAS_H * 68,
                "team": p["team"],
            }

    # ── Zone 14 (10x10 grids) ──
    zone14 = {
        "A": [[random.randint(0, 10) for _ in range(10)] for _ in range(10)],
        "B": [[random.randint(0, 8) for _ in range(10)] for _ in range(10)],
    }
    # Make zone 14 area (right side for A, left side for B) hotter
    for r in range(3, 7):
        for c in range(7, 10):
            zone14["A"][r][c] += random.randint(5, 15)
        for c in range(0, 3):
            zone14["B"][r][c] += random.randint(5, 15)

    # ── Buildups ──
    buildups = [
        {
            "id": 1, "team": "A", "frame_start": 30, "frame_end": 50, "outcome": "shot",
            "passes": [
                {"from_id": 7, "to_id": 9, "frame": 35, "x": 500, "y": 400, "end_x": 700, "end_y": 300},
                {"from_id": 9, "to_id": 10, "frame": 42, "x": 700, "y": 300, "end_x": 800, "end_y": 500},
            ],
        },
        {
            "id": 2, "team": "A", "frame_start": 220, "frame_end": 250, "outcome": "goal",
            "passes": [
                {"from_id": 6, "to_id": 7, "frame": 225, "x": 450, "y": 300, "end_x": 500, "end_y": 500},
                {"from_id": 7, "to_id": 11, "frame": 232, "x": 500, "y": 500, "end_x": 680, "end_y": 800},
                {"from_id": 11, "to_id": 9, "frame": 240, "x": 680, "y": 800, "end_x": 750, "end_y": 300},
            ],
        },
        {
            "id": 3, "team": "B", "frame_start": 350, "frame_end": 380, "outcome": "shot",
            "passes": [
                {"from_id": 18, "to_id": 20, "frame": 360, "x": 1080, "y": 500, "end_x": 880, "end_y": 200},
                {"from_id": 20, "to_id": 21, "frame": 370, "x": 880, "y": 200, "end_x": 800, "end_y": 500},
            ],
        },
    ]

    # ── Duels (8x5 grids) ──
    duels = {
        "grid_a": [[random.randint(0, 8) for _ in range(8)] for _ in range(5)],
        "grid_b": [[random.randint(0, 7) for _ in range(8)] for _ in range(5)],
    }

    # ── Transitions ──
    transitions = [
        {"team": "A", "frame": 100, "type": "defense_to_attack", "x": 50.0, "y": 34.0, "duration_frames": 40, "outcome": "possession_lost"},
        {"team": "A", "frame": 220, "type": "defense_to_attack", "x": 45.0, "y": 30.0, "duration_frames": 30, "outcome": "goal"},
        {"team": "B", "frame": 300, "type": "defense_to_attack", "x": 55.0, "y": 40.0, "duration_frames": 80, "outcome": "shot"},
        {"team": "A", "frame": 310, "type": "attack_to_defense", "x": 60.0, "y": 35.0, "duration_frames": 20},
        {"team": "B", "frame": 400, "type": "defense_to_attack", "x": 52.0, "y": 28.0, "duration_frames": 50, "outcome": "corner"},
    ]

    # ── Set pieces ──
    set_pieces = {
        "corners": [
            {"frame": 120, "team": "A", "outcome": "cleared"},
            {"frame": 420, "team": "B", "outcome": "shot"},
        ],
        "free_kicks": [
            {"frame": 200, "team": "A", "x": 60.0, "y": 40.0, "outcome": "crossed"},
        ],
    }

    # ── Assemble ──
    data = {
        "meta": {
            "fps": FPS,
            "total_frames": N,
            "canvas": {"width": CANVAS_W, "height": CANVAS_H},
            "pitch": {"length": 105, "width": 68},
        },
        "frames": frames,
        "events": events,
        "formations": formations,
        "shots": shots,
        "pass_sonar": pass_sonar,
        "zone14": zone14,
        "buildups": buildups,
        "duels": duels,
        "transitions": transitions,
        "set_pieces": set_pieces,
    }

    with open(args.out, "w") as f:
        json.dump(data, f)

    size_mb = len(json.dumps(data)) / 1024 / 1024
    print(f"Generated {args.out}")
    print(f"  {N} frames, {FPS} fps, {len(all_players)} players")
    print(f"  {len(events)} events, {len(shots)} shots, {len(formations)} formations")
    print(f"  File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
