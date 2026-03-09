import { Delaunay } from "d3-delaunay";
import type { TrackingPlayer } from "../types/tracking";

export interface VoronoiResult {
  cellPaths: { path: Path2D; team: string }[];
}

export function computeVoronoi(
  players: TrackingPlayer[],
  _pitchW: number,
  _pitchH: number,
  toX: (v: number) => number,
  toY: (v: number) => number,
  srcW: number,
  srcH: number,
): VoronoiResult {
  if (players.length < 3) return { cellPaths: [] };

  const fieldPlayers = players.filter((p) => p.team === "A" || p.team === "B");
  if (fieldPlayers.length < 3) return { cellPaths: [] };

  const points = fieldPlayers.map((p) => [toX(p.x), toY(p.y)] as [number, number]);
  const delaunay = Delaunay.from(points);
  const voronoi = delaunay.voronoi([toX(0), toY(0), toX(srcW), toY(srcH)]);

  const cellPaths: VoronoiResult["cellPaths"] = [];
  for (let i = 0; i < fieldPlayers.length; i++) {
    const cell = voronoi.cellPolygon(i);
    if (!cell) continue;
    const path = new Path2D();
    path.moveTo(cell[0][0], cell[0][1]);
    for (let j = 1; j < cell.length; j++) {
      path.lineTo(cell[j][0], cell[j][1]);
    }
    path.closePath();
    cellPaths.push({ path, team: fieldPlayers[i].team });
  }

  return { cellPaths };
}
