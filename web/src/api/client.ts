const BASE = "/api";

export interface Project {
  id: string;
  name: string;
  status: "created" | "uploaded" | "processing" | "completed" | "error";
  video_path: string | null;
  tracking_path: string | null;
  error?: string;
}

export async function createProject(name: string): Promise<Project> {
  const res = await fetch(`${BASE}/projects?name=${encodeURIComponent(name)}`, {
    method: "POST",
  });
  return res.json();
}

export async function listProjects(): Promise<Project[]> {
  const res = await fetch(`${BASE}/projects`);
  return res.json();
}

export async function getProject(id: string): Promise<Project> {
  const res = await fetch(`${BASE}/projects/${id}`);
  return res.json();
}

export async function uploadVideo(projectId: string, file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  await fetch(`${BASE}/projects/${projectId}/upload`, {
    method: "POST",
    body: form,
  });
}

export async function startProcessing(projectId: string): Promise<void> {
  await fetch(`${BASE}/projects/${projectId}/process`, { method: "POST" });
}

export async function getTrackingData(projectId: string): Promise<unknown> {
  const res = await fetch(`${BASE}/projects/${projectId}/tracking`);
  return res.json();
}

export function getVideoUrl(projectId: string): string {
  return `${BASE}/projects/${projectId}/video`;
}

export function connectProgress(
  projectId: string,
  onMessage: (data: { frame: number; total: number; done: boolean; error: string | null }) => void
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/projects/${projectId}/progress`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return ws;
}
