const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchCheckpoints() {
  const res = await fetch(`${API_URL}/api/checkpoints`);
  if (!res.ok) throw new Error(`failed to load checkpoints (${res.status})`);
  const body = await res.json();
  return body.checkpoints;
}

export async function scoreTransaction({ checkpoint, target, edges }) {
  const res = await fetch(`${API_URL}/api/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint, target, edges }),
  });
  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.detail || `scoring failed (${res.status})`);
  }
  return body;
}
