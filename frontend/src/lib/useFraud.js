import { apiErrorMessage } from "./apiError.js";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchFraudConfig() {
  const res = await fetch(`${API_URL}/api/fraud/config`);
  if (!res.ok) throw new Error(`failed to load detector config (${res.status})`);
  return res.json();
}

async function post(path, body, failure) {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(apiErrorMessage(payload, `${failure} (${res.status})`));
  }
  return payload;
}

export function runFraudScan(body) {
  return post("/api/fraud/scan", body, "scan failed");
}

export function uploadFraudScan(body) {
  return post("/api/fraud/upload", body, "file check failed");
}

export async function fetchModelInfo() {
  const res = await fetch(`${API_URL}/api/fraud/model`);
  const payload = await res.json();
  if (!res.ok) throw new Error(apiErrorMessage(payload, `failed to load model details (${res.status})`));
  return payload;
}
