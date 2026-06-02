import type { ChatMessage, ChatResponse, Memory } from "./types";

const BASE = "/api";

export async function listMemories(): Promise<Memory[]> {
  const res = await fetch(`${BASE}/memories`);
  if (!res.ok) throw new Error("Failed to load memories");
  return res.json();
}

export async function ingestImage(file: File): Promise<Memory> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Failed to add photo");
  return res.json();
}

export async function sendChat(
  message: string,
  history: ChatMessage[],
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) throw new Error("Failed to get an answer");
  return res.json();
}
