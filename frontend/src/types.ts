export interface Memory {
  id: string;
  filename: string;
  image_url?: string | null;
  created_at: string;
  captured_at?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  place_name?: string | null;
  caption: string;
  tags: string[];
  scene?: string | null;
  activity?: string | null;
}

export interface SourceMemory {
  id: string;
  when?: string | null;
  where?: string | null;
  caption: string;
  image_url?: string | null;
}

export interface ChatResponse {
  answer: string;
  sources: SourceMemory[];
  found_memories: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
