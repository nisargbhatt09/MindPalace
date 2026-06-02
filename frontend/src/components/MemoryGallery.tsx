import { useRef, useState } from "react";
import { ingestImage } from "../api";
import type { Memory } from "../types";

interface Props {
  memories: Memory[];
  onChange: () => void;
}

export default function MemoryGallery({ memories, onChange }: Props) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      for (const file of Array.from(files)) {
        await ingestImage(file);
      }
      onChange();
    } catch {
      setError("Could not add that photo. Is the backend running with an API key?");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="gallery">
      <div className="gallery-actions">
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
        <button onClick={() => fileRef.current?.click()} disabled={uploading}>
          {uploading ? "Adding…" : "Add photos"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {memories.length === 0 ? (
        <p className="muted">
          No photos yet. Add some, or run <code>python seed.py</code> in the
          backend to load a sample day.
        </p>
      ) : (
        <ul className="grid">
          {memories.map((m) => (
            <li key={m.id} className="grid-item">
              {m.image_url ? (
                <img src={m.image_url} alt={m.caption} />
              ) : (
                <div className="grid-noimage" aria-hidden />
              )}
              <div className="grid-meta">
                <span className="grid-when">
                  {m.captured_at
                    ? new Date(m.captured_at).toLocaleString()
                    : "Time unknown"}
                  {m.place_name ? ` · ${m.place_name}` : ""}
                </span>
                <span className="grid-caption">{m.caption}</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
