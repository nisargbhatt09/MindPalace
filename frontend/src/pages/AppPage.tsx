import { useCallback, useEffect, useState } from "react";
import { listMemories } from "../api";
import type { Memory } from "../types";
import Chat from "../components/Chat";
import MemoryGallery from "../components/MemoryGallery";

export default function AppPage() {
  const [memories, setMemories] = useState<Memory[]>([]);

  const refresh = useCallback(async () => {
    try {
      setMemories(await listMemories());
    } catch {
      /* backend may not be running yet; the gallery shows a gentle hint */
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="app-page">
      <header className="app-intro">
        <h1>Your day, gently remembered</h1>
        <p>
          Ask about your day on the left. Add and revisit your photos on the right.
        </p>
      </header>

      <div className="app-grid">
        <section className="panel">
          <h2>Ask about your day</h2>
          <Chat />
        </section>

        <section className="panel">
          <h2>Your photos ({memories.length})</h2>
          <MemoryGallery memories={memories} onChange={refresh} />
        </section>
      </div>
    </div>
  );
}
