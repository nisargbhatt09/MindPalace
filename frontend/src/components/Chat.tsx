import { useRef, useState } from "react";
import { sendChat } from "../api";
import type { ChatMessage, SourceMemory } from "../types";

interface Turn {
  role: "user" | "assistant";
  content: string;
  sources?: SourceMemory[];
}

const SUGGESTIONS = [
  "Where was I in the morning?",
  "Did I see any animals today?",
  "What did I have for lunch?",
  "What did I do this afternoon?",
];

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy) return;

    const history: ChatMessage[] = turns.map((t) => ({
      role: t.role,
      content: t.content,
    }));

    setTurns((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setBusy(true);
    setError(null);

    try {
      const res = await sendChat(text, history);
      setTurns((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, sources: res.sources },
      ]);
    } catch {
      setError("Sorry, I couldn't reach the assistant. Is the backend running?");
    } finally {
      setBusy(false);
      requestAnimationFrame(() =>
        endRef.current?.scrollIntoView({ behavior: "smooth" }),
      );
    }
  }

  return (
    <div className="chat">
      <div className="chat-log">
        {turns.length === 0 && (
          <div className="suggestions">
            <p className="muted">Try asking:</p>
            {SUGGESTIONS.map((s) => (
              <button key={s} className="chip" onClick={() => ask(s)}>
                {s}
              </button>
            ))}
          </div>
        )}

        {turns.map((turn, i) => (
          <div key={i} className={`bubble ${turn.role}`}>
            <p>{turn.content}</p>
            {turn.sources && turn.sources.length > 0 && (
              <div className="sources">
                {turn.sources.map((s) => (
                  <div key={s.id} className="source-card">
                    {s.image_url ? (
                      <img src={s.image_url} alt={s.caption} />
                    ) : (
                      <div className="source-noimage" aria-hidden />
                    )}
                    <div className="source-meta">
                      <span className="source-when">
                        {s.when ?? "Some time"}
                        {s.where ? ` · ${s.where}` : ""}
                      </span>
                      <span className="source-caption">{s.caption}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {busy && <div className="bubble assistant muted">Thinking…</div>}
        {error && <div className="error">{error}</div>}
        <div ref={endRef} />
      </div>

      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <input
          type="text"
          value={input}
          placeholder="Ask about your day…"
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
          aria-label="Ask about your day"
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}
