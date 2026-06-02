import { Link } from "react-router-dom";

export default function Privacy() {
  return (
    <article className="page">
      <header className="page-header">
        <p className="eyebrow">Privacy</p>
        <h1 className="page-title">Your memories belong to you.</h1>
        <p className="page-stand">
          A memory keeper only works if you can trust it completely. So here, in plain
          language, is exactly how MindPalace treats your photos and your day — and
          what it will never do.
        </p>
      </header>

      <div className="prose">
        <h2>What stays with you</h2>
        <p>
          Your photos and your memory library live <strong>on your own device</strong>
          — in a single file you control. MindPalace doesn't keep an account for you,
          and there's no copy of your library sitting on our servers. If you delete the
          file, it's gone. It was only ever yours.
        </p>

        <h2>What leaves the device, and why — honestly</h2>
        <p>
          To turn a photo into a memory, and to understand a question like “where was
          I this morning?”, MindPalace currently sends the image or your question to a
          trusted AI model provider to be read. We send only what's needed for that
          one answer, and we don't attach your name or an account to it.
        </p>
        <p>
          We want to be straight with you: in this early version, that understanding
          happens in the cloud. <strong>Running it entirely on your own device, so
          nothing travels at all, is the next thing we're building</strong> — and the
          reason we chose a design that keeps your library local from day one.
        </p>

        <h2>What we never do</h2>
        <ul className="notes-list">
          <li>We never sell your photos, your memories, or anything about you.</li>
          <li>We never show advertising, and we don't build a profile of you.</li>
          <li>
            We never use your private memories to train our own products without
            your clear, informed permission.
          </li>
          <li>
            We never invent a memory. (That's a promise about honesty as much as
            privacy — and it matters just as much.)
          </li>
        </ul>

        <h2>You're always in control</h2>
        <p>
          Add a photo, and it becomes a memory. Remove it, and the memory goes with
          it. There's no hidden archive and no long-term retention you can't see. The
          library is yours to keep, move, or erase.
        </p>

        <h2>A note for families</h2>
        <p>
          Because everything centres on a file on a device, you can set MindPalace up
          on a loved one's behalf and keep it as private as the device itself. Treat
          that device with the same care you'd give a diary — because that's rather
          what this is.
        </p>

        <p className="prose-cta">
          <Link to="/about" className="btn btn-ghost">
            Read why we built it
          </Link>
          <Link to="/app" className="btn btn-primary">
            Open MindPalace
          </Link>
        </p>

        <p className="fineprint">
          This is an honest plain-language summary for an early product, not a legal
          contract. As MindPalace grows, we'll publish a full policy — and we'll keep
          it just as readable.
        </p>
      </div>
    </article>
  );
}
