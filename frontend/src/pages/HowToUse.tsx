import { Link } from "react-router-dom";

const QUESTIONS = [
  "Where was I this morning?",
  "Did I eat lunch today?",
  "Did I see anyone this afternoon?",
  "Was I at the doctor's this week?",
  "Did I see any animals today?",
];

export default function HowToUse() {
  return (
    <article className="page">
      <header className="page-header">
        <p className="eyebrow">How to use</p>
        <h1 className="page-title">It's as simple as asking.</h1>
        <p className="page-stand">
          There's nothing to set up and nothing to memorise. Add a few photos, then
          talk to MindPalace the way you'd talk to a friend.
        </p>
      </header>

      <ol className="howto-steps">
        <li>
          <div className="howto-num">1</div>
          <div className="howto-body">
            <h2>Add the day's photos</h2>
            <p>
              On the <Link to="/app">MindPalace</Link> page, choose
              <strong> Add photos</strong> and pick pictures from the day — a meal, a
              walk, a face. You can add a few at once. Each one keeps the time and
              place it was taken, all on its own.
            </p>
          </div>
        </li>
        <li>
          <div className="howto-num">2</div>
          <div className="howto-body">
            <h2>Give it a moment</h2>
            <p>
              MindPalace looks at each photo and writes a gentle note about it — what
              it sees, where you were, the time of day. The photo then appears in
              <strong> Your photos</strong>, ready to be remembered.
            </p>
          </div>
        </li>
        <li>
          <div className="howto-num">3</div>
          <div className="howto-body">
            <h2>Ask anything about your day</h2>
            <p>
              Type your question in plain words and press <strong>Ask</strong>.
              MindPalace looks only at your real photos and answers warmly. With every
              answer, it shows you the photos it looked at, so you can see for
              yourself.
            </p>
            <ul className="question-chips">
              {QUESTIONS.map((q) => (
                <li key={q}>“{q}”</li>
              ))}
            </ul>
          </div>
        </li>
      </ol>

      <section className="callout">
        <h2>A few gentle notes</h2>
        <ul className="notes-list">
          <li>
            <strong>If there's no photo, it will say so.</strong> MindPalace never
            guesses. “I don't have a photo from then” is an honest, good answer.
          </li>
          <li>
            <strong>Times and places come from the photo itself.</strong> Pictures
            taken on a phone usually carry this quietly. If a photo doesn't have it,
            MindPalace simply won't claim to know.
          </li>
          <li>
            <strong>You can keep chatting.</strong> Ask a follow-up — “and the
            afternoon?” — and it remembers what you were talking about.
          </li>
        </ul>
      </section>

      <section className="callout soft">
        <h2>For family and carers</h2>
        <p>
          MindPalace works beautifully as a shared, calming habit. Help add photos at
          the end of the day, and your loved one has an honest, patient answer ready
          whenever a worry surfaces — even when you can't be there. Every answer
          shows its source photos, so nothing is ever taken on faith.
        </p>
        <Link to="/app" className="btn btn-primary">
          Open MindPalace
        </Link>
      </section>
    </article>
  );
}
