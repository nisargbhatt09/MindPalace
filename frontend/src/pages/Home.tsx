import { Link } from "react-router-dom";
import Polaroid, { PolaroidData } from "../components/Polaroid";

const MEMORIES: PolaroidData[] = [
  {
    tint: "linear-gradient(160deg, #f3e7cf, #e7c79a)",
    caption: "Coffee by the window",
    when: "8:15 in the morning",
    where: "Lakeside Cafe",
    tilt: -5,
  },
  {
    tint: "linear-gradient(160deg, #d8e7d2, #a9c89a)",
    caption: "A little brown dog",
    when: "half past ten",
    where: "Riverside Park",
    tilt: 3,
  },
  {
    tint: "linear-gradient(160deg, #e9d7d0, #d3a78f)",
    caption: "Pasta at the kitchen table",
    when: "lunchtime",
    where: "Home",
    tilt: -2,
  },
];

export default function Home() {
  return (
    <>
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-copy">
            <p className="eyebrow">A memory companion</p>
            <h1 className="hero-title">
              Some days slip away.
              <br />
              Your photos don't have to.
            </h1>
            <p className="hero-sub">
              MindPalace quietly remembers where you were and what you did. So when
              the morning feels far away, you can simply ask — and hear it back,
              kindly, in plain words.
            </p>
            <div className="hero-cta">
              <Link to="/app" className="btn btn-primary btn-lg">
                Open MindPalace
              </Link>
              <Link to="/how-to-use" className="btn btn-ghost btn-lg">
                See how it works
              </Link>
            </div>
          </div>

          <div className="hero-art" aria-hidden="true">
            <div className="polaroids">
              {MEMORIES.map((m) => (
                <Polaroid key={m.caption} data={m} />
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="band">
        <div className="container narrow center">
          <p className="kicker">For the moments that get hard to hold</p>
          <p className="lead">
            Living with memory loss — your own, or someone you love — means small
            certainties go missing. <em>Did I eat today? Where did I go this
            morning? Who came by?</em> MindPalace keeps those answers close, and
            hands them back gently, whenever they're needed.
          </p>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <h2 className="section-title">Three quiet steps</h2>
          <ol className="steps">
            <li className="step">
              <span className="step-num">1</span>
              <h3 className="step-title">Add your photos</h3>
              <p>
                A breakfast, a walk, a visitor. Each picture quietly keeps the time
                and place it was taken.
              </p>
            </li>
            <li className="step">
              <span className="step-num">2</span>
              <h3 className="step-title">They become memories</h3>
              <p>
                MindPalace writes a warm little note for every photo — what's in it,
                where you were, the time of day.
              </p>
            </li>
            <li className="step">
              <span className="step-num">3</span>
              <h3 className="step-title">Just ask</h3>
              <p>
                “Where was I this morning?” It answers like a kind friend — and only
                from what really happened.
              </p>
            </li>
          </ol>
        </div>
      </section>

      <section className="section demo-section">
        <div className="container demo-grid">
          <div className="demo-intro">
            <p className="kicker">A real conversation</p>
            <h2 className="section-title">Like asking someone who was there.</h2>
            <p className="muted-lead">
              No menus to learn. No dates to type. You ask the way you'd ask a
              person — and the answer comes back unhurried and clear.
            </p>
          </div>

          <div className="demo-window" aria-label="Example conversation">
            <div className="demo-line user">Where was I in the morning?</div>
            <div className="demo-line assistant">
              This morning, around 8 o'clock, you were at the Lakeside Cafe — coffee
              and a croissant by the window. Later, about half past ten, you walked a
              little brown dog in Riverside Park. A peaceful morning.
            </div>
            <div className="demo-line user">Was I at the beach today?</div>
            <div className="demo-line assistant">
              I don't have any photos from the beach today. Would you like me to look
              at another time?
            </div>
            <p className="demo-foot">
              Notice the last answer. When there's no photo, MindPalace says so —
              <strong> it never makes a memory up.</strong>
            </p>
          </div>
        </div>
      </section>

      <section className="reassure">
        <div className="container narrow center">
          <h2 className="reassure-title">Your memories belong to you.</h2>
          <p className="lead">
            Your photos live on your own device, in a library only you open. We built
            MindPalace to be a trustworthy keeper — never a salesperson.
          </p>
          <Link to="/privacy" className="btn btn-ghost">
            Read our promise on privacy
          </Link>
        </div>
      </section>
    </>
  );
}
