import { Link } from "react-router-dom";

export default function About() {
  return (
    <article className="page">
      <header className="page-header">
        <p className="eyebrow">About</p>
        <h1 className="page-title">Why we built MindPalace</h1>
        <p className="page-stand">
          Forgetting isn't only losing facts. It's losing the thread of your own
          day — and with it, a little of your footing. MindPalace is our attempt to
          hand that thread back.
        </p>
      </header>

      <div className="prose">
        <p>
          It usually starts small. A parent calls to ask whether they've eaten lunch.
          A grandmother worries she missed a visit that happened an hour ago. The
          question underneath is rarely about the meal or the visit. It's
          <em> “Can I still trust my own day?”</em>
        </p>

        <p>
          The phone in their pocket already holds the answer. There's a photo of the
          lunch, taken at half past one. A picture from the park, with the time and
          the place quietly tucked inside it. The day is recorded — it's just locked
          away in a grid of thumbnails that's hard to search and harder to ask.
        </p>

        <p>
          MindPalace unlocks it. It reads each photo, notes when and where it was
          taken, and writes a gentle description. Then it lets you ask in the most
          natural way there is — out loud, in your own words:
          <em> “Where was I this morning?”</em> And it answers like a patient friend
          who happened to be there.
        </p>

        <h2>The one promise that shapes everything</h2>
        <p>
          For someone living with memory loss, a confidently wrong answer is worse
          than no answer at all. An invented detail can unsettle a whole afternoon.
          So we drew one hard line, and every part of MindPalace bends to it:
        </p>
        <blockquote>
          It will never invent a memory. If there's no photo, it says so — plainly
          and kindly — rather than guess.
        </blockquote>
        <p>
          That restraint is the soul of the product. We'd rather MindPalace say
          “I don't have a photo from then” a hundred times than comfort someone with
          a single thing that never happened.
        </p>

        <h2>Who it's for</h2>
        <p>
          For the person who wants to hold onto their own days a little longer. And
          for the family member or carer who can't always be in the room — who wants
          to know that when the worry comes at 4pm, there's a calm, honest voice to
          answer it.
        </p>

        <h2>Where we're going</h2>
        <p>
          This is an early, honest beginning — a working companion, not a finished
          one. Next we want everything to run privately on your own device, so your
          memories never have to travel anywhere at all. We're building slowly and
          carefully, because trust, here, is the whole thing.
        </p>

        <p className="prose-cta">
          <Link to="/app" className="btn btn-primary">
            Try MindPalace
          </Link>
          <Link to="/how-to-use" className="btn btn-ghost">
            How to use it
          </Link>
        </p>
      </div>
    </article>
  );
}
