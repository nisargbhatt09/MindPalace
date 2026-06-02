import { Link } from "react-router-dom";
import Logo from "./Logo";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <span className="brand-mark">
            <Logo />
          </span>
          <p className="footer-tag">
            A gentle place to keep the day,
            <br />
            for the moments that get hard to hold.
          </p>
        </div>

        <nav className="footer-links" aria-label="Footer">
          <Link to="/how-to-use">How to use</Link>
          <Link to="/about">About</Link>
          <Link to="/privacy">Privacy</Link>
          <Link to="/app">Open MindPalace</Link>
        </nav>
      </div>

      <div className="footer-note">
        <span>Made with care.</span>
        <span>MindPalace never invents a memory — it only tells you what really happened.</span>
      </div>
    </footer>
  );
}
