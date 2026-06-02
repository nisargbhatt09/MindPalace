import { Link, NavLink } from "react-router-dom";
import Logo from "./Logo";

const LINKS = [
  { to: "/how-to-use", label: "How to use" },
  { to: "/about", label: "About" },
  { to: "/privacy", label: "Privacy" },
];

export default function Nav() {
  return (
    <header className="nav">
      <div className="nav-inner">
        <Link to="/" className="brand" aria-label="MindPalace home">
          <span className="brand-mark">
            <Logo />
          </span>
          <span className="brand-word">MindPalace</span>
        </Link>

        <nav className="nav-links" aria-label="Primary">
          {LINKS.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
            >
              {l.label}
            </NavLink>
          ))}
          <Link to="/app" className="btn btn-primary nav-cta">
            Open MindPalace
          </Link>
        </nav>
      </div>
    </header>
  );
}
