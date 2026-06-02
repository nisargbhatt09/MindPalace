/** The MindPalace mark: a small arched doorway — a palace you can step into to
 *  retrieve a memory — with a warm light inside. */
export default function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M6 27V14.5a10 10 0 0 1 20 0V27"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <path d="M4 27h24" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="16" cy="15" r="3.2" fill="var(--gold)" />
    </svg>
  );
}
