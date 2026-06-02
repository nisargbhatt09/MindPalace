/** A "polaroid" memory card — the visual heart of the product. A photo (or a
 *  soft developing-tint when there's none), a handwritten caption, and a quiet
 *  time · place line beneath. */
export interface PolaroidData {
  tint: string;
  caption: string;
  when: string;
  where: string;
  tilt?: number;
  imageUrl?: string | null;
}

export default function Polaroid({ data }: { data: PolaroidData }) {
  return (
    <figure
      className="polaroid"
      style={{ "--tilt": `${data.tilt ?? 0}deg`, "--tint": data.tint } as React.CSSProperties}
    >
      <div className="polaroid-photo">
        {data.imageUrl ? <img src={data.imageUrl} alt={data.caption} /> : null}
      </div>
      <figcaption className="polaroid-cap">{data.caption}</figcaption>
      <p className="polaroid-meta">
        {data.when} · {data.where}
      </p>
    </figure>
  );
}
