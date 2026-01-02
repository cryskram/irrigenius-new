"use client";

export default function HeatmapCard({ title, img }: any) {
  return (
    <div className="bg-white shadow-md p-6 rounded-xl flex flex-col items-center">
      <h3 className="text-lg font-semibold mb-3">{title}</h3>
      <img src={img} className="rounded-md shadow" alt={title} />
    </div>
  );
}
