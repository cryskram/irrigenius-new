"use client";

interface Props {
  label: string;
  value: number | string;
  unit?: string;
}

export default function SensorCard({ label, value, unit }: Props) {
  return (
    <div className="border border-gray-300 rounded-xl p-5 flex flex-col bg-white w-full">
      <div className="text-gray-500 text-sm">{label}</div>
      <div className="text-3xl font-semibold mt-2">
        {value} <span className="text-gray-400 text-xl">{unit}</span>
      </div>
    </div>
  );
}
