"use client";

interface Props {
  decision: string;
  confidence: number;
}

export default function DecisionCard({ decision, confidence }: Props) {
  const active = decision === "WATER";

  return (
    <div
      id="decision-card"
      className="border border-gray-300 rounded-xl p-7 bg-white flex flex-col justify-center items-center col-span-2"
    >
      <div className="text-gray-600 text-lg">Decision</div>

      <div
        className={`mt-4 text-5xl font-bold ${
          active ? "text-green-600" : "text-blue-600"
        }`}
      >
        {decision}
      </div>

      {/* <div className="text-gray-500 mt-2 text-sm">
        Confidence: {confidence}%
      </div> */}
    </div>
  );
}
