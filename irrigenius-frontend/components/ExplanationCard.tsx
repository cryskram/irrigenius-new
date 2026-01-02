"use client";

export default function ExplanationCard({ explanation }: any) {
  return (
    <div
      id="explanation-card"
      className="border border-gray-300 rounded-xl p-6 bg-white w-full"
    >
      <div className="text-gray-500 text-sm">AI Explanation</div>
      <p className="mt-3 text-gray-700 text-[15px] leading-relaxed">
        {explanation}
      </p>
    </div>
  );
}
