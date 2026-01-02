"use client";

export default function DiseaseResultCard({ label, confidence }: any) {
  return (
    <div className="bg-white shadow-md rounded-xl p-6 w-full max-w-xl mt-6">
      <h2 className="text-xl font-semibold">Detection Result</h2>

      <p className="mt-3 text-lg">
        <span className="font-bold">Disease:</span> {label}
      </p>

      <p className="mt-1 text-lg">
        <span className="font-bold">Confidence:</span>{" "}
        {(confidence * 100).toFixed(2)}%
      </p>
    </div>
  );
}
