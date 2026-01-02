"use client";

interface Props {
  file: File | null;
  setFile: (f: File | null) => void;
  upload: () => void;
  loading: boolean;
}

export default function DiseaseUploadCard({
  file,
  setFile,
  upload,
  loading,
}: Props) {
  return (
    <div className="bg-white shadow-lg rounded-xl p-6 w-full max-w-xl">
      <h2 className="text-xl font-semibold mb-4">Upload Leaf Image</h2>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="mb-4"
      />

      {file && (
        <p className="text-sm text-gray-600 mb-2">Selected: {file.name}</p>
      )}

      <button
        onClick={upload}
        disabled={loading || !file}
        className={`w-full px-4 py-2 rounded-md text-white ${
          loading ? "bg-gray-400" : "bg-blue-600 hover:bg-blue-700"
        }`}
      >
        {loading ? "Analyzing..." : "Detect Disease"}
      </button>
    </div>
  );
}
