"use client";

import { useState } from "react";
import jsPDF from "jspdf";
import DiseaseUploadCard from "@/components/DiseaseUploadCard";
import DiseaseResultCard from "@/components/DiseaseResultCard";
import HeatmapCard from "@/components/HeatmapCard";

export default function DiseasePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const upload = async () => {
    if (!file) return;
    setLoading(true);

    const form = new FormData();
    form.append("file", file);

    const res = await fetch("http://localhost:8000/detect-disease", {
      method: "POST",
      body: form,
    });

    const json = await res.json();
    setResult(json);
    setLoading(false);
  };

  const downloadPDF = () => {
    if (!result) return;

    const pdf = new jsPDF("p", "mm", "a4");

    pdf.setFontSize(18);
    pdf.text("Plant Disease Detection Report", 10, 10);

    pdf.setFontSize(12);
    pdf.text(`Prediction: ${result.label}`, 10, 25);
    pdf.text(`Confidence: ${(result.confidence * 100).toFixed(2)}%`, 10, 32);
    pdf.text(`Timestamp: ${new Date().toLocaleString()}`, 10, 39);

    pdf.addImage(
      "data:image/jpeg;base64," + result.original,
      "JPEG",
      10,
      50,
      85,
      85
    );

    if (result.heatmap) {
      pdf.addImage(
        "data:image/jpeg;base64," + result.heatmap,
        "JPEG",
        110,
        50,
        85,
        85
      );
    }

    pdf.save(`Plant_Disease_Report_${Date.now()}.pdf`);
  };

  return (
    <main className="p-10 bg-gray-100 min-h-screen flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-6">Plant Disease Detection</h1>

      <DiseaseUploadCard
        file={file}
        setFile={setFile}
        upload={upload}
        loading={loading}
      />

      {result && (
        <>
          <DiseaseResultCard
            label={result.label}
            confidence={result.confidence}
          />

          <div className="grid grid-cols-2 gap-6 mt-6">
            <HeatmapCard
              title="Original Image"
              img={"data:image/jpeg;base64," + result.original}
            />
            <HeatmapCard
              title="Grad-CAM Heatmap"
              img={"data:image/jpeg;base64," + result.heatmap}
            />
          </div>

          <button
            onClick={downloadPDF}
            className="mt-10 px-6 py-3 bg-green-700 text-white rounded-lg hover:bg-green-600 transition-all"
          >
            Download as PDF
          </button>
        </>
      )}
    </main>
  );
}
