"use client";

import { useState, useEffect } from "react";
import jsPDF, { Html2CanvasOptions } from "jspdf";
import html2canvas from "html2canvas";
import ModeSwitch from "@/components/ModeSwitch";
import PHCard from "@/components/PHCard";
import DecisionCard from "@/components/DecisionCard";
import ExplanationCard from "@/components/ExplanationCard";
import GraphCard from "@/components/GraphCard";

interface GraphPoint {
  time: number;
  value: number;
}

interface History {
  temp: GraphPoint[];
  humidity: GraphPoint[];
  moisture: GraphPoint[];
}

interface SensorResponse {
  temp: number;
  humidity: number;
  moisture: number;
  decision: string;
  confidence: number;
  explanation: string;
  ph: number;
}

export default function IrrigationPage() {
  const [data, setData] = useState<SensorResponse | null>(null);
  const [mode, setMode] = useState<string>("normal");

  const [count, setCount] = useState<number>(0);
  const [speed, setSpeed] = useState<number>(1);
  const [alertSent, setAlertSent] = useState(false);

  const [history, setHistory] = useState<History>({
    temp: [],
    humidity: [],
    moisture: [],
  });

  const fetchData = async () => {
    try {
      const res = await fetch(`http://localhost:8000/simulate?mode=${mode}`);
      const json: SensorResponse = await res.json();

      setData(json);

      setCount((c) => c + 1);

      setHistory((prev) => ({
        temp: [...prev.temp.slice(-49), { time: Date.now(), value: json.temp }],
        humidity: [
          ...prev.humidity.slice(-49),
          { time: Date.now(), value: json.humidity },
        ],
        moisture: [
          ...prev.moisture.slice(-49),
          { time: Date.now(), value: json.moisture },
        ],
      }));
      if (json.decision === "STOP" && !alertSent) {
        sendStopAlert(json);
        setAlertSent(true);
      }

      if (json.decision === "WATER" && alertSent) {
        setAlertSent(false);
      }
    } catch (e) {
      console.error("API ERROR:", e);
    }
  };

  async function sendStopAlert(json: SensorResponse) {
    await fetch("/api/send-email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        decision: json.decision,
        temp: json.temp,
        humidity: json.humidity,
        moisture: json.moisture,
        confidence: json.confidence,
      }),
    });
  }

  useEffect(() => {
    fetchData();

    const interval = 2000 / speed;

    const timer = setInterval(fetchData, interval);
    return () => clearInterval(timer);
  }, [mode, speed]);

  const downloadPDF = async () => {
    const pdf = new jsPDF("p", "mm", "a4");
    let y = 10;

    const addSection = async (id: string, title: string) => {
      const element = document.getElementById(id);
      if (!element) return;

      document.body.classList.add("export-mode");

      const canvas = await html2canvas(
        element as HTMLElement,
        {
          scale: 2,
        } as any
      );

      document.body.classList.remove("export-mode");

      const imgData = canvas.toDataURL("image/png");
      const imgWidth = 180;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      if (y + imgHeight > 290) {
        pdf.addPage();
        y = 10;
      }

      pdf.setFontSize(14);
      pdf.text(title, 10, y);
      y += 6;

      pdf.addImage(imgData, "PNG", 10, y, imgWidth, imgHeight);
      y += imgHeight + 10;
    };

    pdf.setFontSize(18);
    pdf.text("Irrigenius Report", 10, y);
    y += 10;

    pdf.setFontSize(12);
    pdf.text(`Mode: ${mode}`, 10, y);
    y += 6;

    pdf.text(`Timestamp: ${new Date().toLocaleString()}`, 10, y);
    y += 6;

    pdf.text(`Total Data Points Received: ${count}`, 10, y);
    y += 10;

    await addSection("decision-card", "Decision");
    await addSection("explanation-card", "Explanation");
    await addSection("temperature-graph", "Temperature Trend");
    await addSection("humidity-graph", "Humidity Trend");
    await addSection("moisture-graph", "Moisture Trend");

    pdf.save(`Irrigenius_Report_${new Date().toLocaleString()}.pdf`);
  };

  if (!data) return <div className="p-10">Loading...</div>;

  return (
    <main className="p-10 bg-gray-100 min-h-screen w-full flex flex-col items-center justify-center">
      <div className="flex items-center justify-between mb-6 w-full">
        <ModeSwitch mode={mode} setMode={setMode} />

        <div className="flex gap-2">
          {[1, 2, 4, 6].map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-4 py-2 rounded-md border 
                ${speed === s ? "bg-blue-600 text-white" : "bg-white"}`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <DecisionCard decision={data.decision} confidence={data.confidence} />
        <div className="col-span-2 text-center text-gray-600 text-sm mt-2">
          Data points received: <span className="font-semibold">{count}</span>
        </div>
        <ExplanationCard
          explanation={`${data.explanation} | Recommended pH: ${data.ph}`}
        />
        <PHCard ph={data.ph} />

        <div id="temperature-graph">
          <GraphCard label="Temperature Trend" data={history.temp} />
        </div>

        <div id="humidity-graph">
          <GraphCard label="Humidity Trend" data={history.humidity} />
        </div>

        <div id="moisture-graph">
          <GraphCard label="Soil Moisture Trend" data={history.moisture} />
        </div>
      </div>
      <button
        onClick={downloadPDF}
        className="mt-10 px-6 py-3 bg-gray-900 text-white rounded-lg hover:bg-gray-700 transition-all"
      >
        Download Report as PDF
      </button>
    </main>
  );
}
