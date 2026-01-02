"use client";

import React, { useEffect, useRef, useState } from "react";

type Plant = { x: number; y: number; severity: number };

export default function DroneSimPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [plants, setPlants] = useState<Plant[]>([]);
  const [order, setOrder] = useState<number[]>([]);
  const [path, setPath] = useState<Plant[]>([]);
  const [dronePos, setDronePos] = useState({ x: 0, y: 0 });
  const [running, setRunning] = useState(false);

  const [liveDronePos, setLiveDronePos] = useState({ x: 0, y: 0 });
  const [speedMs, setSpeedMs] = useState(600);
  const [airsimMode, setAirSimMode] = useState(false);

  const FIELD_SIZE = 50;
  const CANVAS_SIZE = 600;
  const SCALE = CANVAS_SIZE / FIELD_SIZE;

  async function savePlantsToBackend(plants: Plant[]) {
    try {
      await fetch("http://localhost:8000/save-plants", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plants }),
      });
      console.log("Plants synced with backend ✔");
    } catch (err) {
      console.error("Failed to save plants to backend:", err);
    }
  }

  useEffect(() => {
    let interval: number | undefined;

    if (airsimMode) {
      interval = window.setInterval(async () => {
        try {
          const res = await fetch("http://localhost:8000/drone/position");
          const data = await res.json();

          if (data.status === "ok" && data.pos) {
            setLiveDronePos({
              x: Number(data.pos.x) || 0,
              y: Number(data.pos.y) || 0,
            });
          }
        } catch (err) {
          console.error("AirSim position error:", err);
        }
      }, 150);
    }

    return () => {
      if (interval !== undefined) {
        clearInterval(interval);
      }
    };
  }, [airsimMode]);

  useEffect(() => {
    drawScene();
  }, [plants, dronePos, path, liveDronePos, airsimMode]);

  function drawScene() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    ctx.fillStyle = "#0a0f1a";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    const GridStep = 30;
    ctx.strokeStyle = "#1c2635";
    ctx.lineWidth = 1;

    for (let x = 0; x <= CANVAS_SIZE; x += GridStep) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, CANVAS_SIZE);
      ctx.stroke();
    }
    for (let y = 0; y <= CANVAS_SIZE; y += GridStep) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(CANVAS_SIZE, y);
      ctx.stroke();
    }

    if (path.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(0,255,200,0.7)";
      ctx.lineWidth = 2;

      ctx.moveTo(0, CANVAS_SIZE - 0);
      path.forEach((p) => {
        ctx.lineTo(p.x * SCALE, CANVAS_SIZE - p.y * SCALE);
      });
      ctx.stroke();

      ctx.fillStyle = "rgba(0,255,200,0.9)";
      ctx.font = "12px monospace";

      path.forEach((p, idx) => {
        const px = p.x * SCALE;
        const py = CANVAS_SIZE - p.y * SCALE;

        ctx.fillText(String(idx + 1), px + 10, py - 6);
      });
    }

    plants.forEach((p, i) => {
      const px = p.x * SCALE;
      const py = CANVAS_SIZE - p.y * SCALE;

      const sev = p.severity;
      const r = Math.min(255, Math.round((sev / 10) * 255));
      ctx.fillStyle = `rgb(${r}, 100, 100)`;

      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.font = "10px monospace";
      ctx.fillText(String(i), px - 3, py + 4);

      if (p.severity === 0) {
        ctx.strokeStyle = "rgba(0,255,200,0.9)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    const current = airsimMode ? liveDronePos : dronePos;

    const dx = current.x * SCALE;
    const dy = CANVAS_SIZE - current.y * SCALE;

    ctx.fillStyle = "cyan";
    ctx.beginPath();
    ctx.arc(dx, dy, 10, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "#003";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(dx - 14, dy);
    ctx.lineTo(dx + 14, dy);
    ctx.moveTo(dx, dy - 14);
    ctx.lineTo(dx, dy + 14);
    ctx.stroke();
  }

  async function loadRandomPlants() {
    try {
      const res = await fetch("http://localhost:8000/generate/random-plants", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      const data = await res.json();
      console.log("Generated plants:", data);

      setPlants(data.plants);

      await savePlantsToBackend(data.plants);

      setOrder([]);
      setPath([]);
      setDronePos({ x: 0, y: 0 });
    } catch (e) {
      console.error(e);
      alert("Backend error: Could not load plants.");
    }
  }

  async function requestAIPath() {
    if (plants.length === 0) {
      alert("Load plants first!");
      return;
    }

    try {
      const res = await fetch("http://localhost:8000/predict/drone-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plants }),
      });

      const data = await res.json();

      if (!res.ok) {
        alert("Server Error: " + data.detail);
        return;
      }

      setPath(data.path);
      setOrder(data.order);
    } catch (e) {
      console.error(e);
      alert("Backend failed during path generation.");
    }
  }

  async function runSimulation() {
    if (path.length === 0) {
      alert("Request AI Path first!");
      return;
    }

    setRunning(true);
    let current = { x: 0, y: 0 };

    for (let i = 0; i < path.length; i++) {
      const next = { x: path[i].x, y: path[i].y };

      await animateMove(current, next, speedMs);

      const idx = order[i];
      setPlants((prev) => {
        const copy = prev.map((p) => ({ ...p }));
        if (copy[idx]) copy[idx].severity = 0;
        return copy;
      });

      current = next;
    }

    setRunning(false);
  }

  function animateMove(
    from: { x: number; y: number },
    to: { x: number; y: number },
    duration = 600
  ) {
    const fps = 60;
    const frames = Math.max(1, Math.round((duration / 1000) * fps));

    return new Promise<void>((resolve) => {
      let frame = 0;

      function step() {
        frame++;

        const t = frame / frames;
        const tt = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

        setDronePos({
          x: from.x + (to.x - from.x) * tt,
          y: from.y + (to.y - from.y) * tt,
        });

        if (frame < frames) requestAnimationFrame(step);
        else {
          setDronePos({ x: to.x, y: to.y });
          setTimeout(resolve, 120);
        }
      }

      requestAnimationFrame(step);
    });
  }

  function resetAll() {
    setPlants([]);
    setPath([]);
    setOrder([]);
    setDronePos({ x: 0, y: 0 });
    setRunning(false);
  }

  function toggleAirSimMode() {
    setAirSimMode((prev) => !prev);
  }

  const displayPos = airsimMode ? liveDronePos : dronePos;

  return (
    <div className="p-6 min-h-screen bg-slate-900 text-white">
      <h1 className="text-3xl font-bold mb-6">
        Precision Drug Delivery Simulation
      </h1>

      <div className="flex gap-8">
        <div>
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="border border-slate-700 rounded-lg shadow-xl"
          />
        </div>

        <div className="flex flex-col gap-4 w-64">
          <button
            onClick={toggleAirSimMode}
            className={`px-4 py-2 rounded ${
              airsimMode
                ? "bg-purple-600 hover:bg-purple-700"
                : "bg-gray-600 hover:bg-gray-700"
            }`}
          >
            {airsimMode ? "AirSim Mode: ON" : "AirSim Mode: OFF"}
          </button>

          <button
            onClick={loadRandomPlants}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-700"
          >
            Generate Random Plants
          </button>

          <button
            onClick={requestAIPath}
            className="px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-700"
          >
            Request AI Path
          </button>

          <button
            disabled={running}
            onClick={() => {
              if (airsimMode) {
                alert(
                  "AirSim Mode Enabled:\nRun flight.py for real drone simulation."
                );
                return;
              }
              runSimulation();
            }}
            className={`px-4 py-2 rounded ${
              running ? "bg-gray-500" : "bg-green-600 hover:bg-green-700"
            }`}
          >
            Start Simulation
          </button>

          <button
            onClick={resetAll}
            className="px-4 py-2 rounded bg-red-600 hover:bg-red-700"
          >
            Reset
          </button>

          <div className="mt-4 p-3 rounded bg-slate-800 border border-slate-700">
            <p className="text-sm opacity-80">Drone Position:</p>
            <p className="text-lg">
              ({displayPos.x.toFixed(2)}, {displayPos.y.toFixed(2)})
            </p>

            <p className="mt-2 opacity-80 text-sm">Plants: {plants.length}</p>
            <p className="opacity-80 text-sm">Path Steps: {path.length}</p>

            <div className="mt-2">
              <label className="text-sm opacity-80">Speed (ms per leg):</label>
              <input
                type="range"
                min={100}
                max={2000}
                value={speedMs}
                onChange={(e) => setSpeedMs(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-xs opacity-70">{speedMs} ms</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
