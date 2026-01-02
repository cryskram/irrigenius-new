"use client";
import React from "react";

export default function ModeSwitch({ mode, setMode }: any) {
  const modes = ["normal", "dry", "humid", "rainy"];

  return (
    <div className="flex gap-3">
      {modes.map((m) => (
        <button
          key={m}
          className={`px-4 py-2 rounded-lg border text-sm ${
            mode === m
              ? "bg-black text-white border-black"
              : "bg-white text-gray-700 border-gray-300"
          }`}
          onClick={() => setMode(m)}
        >
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  );
}
