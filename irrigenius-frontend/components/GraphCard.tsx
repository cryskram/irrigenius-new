"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function GraphCard({ label, data }: any) {
  return (
    <div className="rounded-2xl p-6 bg-white shadow-sm border border-gray-200 hover:shadow-md transition-all w-full">
      <div className="text-gray-800 font-medium text-sm mb-3">{label}</div>

      <div className="w-full h-40">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis dataKey="time" hide />
            <YAxis hide />

            <Tooltip
              contentStyle={{
                borderRadius: "12px",
                background: "white",
                border: "1px solid #e2e8f0",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
              }}
              labelFormatter={() => ""}
              formatter={(val: number) => [`${val.toFixed(1)}`, "Value"]}
            />

            <Line
              type="monotone"
              dataKey="value"
              stroke="#4F46E5"
              strokeWidth={3}
              dot={false}
              animationDuration={600}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
