import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center bg-slate-900 min-h-screen text-slate-200 w-full">
      <h1 className="font-black text-4xl text-center mt-10 max-w-4xl">
        A Multi-Module Artificial Intelligence Framework for Automated Crop
        Monitoring and Treatment
      </h1>
      <div className="grid grid-cols-3 gap-4 mt-20">
        <Link
          href="/water"
          className="w-full py-8 px-6 font-bold bg-slate-800 hover:bg-slate-700 transition-colors duration-300 text-white text-center border-2 border-slate-700 rounded-2xl hover:shadow-white hover:shadow-sm"
        >
          Water Monitoring
        </Link>
        <Link
          href="/disease"
          className="w-full py-8 px-6 font-bold bg-slate-800 hover:bg-slate-700 transition-colors duration-300 text-white text-center border-2 border-slate-700 rounded-2xl hover:shadow-white hover:shadow-sm"
        >
          Disease Detection
        </Link>
        <Link
          href="/drone"
          className="w-full py-8 px-6 font-bold bg-slate-800 hover:bg-slate-700 transition-colors duration-300 text-white text-center border-2 border-slate-700 rounded-2xl hover:shadow-white hover:shadow-sm"
        >
          Precision Drug Delivery
        </Link>
      </div>
    </div>
  );
}
