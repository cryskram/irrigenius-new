export default function PHCard({ ph }: { ph: number }) {
  return (
    <div className="p-6 bg-white shadow rounded-lg text-center">
      <h2 className="text-lg font-semibold">Recommended Water pH</h2>
      <p className="text-3xl font-bold mt-2">{ph}</p>
    </div>
  );
}
