import { useEffect, useState } from "react";

interface Stock {
  symbol: string;
  price: number;
  change: number;
  percent: number;
}

interface StockDetail {
  symbol: string;
  last_price: number;
  predicted_price: number;
}

export default function StockList() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [selected, setSelected] = useState<StockDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/api/stocks")
      .then((res) => res.json())
      .then(setStocks)
      .catch((err) => console.error(err));
  }, []);

  const handleSelect = async (symbol: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/predict?symbol=${symbol}`);
      const data = await res.json();
      setSelected(data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="p-6 flex flex-col items-center">
      <h1 className="text-2xl font-bold mb-4">📊 Top 10 Stocks</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {stocks.map((s) => (
          <div
            key={s.symbol}
            onClick={() => handleSelect(s.symbol)}
            className="border p-4 rounded-lg shadow hover:bg-gray-100 transition cursor-pointer text-center"
          >
            <h2 className="font-semibold text-lg">{s.symbol}</h2>
            <p>💰 {s.price}</p>
            <p
              className={
                s.change >= 0 ? "text-green-600" : "text-red-600"
              }
            >
              {s.change >= 0 ? "+" : ""}
              {s.change} ({s.percent}%)
            </p>
          </div>
        ))}
      </div>

      {loading && <p className="mt-6 text-blue-600">Fetching data...</p>}

      {selected && !loading && (
        <div className="mt-6 border p-6 rounded-lg shadow-lg text-center w-80">
          <h2 className="text-xl font-semibold mb-2">{selected.symbol}</h2>
          <p>Last Price: ${selected.last_price}</p>
          <p className="text-blue-600 font-bold">
            Predicted Price: ${selected.predicted_price}
          </p>
        </div>
      )}
    </div>
  );
}
