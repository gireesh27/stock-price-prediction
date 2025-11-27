import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

interface Stock {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  volume: number;
}

export default function Home() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const navigate = useNavigate();
  const API_BASE = import.meta.env.VITE_API_URL;

  useEffect(() => {
    const fetchStocks = async () => {
      try {
        console.log("API_BASE =", API_BASE);
        const res = await fetch(`${API_BASE}/api/stocks/latest`);
        const data = await res.json();

        const formatted = data.map((item: any) => ({
          symbol: item.symbol,
          price: item.price,
          change: item.change,
          changePercent: item.percent,
          high: item.high ?? 0,
          low: item.low ?? 0,
          volume: item.volume ?? 0,
        }));

        setStocks(formatted);
      } catch (err) {
        console.error("Error loading stocks:", err);
      }
    };

    fetchStocks();
  }, []);

  return (
    <div className="app-container">
      <h1 className="app-title">Trending Stocks - Top 10</h1>

      <div className="stock-grid">
        {stocks.map((s) => (
          <div
            key={s.symbol}
            className="stock-card glass-card"
            onClick={() => navigate(`/stock/${s.symbol}`)}
          >
            <div className="stock-header">
              <span className="symbol">{s.symbol}</span>
              <span className={s.change >= 0 ? "positive" : "negative"}>
                {s.change >= 0 ? "+" : ""}
                {s.change.toFixed(2)} ({s.changePercent.toFixed(2)}%)
              </span>
            </div>

            <div className="stock-info">
              <p>
                <strong>Price:</strong> ${s.price.toFixed(2)}
              </p>
              <p>
                <strong>High:</strong> ${s.high}
              </p>
              <p>
                <strong>Low:</strong> ${s.low}
              </p>
              <p>
                <strong>Volume:</strong> {s.volume}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
