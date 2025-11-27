import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import "./App.css";

export default function StockDetails() {
  const { symbol } = useParams();
  const navigate = useNavigate();

  const [stock, setStock] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const API_BASE = import.meta.env.VITE_API_URL;
  useEffect(() => {
    if (!symbol) return;

    const loadStock = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/stocks/${symbol}`);
        const data = await res.json();

        if (data.error) {
          console.error(data.error);
          return;
        }

        setStock({
          symbol: data.symbol,
          price: data.last_price,
          predicted: data.predicted_price,
          timestamp: data.timestamp,
        });
      } catch (err) {
        console.error("Error loading stock:", err);
      } finally {
        setLoading(false);
      }
    };

    loadStock();
  }, [symbol]);

  if (loading || !stock) return <p>Loading...</p>;

  return (
    <div className="details-container glass-card">
      <button onClick={() => navigate("/")} className="back-btn">
        ← Back
      </button>

      <h1 className="details-title">{stock.symbol}</h1>

      <p className="details-line">
        💰 Current Price:
        <span className="highlight">${stock.price.toFixed(2)}</span>
      </p>

      <p className="details-line predicted">
        📈 Predicted Price:
        <span className="highlight green">${stock.predicted.toFixed(2)}</span>
      </p>

      <p className="timestamp">
        ⏱ Updated: {new Date(stock.timestamp).toLocaleString()}
      </p>
    </div>
  );
}
