import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import "./App.css";

export default function StockDetails() {
  const { symbol } = useParams();
  const navigate = useNavigate();

  const [stock, setStock] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!symbol) return;

    const loadStock = async () => {
      try {
        const res = await fetch(`/api/stocks/${symbol}`);
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
    <div className="details-container">
      <button onClick={() => navigate("/")} className="back-btn">
        ← Back
      </button>

      <h1>{stock.symbol}</h1>

      <p> Current Price: ${stock.price.toFixed(2)}</p>

      <p className="predicted">
         Predicted Price: ${stock.predicted.toFixed(2)}
      </p>

      <p className="timestamp">
        Updated At: {new Date(stock.timestamp).toLocaleString()}
      </p>
    </div>
  );
}
