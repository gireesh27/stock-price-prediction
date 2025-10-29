import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
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

export default function App() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [selected, setSelected] = useState<Stock | null>(null);
  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const topSymbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NFLX", "NVDA", "IBM", "ORCL",
  ];

  useEffect(() => {
    const fetchStocks = async () => {
      try {
        const responses = await Promise.all(
          topSymbols.map((s) =>
            fetch(`/api/quote?symbol=${s}`).then((res) => res.json())
          )
        );

        const data = responses.map((r, i) => ({
          symbol: topSymbols[i],
          price: r.c,
          change: r.d,
          changePercent: r.dp,
          high: r.h,
          low: r.l,
          volume: r.v || 0,
        }));

        setStocks(data);
      } catch (error) {
        console.error("Error fetching stock data:", error);
      }
    };

    fetchStocks();
  }, []);

  const handleSelect = async (symbol: string) => {
    const selectedStock = stocks.find((s) => s.symbol === symbol);
    if (!selectedStock) return;

    setSelected(selectedStock);
    setPredictedPrice(null);
    setLoading(true);

    try {
      const res = await fetch(`/api/predict?symbol=${symbol}`);
      const data = await res.json();
      setPredictedPrice(data.predicted_price);
    } catch (err) {
      console.error("Prediction fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">📈 Top 10 Trending Stocks</h1>

      <div className="stock-grid">
        {stocks.map((s) => (
          <motion.div
            key={s.symbol}
            whileHover={{ scale: 1.05 }}
            className="stock-card"
            onClick={() => handleSelect(s.symbol)}
          >
            <div className="stock-header">
              <span className="symbol">{s.symbol}</span>
              <span className={s.change >= 0 ? "positive" : "negative"}>
                {s.change >= 0 ? "+" : ""}
                {s.change.toFixed(2)} ({s.changePercent.toFixed(2)}%)
              </span>
            </div>
            <p>💰 Price: ${s.price.toFixed(2)}</p>
            <p>🔺 High: ${s.high}</p>
            <p>🔻 Low: ${s.low}</p>
            <p>📊 Volume: {s.volume}</p>
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {selected && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="modal"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
            >
              <h2>{selected.symbol}</h2>
              <p>💰 Current Price: ${selected.price.toFixed(2)}</p>
              <p>🔺 High: ${selected.high}</p>
              <p>🔻 Low: ${selected.low}</p>
              <p>📊 Volume: {selected.volume}</p>

              {loading ? (
                <p className="loading">Predicting next price...</p>
              ) : (
                predictedPrice && (
                  <p className="predicted">
                    🔮 Predicted Price: ${predictedPrice.toFixed(2)}
                  </p>
                )
              )}

              <button onClick={() => setSelected(null)} className="close-btn">
                Close
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
