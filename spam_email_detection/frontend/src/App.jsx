import { useState } from "react";
import axios from "axios";

const App = () => {
  const [message, setMessage] = useState("");
  const [prediction, setPrediction] = useState("");

  const handleCheck = async () => {
    if (!message.trim()) {
      setPrediction("⚠️ Please enter a message.");
      return;
    }
    try {
      const response = await axios.post("http://127.0.0.1:5000/", { message });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error:", error);
      setPrediction("❌ Error checking message.");
    }
  };

  return (
    <div className="container">
      <h1>📩 SMS Spam Detector</h1>
      <textarea
        className="message-box"
        placeholder="Enter SMS message here..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />
      <button className="check-btn" onClick={handleCheck}>
        🔍 Check Message
      </button>
      {prediction && <p className="result">{prediction}</p>}
    </div>
  );
};

export default App;
