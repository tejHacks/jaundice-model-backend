import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
    setPrediction('');
    setErrorMsg('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    const formData = new FormData();
    formData.append('image', image);

    try {
      setLoading(true);
      const res = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(res.data.prediction);
    } catch (err) {
      console.error(err);
      setErrorMsg('Something went wrong. Try another image.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial, sans-serif' }}>
      <h1>🧠 Jaundice Detection App</h1>

      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <br /><br />
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Submit Image'}
        </button>
      </form>

      {prediction && (
        <div style={{ marginTop: '2rem', color: 'green' }}>
          <h3>✅ Prediction Result:</h3>
          <p>{prediction === "0" ? "No Jaundice Detected" : "Signs of Jaundice Detected"}</p>
        </div>
      )}

      {errorMsg && (
        <div style={{ marginTop: '2rem', color: 'red' }}>
          <p>{errorMsg}</p>
        </div>
      )}
    </div>
  );
}

export default App;
