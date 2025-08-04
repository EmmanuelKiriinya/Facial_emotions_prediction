// src/config.ts
export const API_URL = "http://127.0.0.1:8000"
  import.meta.env.MODE === "development"
    ? "http://127.0.0.1:8000" // local FastAPI
    : "https://facial-emotions-prediction.onrender.com"; // Render backend
