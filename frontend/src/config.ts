export const PREFERRED_LOCAL_API = "http://127.0.0.1:8000";
export const PRODUCTION_API = "https://facial-emotions-prediction.onrender.com";

const isDev =
  (typeof process !== "undefined" && process.env.NODE_ENV === "development") ||
  (typeof import.meta !== "undefined" && (import.meta as any)?.env?.MODE === "development");

const fallbackApi = isDev ? PREFERRED_LOCAL_API : PRODUCTION_API;

export const API_URL: string = PRODUCTION_API;

export const PREDICT_PATH = "/predict";