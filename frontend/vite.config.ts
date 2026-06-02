import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api to the FastAPI backend so the app runs from a single origin in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Fail loudly if 5173 is taken instead of silently moving to 5174 (which is
    // confusing — you'd open 5173 and hit a stale process). Free it with:
    //   lsof -ti tcp:5173 | xargs kill -9
    strictPort: true,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
