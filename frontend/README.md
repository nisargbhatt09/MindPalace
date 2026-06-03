# Welcome to the FrontEnd!

# MindPalace frontend

React + Vite + TypeScript web app. A calm, large-text interface with two panels:
ask about your day (chat), and your photos (gallery + upload).

## Setup & run

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to the backend on
port 8000, so start the backend first (see `../backend/README.md`).

## Structure

```
src/
  api.ts                  fetch wrappers for the backend
  types.ts                shared TypeScript types
  App.tsx                 layout (chat + gallery panels)
  components/Chat.tsx     conversation UI; renders the photos each answer used
  components/MemoryGallery.tsx   gallery + multi-file upload
  styles.css              calm, accessible styling
```
