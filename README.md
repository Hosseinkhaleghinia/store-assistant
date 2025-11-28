---
title: Store Assistant Backend
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# Store Assistant Backend API
This is the backend for the RAG Store Assistant, built with FastAPI and LangGraph.

# Store Assistant AI 🤖

An intelligent RAG-based assistant that provides smart answers to customers about products and services using AI.

This is a Full-Stack AI project using Python (FastAPI) and React (Vite).

## Prerequisites

### ⚠️ Important: Environment Variables (.env file)

**Before running the project, you MUST create a `.env` file in the root directory** with your AI provider API keys. Without this file, the application will not work!

Create a `.env` file in the root directory with the following variables:

```env
METIS_API_KEY=your_api_key_here
METIS_BASE_URL=https://your-openai-compatible-api-url
METIS_BASE_URL_GEMINI=https://your-gemini-api-url
```

**Why is `.env` important?**
- The backend needs API keys to connect to AI services (OpenAI, Gemini, etc.)
- Without valid API keys, the RAG agent cannot process requests
- The `.env` file keeps your sensitive keys secure and out of version control

You can use different AI providers like OpenAI, Google Gemini, or any compatible API service. Just update the URLs accordingly.

## How to Run

**Important:** You need **two separate terminal windows** running simultaneously. Keep both terminals open - do not close them!

### Terminal 1: Backend Server

Open your first terminal and go to the `backend/` folder:

```bash
cd backend
pip install -r requirements.txt
python src/server.py
```

**First-time setup:** Before running the server for the first time, you need to build the vector database:

```bash
python src/setup.py
```

**Keep this terminal open!** The backend server must remain running to handle API requests.

### Terminal 2: Frontend Development Server

Open a **second terminal window** (keep the first one running!) and go to the `frontend/` folder:

```bash
cd frontend
npm install
npm run dev
```

**Keep this terminal open too!** The frontend dev server needs to stay running for the React app to work.

---

**Summary:**
- ✅ Terminal 1: Backend running (`python src/server.py`)
- ✅ Terminal 2: Frontend running (`npm run dev`)
- ✅ Both terminals must stay open while using the application
