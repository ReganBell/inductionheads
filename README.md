# Induction Heads Visualization

Interactive visualization tool for exploring induction heads in transformer models.

## Features

- Visual transformer architecture with composition connections
- Token-level attention analysis (attention & value-weighted)
- Head ablation analysis
- Q-, K-, and V-composition visualization
- Support for 1-layer and 2-layer attention-only transformers

## Deployment

### Frontend (Cloudflare Pages)

1. Push code to GitHub
2. Go to [Cloudflare Pages](https://pages.cloudflare.com/)
3. Connect your GitHub repository
4. Configure build settings:
   - **Build command**: `cd frontend && npm install && npm run build`
   - **Build output directory**: `frontend/dist`
   - **Root directory**: `/` (or leave empty)

### Backend (Railway/Render/Fly.io)

The backend requires:
- Python 3.9+
- PyTorch
- TransformerLens

#### Deploy to Railway:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

#### Deploy to Render:
1. Create a new Web Service
2. Connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`

#### Update Frontend API URL:
After deploying backend, update the frontend to point to your backend URL by modifying the fetch calls in `frontend/src/main.tsx` to use `https://your-backend-url.com/api/...` instead of `/api/...`

## Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Tech Stack

- Frontend: React, TypeScript, Vite
- Backend: Python, FastAPI, TransformerLens, PyTorch
- Visualization: Custom SVG rendering
