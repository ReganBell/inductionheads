# Deployment Guide

## Deploy to reganbell.com

### Step 1: Deploy Backend to Railway (api.reganbell.com)

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy Backend**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli

   # Login
   railway login

   # From the root directory
   railway init

   # Set start directory to backend
   railway up
   ```

   In Railway dashboard:
   - Set **Start Command**: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Or set **Root Directory**: `backend`

3. **Configure Domain**
   - In Railway dashboard, go to your service settings
   - Add custom domain: `api.reganbell.com`
   - Railway will provide a CNAME record

4. **Update DNS (GoDaddy)**
   - Go to GoDaddy DNS settings for reganbell.com
   - Add CNAME record:
     - Type: CNAME
     - Name: api
     - Value: (the CNAME from Railway)
     - TTL: 600

5. **Set Environment Variable (if needed)**
   ```bash
   railway variables set PORT=8000
   ```

### Step 2: Deploy Frontend to Cloudflare Pages

1. **Push to GitHub**
   ```bash
   git push origin main
   ```

2. **Create Cloudflare Pages Project**
   - Go to [dash.cloudflare.com](https://dash.cloudflare.com)
   - Pages → Create a project → Connect to Git
   - Select your `inductionviz` repository

3. **Configure Build Settings**
   - Framework preset: Vite
   - Build command: `cd frontend && npm install && npm run build`
   - Build output directory: `frontend/dist`
   - Root directory: `/` (leave blank)

4. **Add Environment Variable**
   - In Cloudflare Pages project settings → Environment variables
   - Add variable:
     - Name: `VITE_API_URL`
     - Value: `https://api.reganbell.com`
     - Production only

5. **Deploy**
   - Click "Save and Deploy"
   - Wait for build to complete

6. **Configure Custom Domain**
   - In Cloudflare Pages → Custom domains
   - Add domain: `inductionviz.reganbell.com`
   - Cloudflare will auto-configure DNS if domain is on Cloudflare
   - OR add to your GoDaddy DNS:
     - Type: CNAME
     - Name: inductionviz
     - Value: (provided by Cloudflare)

### Step 3: Test

Visit `https://inductionviz.reganbell.com` and verify:
- Page loads correctly
- Can analyze text
- Backend API calls work
- Composition visualization loads

## Costs

- **Railway**: ~$5-10/month (Hobby plan, pay-as-you-go)
- **Cloudflare Pages**: Free (unlimited bandwidth)
- **Total**: ~$5-10/month

## Local Development

Backend:
```bash
cd backend
uvicorn app:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

Frontend will use `http://localhost:8000` for API calls automatically.
