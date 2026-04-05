# World Current Affairs Pulse

A lightweight Python news website that covers geopolitics, technology, education, and economics with:

- cause behind the event
- impact on India
- impact on major countries
- impact on citizens

## Run locally

```powershell
cd C:\Users\kanar\OneDrive\Desktop\ai_news_agent
.\venv\Scripts\python.exe .\news_agent.py
```

Open `http://127.0.0.1:8000`.

## Environment variables

Create a `.env` file with:

```env
NEWS_API_KEY=your_newsapi_key
GROQ_API_KEY=your_groq_key
```

Optional:

```env
GROQ_MODEL=llama3-8b-8192
PORT=8000
HOST=0.0.0.0
CACHE_TTL_SECONDS=900
```

## Deploy on Render

1. Push this project to GitHub.
2. Create an account on [Render](https://render.com/).
3. Click `New +` and choose `Blueprint`.
4. Connect your GitHub repository.
5. Render will detect `render.yaml`.
6. Add secret environment variables:
   - `NEWS_API_KEY`
   - `GROQ_API_KEY`
7. Deploy the service.
8. Share the generated public URL.

## Notes

- Keep `.env` private.
- Rotate any API keys that were shared publicly.
- If live APIs fail, the site falls back to a built-in briefing layout.
