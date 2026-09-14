# Zomato PersonalAI — Tool 3 Streamlit Prototype

Academic prototype for the Advanced Digital Marketing & Agentic AI assessment.

## Architecture

Natural-language request → Gemini intent extraction → Python validation → deterministic recommendation engine → Gemini explanation → feedback capture.

## Data

All customer, restaurant, order and interaction records are synthetic/illustrative.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Gemini key

For local development, use Streamlit secrets:

`.streamlit/secrets.toml`

```toml
GEMINI_API_KEY = "YOUR_KEY"
```

Never commit this file to GitHub.

## Deploy

Deploy `app.py` from GitHub using Streamlit Community Cloud. Add `GEMINI_API_KEY` in the app's Secrets / Advanced settings rather than placing it in source code.
