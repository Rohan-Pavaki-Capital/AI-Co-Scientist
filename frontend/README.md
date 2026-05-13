# Frontend

Simple Streamlit UI for running hypothesis generation.

## Setup

1. Install dependencies:

```
pip install -r frontend/requirements.txt
```

2. Run the app:

```
streamlit run frontend/app.py
```

## Notes

- The app uses the same .env files as the CLI (repo root .env and dev/.env if present).
- For literature review and tool calling, an MCP server must be running.
