# Frontend Web

This folder contains a basic HTML, CSS, and JavaScript UI for running Open Coscientist workflows.

## Demo mode (no server required)

Open frontend_web/index.html in a browser and leave Run mode set to Demo data.

## Live mode (server + SSE)

Start the local server:

```
python frontend_web/server.py
```

Then open http://localhost:8080 in your browser. Set Run mode to Live server.

Notes:
- You still need your LLM API key configured (same as the CLI).
- If you want literature review, run the MCP server separately.
