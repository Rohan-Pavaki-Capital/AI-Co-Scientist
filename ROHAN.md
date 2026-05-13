# Open Coscientist - Simple Guide (ROHAN)

This file explains the full pipeline in simple language:
- How to start
- What each agent does
- What happens at each step
- How the loop works

---

## 1. What This Project Does

Open Coscientist helps you generate research hypotheses using multiple AI agents.

In simple terms:
1. It understands your research goal.
2. It can read literature (if MCP server is running).
3. It creates multiple hypotheses.
4. It reviews and ranks them.
5. It improves top hypotheses in loops.
6. It returns the best final set.

---

## 2. Quick Startup (Windows PowerShell)

Run these from project root: `open-coscientist`

### Step A: Create env and install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Step B: Add your LLM API key

Use any provider supported by LiteLLM.

```powershell
$env:TOGETHERAI_API_KEY="your_key_here"
# or:
# $env:OPENAI_API_KEY="your_key_here"
# $env:ANTHROPIC_API_KEY="your_key_here"
```

You can also put keys in `.env` or `dev/.env` (the example script loads both).

### Step C: Run the app

```powershell
python .\examples\run.py
```

It will ask for a research goal interactively.

---

## 3. Startup With Literature (Recommended)

If you want real paper-backed output, start MCP server too.

### Option 1: Docker (easiest)

```powershell
Copy-Item .\mcp_server\.env.example .\mcp_server\.env
# edit .\mcp_server\.env and set:
# ENTREZ_EMAIL=you@example.com
# ENTREZ_API_KEY=your_ncbi_key (optional but recommended)

docker compose up -d
curl http://localhost:8888
```

If server is up, MCP endpoint is: `http://localhost:8888/mcp`

Then run:

```powershell
python .\examples\run.py
```

To stop MCP later:

```powershell
docker compose down
```

### Option 2: Local MCP server (no Docker)

```powershell
cd .\mcp_server
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .\.env.example .\.env
# edit .\.env with ENTREZ_EMAIL (+ optional ENTREZ_API_KEY)
cd ..
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8888
```

---

## 4. Full Workflow Diagram (Simple)

```text
START
  |
  v
SUPERVISOR
  |
  +--> (if MCP available and enabled) LITERATURE REVIEW
  |                                      |
  |                                      v
  |                                   GENERATE
  |                                      |
  |                                      v
  |                                   REFLECTION
  |                                      |
  |                                      v
  +--> (if no MCP or disabled) --------> REVIEW
                                         |
                                         v
                                       RANKING
                                         |
                    +--------------------+--------------------+
                    |                                         |
      max_iterations == 0                             max_iterations > 0
                    |                                         |
                    v                                         v
                   END                                   META-REVIEW
                                                             |
                                                             v
                                                           EVOLVE
                                                             |
                                                             v
                                                           REVIEW
                                                             |
                                                             v
                                                           RANKING
                                                             |
                                                             v
                                                          PROXIMITY
                                                             |
                                +----------------------------+----------------------------+
                                |                                                         |
                       iteration < max_iterations                                iteration >= max_iterations
                                |                                                         |
                                v                                                         v
                          back to META-REVIEW                                             END
```

---

## 5. Every Agent in Simple Language

### Supervisor
- Reads your goal.
- Creates strategy for generation/review.
- Decides what to focus on.

### Literature Review (optional but recommended)
- Uses MCP tools (like PubMed).
- Finds relevant papers.
- Builds summary for downstream steps.

### Generate
- Creates initial hypotheses.
- Can use debate strategy, literature summaries, and optional live tool calls.
- Output for each hypothesis includes:
  - `text`
  - `explanation`
  - `literature_grounding`
  - `experiment`

### Reflection (runs when literature is enabled)
- Compares each hypothesis with literature findings.
- Adds classification-like notes:
  - already explained
  - other explanations more likely
  - missing piece
  - neutral
  - disproved

### Review
- Scores hypotheses on 6 criteria:
  - scientific soundness
  - novelty
  - relevance
  - testability
  - clarity
  - potential impact
- For small sets (<=5), it compares together.
- For larger sets (>5), it reviews in parallel.

### Ranking
- Orders hypotheses using review results.
- Sends hypotheses into pairwise comparisons.

### Tournament (pairwise judge)
- Compares two hypotheses at a time.
- Picks winner for each matchup.
- Updates Elo-style ratings after each matchup.
- Feeds those ratings back to ranking.

### Meta-Review (iteration phase)
- Looks at all review outcomes together.
- Finds recurring strengths/weaknesses.
- Suggests what to improve next.

### Evolve
- Refines top hypotheses using feedback.
- Keeps diversity while improving quality.

### Proximity
- Finds near-duplicate hypotheses.
- Removes/merges redundant ideas.

---

## 6. Step-by-Step Pipeline (What Happens in One Run)

1. You give a research goal.
2. Supervisor creates plan.
3. If literature is enabled and MCP is available:
   - Literature Review runs first.
4. Generate creates initial hypothesis set.
5. Reflection checks hypotheses vs literature (if enabled).
6. Review scores all hypotheses.
7. Ranking orders them and updates Elo.
8. If `max_iterations = 0`, run ends here.
9. If `max_iterations > 0`, iterative cycle begins:
   - Meta-Review -> Evolve -> Review -> Ranking -> Proximity
10. Loop repeats until iteration limit reached.
11. Final ranked hypotheses are returned.

---

## 7. Generation Modes (Simple)

## Mode 1: No literature (fastest)
- `opts = {"enable_literature_review_node": False}`
- Best for quick testing.

## Mode 2: Literature-informed (recommended default)
- `opts = {"enable_literature_review_node": True}`
- Good speed + good grounding.

## Mode 3: Tool-calling generation (best grounding, slower)
- `opts = {"enable_literature_review_node": True, "enable_tool_calling_generation": True}`
- Generate node can call tools directly during generation.

---

## 8. Main Config Knobs You Will Use Most

In `HypothesisGenerator(...)`:
- `model_name`: which LLM to use
- `max_iterations`: number of improve loops
- `initial_hypotheses_count`: size of initial pool
- `evolution_max_count`: how many top items to evolve each loop
- `enable_cache`: reuse prior LLM calls

In `generate_hypotheses(..., opts={...})`:
- `enable_literature_review_node`
- `enable_tool_calling_generation`
- `dev_test_lit_tools_isolation` (dev only)

---

## 9. Common Practical Notes

- If MCP is not running, literature step is auto-disabled and pipeline falls back to no-literature mode.
- `stream=True` gives live progress updates while running.
- Non-streaming runs can feel long (docs mention sometimes >10 minutes when uncached).
- Cache folder `.coscientist_cache` is safe to delete any time.

---

## 10. Minimal Example Code

```python
import asyncio
from open_coscientist import HypothesisGenerator

async def main():
    generator = HypothesisGenerator(
        model_name="together_ai/deepseek-ai/DeepSeek-V4-Pro",
        max_iterations=1,
        initial_hypotheses_count=5,
        evolution_max_count=3,
    )

    async for node_name, state in generator.generate_hypotheses(
        research_goal="Your research goal here",
        stream=True,
        opts={
            "enable_literature_review_node": True,
            "enable_tool_calling_generation": False,
        },
    ):
        print("Completed:", node_name)

asyncio.run(main())
```

---

If you want, this file can be further customized for your exact setup (provider, model, MCP source mix, and desired speed vs quality profile).
