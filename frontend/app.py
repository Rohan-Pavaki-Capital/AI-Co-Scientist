import asyncio
import sys
import threading
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if load_dotenv:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / "dev" / ".env")

from open_coscientist import HypothesisGenerator


def run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container = {}
    error_container = {}

    def runner():
        try:
            result_container["result"] = asyncio.run(coro)
        except Exception as exc:
            error_container["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("result")


def parse_cited_papers(raw_text: str) -> list[str]:
    papers = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if cleaned:
            papers.append(cleaned)
    return papers


def format_log_line(label: str, message: str) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {label}: {message}"


st.set_page_config(page_title="Open Coscientist", layout="wide")

st.title("Open Coscientist")
st.write("Streamlit frontend for hypothesis generation.")

with st.sidebar:
    st.header("Configuration")
    model_name = "claude-haiku-4-5-20251001"
    st.text_input("Model name", value=model_name, disabled=True)
    max_iterations = st.number_input("Max iterations", min_value=0, max_value=5, value=1, step=1)
    initial_hypotheses_count = st.number_input(
        "Initial hypotheses", min_value=1, max_value=12, value=5, step=1
    )
    evolution_max_count = st.number_input(
        "Evolution count", min_value=0, max_value=10, value=3, step=1
    )
    enable_literature_review_node = st.checkbox(
        "Enable literature review (MCP required)", value=True
    )
    enable_tool_calling_generation = st.checkbox(
        "Enable tool calling generation", value=False
    )

st.subheader("Research goal")
research_goal = st.text_area(
    "Describe your research goal", value="", height=240, placeholder="Enter a clear research goal"
)
st.caption(f"Characters: {len(research_goal)}")

if research_goal.strip():
    with st.expander("Full research goal preview", expanded=False):
        st.write(research_goal)

st.subheader("Cited research papers")
cited_papers_text = st.text_area(
    "Paste cited papers, one per line",
    value="",
    height=180,
    placeholder=(
        "One paper per line, for example:\n"
        "Smith et al. 2023 - Title - DOI or URL\n"
        "Jones et al. 2024 - Title - DOI or URL"
    ),
)
st.caption("These entries are passed into the workflow as user-provided literature context.")

if cited_papers_text.strip():
    cited_papers = parse_cited_papers(cited_papers_text)
    with st.expander(f"Parsed cited papers ({len(cited_papers)})", expanded=False):
        for paper in cited_papers:
            st.write(f"- {paper}")
else:
    cited_papers = []

run_clicked = st.button("Generate hypotheses", type="primary")

log_placeholder = st.empty()
status_placeholder = st.empty()
result_placeholder = st.container()

if "run_logs" not in st.session_state:
    st.session_state["run_logs"] = []


def append_log(label: str, message: str):
    st.session_state["run_logs"].append(format_log_line(label, message))
    log_placeholder.code("\n".join(st.session_state["run_logs"][-40:]), language="text")


async def progress_callback(phase: str, data: dict):
    message = data.get("message") or phase
    append_log(phase, str(message))
    status_placeholder.info(message)


async def run_generation_async(generator: HypothesisGenerator, goal: str, opts: dict):
    final_state = None
    async for node_name, state in generator.generate_hypotheses(
        research_goal=goal,
        progress_callback=progress_callback,
        opts=opts,
        stream=True,
    ):
        final_state = state
        metrics = state.get("metrics", {})
        append_log(
            node_name,
            (
                f"progressed to {node_name} | "
                f"hypotheses={len(state.get('hypotheses', []))} | "
                f"llm_calls={metrics.get('llm_calls', 0)}"
            ),
        )
        status_placeholder.info(f"Completed {node_name}")

    return final_state

if run_clicked:
    if not research_goal.strip():
        st.error("Research goal cannot be empty.")
    else:
        st.session_state["run_logs"] = []
        opts = {
            "enable_literature_review_node": enable_literature_review_node,
            "enable_tool_calling_generation": enable_tool_calling_generation,
            "user_inputs": {
                "literature": cited_papers,
            },
        }

        generator = HypothesisGenerator(
            model_name=model_name,
            max_iterations=int(max_iterations),
            initial_hypotheses_count=int(initial_hypotheses_count),
            evolution_max_count=int(evolution_max_count),
        )

        append_log("startup", "Starting hypothesis generation")
        with st.spinner("Running hypothesis generation. This can take a while."):
            result = run_async(run_generation_async(generator, research_goal.strip(), opts))

        st.session_state["last_result"] = result
        st.success("Generation complete.")

result = st.session_state.get("last_result")
if result:
    with result_placeholder:
        st.subheader("Summary")
        metrics = result.get("metrics", {})
        st.write(
            f"Hypotheses: {len(result.get('hypotheses', []))} | "
            f"LLM calls: {metrics.get('llm_calls', 0)} | "
            f"Total time: {metrics.get('total_time', 0):.1f}s"
        )

        meta_review = result.get("meta_review", {})
        if meta_review:
            st.subheader("Meta review")
            summary = meta_review.get("summary", "")
            if summary:
                st.write(summary)

        st.subheader("Hypotheses")
        for index, hyp in enumerate(result.get("hypotheses", []), start=1):
            score = hyp.get("score", 0.0)
            elo = hyp.get("elo_rating", 0)
            header = f"Hypothesis {index} | score {score:.2f} | elo {elo}"
            with st.expander(header):
                st.markdown("**Text**")
                st.write(hyp.get("text", ""))

                explanation = hyp.get("explanation")
                if explanation:
                    st.markdown("**Explanation**")
                    st.write(explanation)

                experiment = hyp.get("experiment")
                if experiment:
                    st.markdown("**Experiment**")
                    st.write(experiment)

                literature = hyp.get("literature_grounding")
                if literature:
                    st.markdown("**Literature grounding**")
                    st.write(literature)

st.subheader("Process log")
if st.session_state.get("run_logs"):
    log_placeholder.code("\n".join(st.session_state["run_logs"][-40:]), language="text")
else:
    log_placeholder.info("No run logs yet. Start a generation run to see progress here.")
