"""
Example for Open Coscientist with streaming output.

This demonstrates hypothesis generation with literature review integration,
showing real-time streaming of results as they're generated.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
if src_root.exists():
    sys.path.insert(0, str(src_root))

try:
    from dotenv import load_dotenv

    # Load common local env files if present.
    load_dotenv(repo_root / ".env")
    load_dotenv(repo_root / "dev" / ".env")
except Exception:
    # Optional convenience only; script still works with shell env vars.
    pass

from open_coscientist import HypothesisGenerator
from open_coscientist.console import ConsoleReporter, default_progress_callback, run_console
from open_coscientist.llm import describe_model_backend
# install rich in your environment
from rich.console import Console
from rich.panel import Panel
"""
Prerequisites:
- MCP server running (on http://localhost:8888/mcp)
- Set ANTHROPIC_API_KEY in your environment before running.
"""

MODEL_NAME = "claude-haiku-4-5-20251001"

async def main():
    # Prompt user for research goal with rich formatting
    console = Console()
    model_info = describe_model_backend(MODEL_NAME)
    console.print()
    console.print(
        f"[dim]LLM engine: {model_info['engine']} | provider: {model_info['provider']} | model: {model_info['full_model_name']}[/dim]"
    )
    console.print(
        f"[dim]Expected API key: {model_info['api_key_env'] or 'unknown'} | Present: {model_info['api_key_present'] if model_info['api_key_env'] else 'unknown'}[/dim]"
    )
    console.print(
        Panel(
            "[bold]Enter research goal[/bold]\n\n"
            "[dim]For example:[/dim] Develop novel approaches for early detection of "
            "Alzheimer's disease using non-invasive biomarkers",
            title="[cyan]Research Goal[/cyan]",
            border_style="cyan",
        )
    )
    research_goal = console.input("\n[bold cyan]Research goal:[/bold cyan] ").strip()
    if not research_goal:
        console.print("[bold red]Error:[/bold red] Research goal cannot be empty.")
        return
    generator = HypothesisGenerator(
        model_name=MODEL_NAME,
        max_iterations=2,
        initial_hypotheses_count=7,
        evolution_max_count=4,
    )

    # for rich terminal output
    reporter = ConsoleReporter()

    # wrap with built-in console/terminal reporter
    await reporter.run(
        event_stream=generator.generate_hypotheses(
            research_goal=research_goal,
            progress_callback=default_progress_callback,
            # explicitly enable literature review/generate with tool calling
            opts={
                "enable_literature_review_node": True,
                "enable_tool_calling_generation": True,
            },
            stream=True,
        ),
        research_goal=research_goal,
    )


if __name__ == "__main__":
    # wrap with run_console for graceful shutdown on KeyboardInterrupt and hide internal warnings
    run_console(main())
