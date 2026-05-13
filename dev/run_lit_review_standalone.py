"""
Test literature review node in isolation.

Requires:
- MCP server running (provides PubMed and other tools)
- API key env var for your chosen model (default Anthropic: ANTHROPIC_API_KEY)

Set COSCIENTIST_DEV_MODE=true for faster testing with reduced paper counts.

Create a .env file in this directory (dev/) with your API keys:
  ANTHROPIC_API_KEY=your_key
  MCP_SERVER_URL=http://localhost:8888/mcp  (or your MCP server URL)
"""

import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load .env file from dev/ directory if it exists
console = Console()

try:
    from dotenv import load_dotenv
    dev_dir = Path(__file__).parent
    env_file = dev_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        console.print(f"[dim]Loaded environment from {env_file}[/dim]")
    else:
        console.print(f"[dim]No .env file found at {env_file}, using system environment variables[/dim]")
except ImportError:
    console.print("[dim]python-dotenv not installed, using system environment variables only[/dim]")

from state_helpers import make_base_state
from logging_utils import initialize_run_logging, cleanup_run_logging
from open_coscientist.nodes.literature_review import literature_review_node
from open_coscientist.mcp_client import check_mcp_available, check_pubmed_available_via_mcp
from open_coscientist.constants import LITERATURE_REVIEW_FAILED
from open_coscientist.llm import (
    describe_model_backend,
    get_expected_api_key_env,
    get_configured_api_key_env,
    get_api_key_env_candidates,
)


async def test_literature_review():
    """Run literature review node with minimal state."""

    # Initialize run logging to avoid warnings
    run_id = f"lit_review_test_{int(asyncio.get_event_loop().time())}"
    initialize_run_logging(run_id)

    console.print("\n[bold cyan]Testing literature review node[/bold cyan]\n")

    # Check prerequisites
    console.print("[yellow]Checking prerequisites...[/yellow]\n")

    model_name = os.getenv("MODEL_NAME", "claude-haiku-4-5-20251001")
    model_info = describe_model_backend(model_name)
    console.print(
        f"[dim]LLM engine: {model_info['engine']} | provider: {model_info['provider']} | model: {model_info['full_model_name']}[/dim]"
    )

    # Check required API keys first
    errors = []
    warnings = []

    configured_api_key = get_configured_api_key_env(model_name)
    required_api_key = get_expected_api_key_env(model_name) or "OPENAI_API_KEY"
    if not configured_api_key:
        candidates = get_api_key_env_candidates(model_name)
        if candidates:
            errors.append(
                f"No API key found for model {model_name}. Set one of: {', '.join(candidates)}"
            )
        else:
            errors.append(f"{required_api_key} not set - required for model {model_name}")
    else:
        console.print(f"[green]{configured_api_key} found[/green]")

    # Check MCP server
    mcp_ok = await check_mcp_available()
    if not mcp_ok:
        errors.append("MCP server not available")
    else:
        console.print("[green]MCP server available[/green]")

    # Check PubMed via MCP
    pubmed_ok = await check_pubmed_available_via_mcp()
    if not pubmed_ok:
        warnings.append("PubMed not available via MCP - PubMed search will be disabled")
        console.print("[yellow]PubMed not available via MCP - will be disabled[/yellow]")
    else:
        console.print("[green]PubMed available via MCP[/green]")

    # Fail early if critical errors
    if errors:
        console.print("\n[bold red]Critical errors detected:[/bold red]")
        for error in errors:
            console.print(f"  [red]{error}[/red]")
        console.print("\n[yellow]Fix these issues and try again[/yellow]")
        console.print("[dim]Tip: create a .env file in dev/ directory with your API keys[/dim]")
        return

    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  [yellow]{warning}[/yellow]")

    dev_mode = os.getenv("COSCIENTIST_DEV_MODE", "").lower() == "true"
    if dev_mode:
        console.print("\n[yellow]Dev mode enabled - using reduced paper counts[/yellow]")

    # Create minimal state
    state = make_base_state(
        research_goal="How can we detect Alzheimer's disease earlier using retinal imaging?",
        model_name=model_name,
    )
    state["mcp_available"] = mcp_ok
    state["pubmed_available"] = pubmed_ok

    console.print(f"\n[yellow]Research goal:[/yellow] {state['research_goal']}\n")

    # Run node
    console.print("[yellow]Calling literature review node (this may take a couple of minutes)...[/yellow]\n")
    try:
        result = await literature_review_node(state)
    except Exception as e:
        import traceback
        console.print(f"\n[bold red]Literature review node crashed![/bold red]")
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        console.print("[dim]Full traceback:[/dim]")
        console.print(traceback.format_exc())
        return

    # Display results
    articles = result.get("articles", [])
    queries = result.get("literature_review_queries", [])
    summary = result.get("articles_with_reasoning", "")

    console.print(f"\n[dim]Debug - summary value: {repr(summary[:100] if summary else summary)}[/dim]")
    console.print(f"[dim]Debug - LITERATURE_REVIEW_FAILED: {repr(LITERATURE_REVIEW_FAILED)}[/dim]")
    console.print(f"[dim]Debug - summary == LITERATURE_REVIEW_FAILED: {summary == LITERATURE_REVIEW_FAILED}[/dim]")

    # Check if literature review failed and show error message
    if summary == LITERATURE_REVIEW_FAILED:
        console.print(f"\n[bold red]Literature review failed![/bold red]")
        console.print("[yellow]The system will fall back to standard generation without literature context[/yellow]")
        # Print error message if available
        messages = result.get("messages", [])
        if messages:
            console.print(f"\n[yellow]Messages ({len(messages)}):[/yellow]")
            for msg in messages:
                content = msg.get('content', '')
                is_error = msg.get("metadata", {}).get("error", False)
                style = "[red]" if is_error else "[dim]"
                console.print(f"{style}{content}[/{('red]' if is_error else 'dim]')}")
        return

    # Show article breakdown by source (PubMed-only now)
    pm_articles = [a for a in articles if a.source == "pubmed"]
    console.print(f"\n[bold yellow]Article breakdown:[/bold yellow]")
    console.print(f"  PubMed: {len(pm_articles)} total")

    # Queries table
    query_table = Table(title="Search queries generated")
    query_table.add_column("Query", style="cyan")
    for q in queries:
        query_table.add_row(q)
    console.print(query_table)

    # Articles table
    article_table = Table(title=f"Articles found ({len(articles)} total)")
    article_table.add_column("Title", style="cyan", max_width=50)
    article_table.add_column("Year", style="yellow")
    article_table.add_column("Citations", style="green")

    for article in articles[:10]:  # Show first 10
        article_table.add_row(
            article.title[:50] + "..." if len(article.title) > 50 else article.title,
            str(article.year) if article.year else "n/a",
            str(article.citations) if article.citations else "n/a",
        )

    console.print(article_table)

    # Summary
    console.print(Panel(
        summary[:500] + "..." if len(summary) > 500 else summary,
        title="[bold green]Literature review summary (first 500 chars)[/bold green]",
        border_style="green"
    ))

    console.print(f"\n[bold]Summary stats:[/bold]")
    console.print(f"  Queries generated: {len(queries)}")
    console.print(f"  Articles found: {len(articles)}")
    console.print(f"  Summary length: {len(summary)} chars")
    console.print(f"  Articles with reasoning available: {bool(summary)}")

    # Cleanup run logging
    cleanup_run_logging()


if __name__ == "__main__":
    try:
        asyncio.run(test_literature_review())
    except Exception as e:
        import traceback
        console.print(f"\n[bold red]Fatal error:[/bold red]")
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        console.print("[dim]Full traceback:[/dim]")
        console.print(traceback.format_exc())
    finally:
        cleanup_run_logging()
