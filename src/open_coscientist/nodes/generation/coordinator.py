"""
Generation coordinator - orchestrates all generation strategies.

All generation paths output hypotheses with explanation, literature_grounding, and experiment fields.
When no literature is available, literature_grounding contains an explicit warning message.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...constants import (
    PROGRESS_GENERATE_START,
    PROGRESS_GENERATE_COMPLETE,
    LITERATURE_REVIEW_FAILED,
)
from ...mcp_client import get_mcp_client
from ...models import Hypothesis
from ...state import WorkflowState
from .citations import ReferenceIndex, build_reference_index
from .debate import generate_with_debate
from .literature_tools import generate_with_tools

logger = logging.getLogger(__name__)


@dataclass
class GenerationCounts:
    """Encapsulates hypothesis count allocation across generation methods"""

    tools_count: int
    debate_with_lit_count: int
    debate_only_count: int
    is_dev_isolation: bool = False
    is_degraded_mode: bool = False


@dataclass
class GenerationResults:
    """Encapsulates results from parallel generation execution"""

    tools_hypotheses: List[Hypothesis]
    debate_with_lit_hypotheses: List[Hypothesis]
    debate_only_hypotheses: List[Hypothesis]
    debate_transcripts: List[Dict[str, Any]]


# helper functions


def _get_tools_generation_timeout_seconds() -> float:
    """
    Timeout guard for tool-based generation branch.

    This prevents a single slow/hanging tool-call pipeline from blocking the
    entire generation node when debate generation has already completed.
    """
    default_seconds = 420.0  # 7 minutes
    raw = os.getenv("COSCIENTIST_TOOLS_GENERATION_TIMEOUT_SECONDS")
    if not raw:
        return default_seconds

    try:
        parsed = float(raw)
        if parsed <= 0:
            raise ValueError("must be > 0")
        return parsed
    except ValueError:
        logger.warning(
            "Invalid COSCIENTIST_TOOLS_GENERATION_TIMEOUT_SECONDS='%s'; using default %.0fs",
            raw,
            default_seconds,
        )
        return default_seconds

def _check_literature_availability(
    articles_with_reasoning: Optional[str],
    mcp_available: bool
) -> bool:
    """Determine if literature review is available and valid"""
    return (
        articles_with_reasoning is not None
        and articles_with_reasoning != LITERATURE_REVIEW_FAILED
        and mcp_available
    )


def _determine_generation_counts(
    state: WorkflowState,
    total_count: int,
    has_literature: bool,
    enable_tool_calling: bool
) -> GenerationCounts:
    """Determine how many hypotheses to generate with each method"""
    if state.get("dev_test_lit_tools_isolation", False):
        return GenerationCounts(
            tools_count=total_count,
            debate_with_lit_count=0,
            debate_only_count=0,
            is_dev_isolation=True,
        )

    # condition (a)
    if has_literature and enable_tool_calling:
        # split 50/50, but ensure we don't exceed total_count
        tools_count = max(1, total_count // 2)
        debate_with_lit_count = total_count - tools_count
        # if total_count=1, tools_count=1, debate_with_lit_count=0
        # in this case, adjust to just use tools
        if debate_with_lit_count == 0:
            tools_count = total_count
        return GenerationCounts(
            tools_count=tools_count,
            debate_with_lit_count=debate_with_lit_count,
            debate_only_count=0,
        )

    # condition (c)
    if has_literature and not enable_tool_calling:
        return GenerationCounts(
            tools_count=0,
            debate_with_lit_count=total_count,
            debate_only_count=0,
        )

    # condition (b)
    return GenerationCounts(
        tools_count=0,
        debate_with_lit_count=0,
        debate_only_count=total_count,
        is_degraded_mode=True,
    )


def _log_generation_strategy(
    counts: GenerationCounts,
    total_count: int,
    mcp_available: bool,
):
    """Log which generation strategy is being used"""
    if counts.is_dev_isolation:
        logger.info(
            "Dev isolation mode: allocating all hypotheses to lit tools generation (no debate)"
        )
        return

    if counts.tools_count > 0 and counts.debate_with_lit_count > 0:
        logger.info(
            f"Condition (a): Generating {total_count} hypotheses with literature review "
            f"({counts.tools_count} tool-based + {counts.debate_with_lit_count} debate-with-literature)"
        )
    elif counts.debate_with_lit_count > 0:
        logger.info(
            f"Condition (c): Generating {total_count} hypotheses with debate-with-literature"
        )
    elif counts.is_degraded_mode:
        logger.warning("=" * 80)
        if mcp_available:
            logger.warning("Literature review unavailable or failed")
        else:
            logger.warning("No literature review tools available")
        logger.warning("Generating hypotheses from model latent knowledge only")
        logger.warning("=" * 80)


async def _emit_start_progress(state: WorkflowState, counts: GenerationCounts, total_count: int):
    """Emit progress callback for generation start"""
    progress_callback = state.get("progress_callback")
    if not progress_callback:
        return

    if counts.is_dev_isolation:
        await progress_callback(
            "generation_start",
            {
                "message": f"Generating {total_count} hypotheses with lit tools only (dev isolation mode)...",
                "progress": PROGRESS_GENERATE_START,
                "dev_isolation_mode": True,
            },
        )
    elif counts.tools_count > 0 and counts.debate_with_lit_count > 0:
        await progress_callback(
            "generation_start",
            {
                "message": f"Generating {total_count} hypotheses ({counts.tools_count} tool-based + {counts.debate_with_lit_count} debate-with-literature)...",
                "progress": PROGRESS_GENERATE_START,
            },
        )
    elif counts.debate_with_lit_count > 0:
        await progress_callback(
            "generation_start",
            {
                "message": f"Generating {total_count} hypotheses with debate-with-literature...",
                "progress": PROGRESS_GENERATE_START,
            },
        )
    elif counts.is_degraded_mode:
        await progress_callback(
            "generation_start",
            {
                "message": f"Generating {counts.debate_only_count} hypotheses without literature review...",
                "progress": PROGRESS_GENERATE_START,
                "literature_review_available": False,
                "degraded_mode": True,
            },
        )


async def _execute_generation_tasks(
    state: WorkflowState,
    counts: GenerationCounts,
    articles_with_reasoning: Optional[str],
    reference_index: ReferenceIndex,
) -> GenerationResults:
    """Execute parallel generation tasks and return results"""
    tools_hypotheses = []
    debate_with_lit_hypotheses = []
    debate_only_hypotheses = []
    debate_transcripts = []

    # collect tasks to run in parallel
    tasks = []

    if counts.tools_count > 0:
        logger.info(f"Running tool-based generation for {counts.tools_count} hypotheses")
        tools_timeout = _get_tools_generation_timeout_seconds()
        logger.info(f"Tool-based generation timeout guard: {tools_timeout:.0f}s")
        tools_task = asyncio.wait_for(
            generate_with_tools(state, counts.tools_count, reference_index), timeout=tools_timeout
        )
        tasks.append(("tools", tools_task))

    if counts.debate_with_lit_count > 0:
        logger.info(f"Running debate-with-literature for {counts.debate_with_lit_count} hypotheses")
        tasks.append(
            (
                "debate_lit",
                generate_with_debate(
                    state=state,
                    count=counts.debate_with_lit_count,
                    articles_with_reasoning=articles_with_reasoning,
                    reference_index=reference_index,
                ),
            )
        )

    if counts.debate_only_count > 0:
        logger.info(f"Running debate-only for {counts.debate_only_count} hypotheses")
        tasks.append(
            (
                "debate_only",
                generate_with_debate(
                    state=state,
                    count=counts.debate_only_count,
                    articles_with_reasoning=None,  # explicitly no literature
                    reference_index=ReferenceIndex(text="", sources={}),
                ),
            )
        )

    # run all tasks in parallel, allowing partial success if one branch fails
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

    # unpack results
    for i, (task_type, _) in enumerate(tasks):
        task_result = results[i]

        if isinstance(task_result, Exception):
            logger.warning(
                "Generation task '%s' failed (%s): %s",
                task_type,
                type(task_result).__name__,
                task_result,
            )
            if task_type == "tools":
                # Continue with debate outputs when tool-based path fails/times out.
                # If this run was tools-only, fallback to debate-with-literature so
                # the user still receives hypotheses.
                if counts.debate_with_lit_count == 0 and counts.debate_only_count == 0:
                    logger.warning(
                        "Tool-based generation was the only path and failed; falling back to debate generation"
                    )
                    fallback_hypotheses, fallback_transcripts = await generate_with_debate(
                        state=state,
                        count=counts.tools_count,
                        articles_with_reasoning=articles_with_reasoning,
                        reference_index=reference_index,
                    )
                    debate_with_lit_hypotheses = fallback_hypotheses
                    debate_transcripts.extend(fallback_transcripts)
                continue

            # Non-tools branch failed; continue so any successful branches can be returned.
            continue

        if task_type == "tools":
            tools_hypotheses = task_result
        elif task_type == "debate_lit":
            debate_with_lit_hypotheses, transcripts = task_result
            debate_transcripts.extend(transcripts)
        elif task_type == "debate_only":
            debate_only_hypotheses, transcripts = task_result
            debate_transcripts.extend(transcripts)

    if not tools_hypotheses and not debate_with_lit_hypotheses and not debate_only_hypotheses:
        raise RuntimeError("All generation branches failed; no hypotheses were produced")

    return GenerationResults(
        tools_hypotheses=tools_hypotheses,
        debate_with_lit_hypotheses=debate_with_lit_hypotheses,
        debate_only_hypotheses=debate_only_hypotheses,
        debate_transcripts=debate_transcripts,
    )


def _apply_degraded_mode_fallback(hypotheses: List[Hypothesis]):
    """
    Set explicit literature_grounding message for hypotheses without literature review
    """
    for hyp in hypotheses:
        # always overwrite in non-lit-mcp mode to prevent hallucinated citations
        hyp.literature_grounding = (
            "No literature review available. This hypothesis is based on the model's "
            "latent knowledge and has not been validated against current research literature. "
            "Novelty and scientific validity should be independently verified."
        )


def _log_generation_summary(results: GenerationResults):
    """Log summary of generated hypotheses"""
    total = (
        len(results.tools_hypotheses)
        + len(results.debate_with_lit_hypotheses)
        + len(results.debate_only_hypotheses)
    )
    logger.info(
        f"Generated {total} total hypotheses "
        f"({len(results.tools_hypotheses)} tool-based, {len(results.debate_with_lit_hypotheses)} debate-with-lit, "
        f"{len(results.debate_only_hypotheses)} debate-only)"
    )

    if results.tools_hypotheses:
        logger.debug(
            f"tool-based generation_methods: {[h.generation_method for h in results.tools_hypotheses]}"
        )
    if results.debate_with_lit_hypotheses:
        logger.debug(
            f"debate-with-Lit generation_methods: {[h.generation_method for h in results.debate_with_lit_hypotheses]}"
        )
    if results.debate_only_hypotheses:
        logger.debug(
            f"debate-only generation_methods: {[h.generation_method for h in results.debate_only_hypotheses]}"
        )


def _build_summary_message_parts(results: GenerationResults, counts: GenerationCounts) -> List[str]:
    """Build message parts for summary output"""
    parts = []
    if counts.tools_count > 0:
        parts.append(f"{len(results.tools_hypotheses)} tool-based")
    if counts.debate_with_lit_count > 0:
        parts.append(f"{len(results.debate_with_lit_hypotheses)} debate-with-literature")
    if counts.debate_only_count > 0:
        suffix = ""
        parts.append(f"{len(results.debate_only_hypotheses)} debate-only{suffix}")
    return parts


async def _emit_complete_progress(
    state: WorkflowState,
    results: GenerationResults,
    counts: GenerationCounts
):
    """Emit progress callback for generation complete"""
    progress_callback = state.get("progress_callback")
    if not progress_callback:
        return

    parts = _build_summary_message_parts(results, counts)
    all_hypotheses = (
        results.tools_hypotheses
        + results.debate_with_lit_hypotheses
        + results.debate_only_hypotheses
    )

    message = f"Generated {len(all_hypotheses)} hypotheses ({', '.join(parts)})"

    await progress_callback(
        "generation_complete",
        {
            "message": message,
            "progress": PROGRESS_GENERATE_COMPLETE,
            "hypotheses_count": len(all_hypotheses),
        },
    )


# enrichment


async def _enrich_hypotheses(
    hypotheses: List[Hypothesis],
    state: WorkflowState,
) -> None:
    """Run post-generation enrichment tools and attach results to hypotheses.

    Reads enrichment configs from the tool registry. For each config, calls
    the specified tool with each hypothesis's input_field value and stores
    the result in hypothesis.enrichments[output_key].
    """
    tool_registry = state.get("tool_registry")
    if not tool_registry:
        return

    enrichment_configs = tool_registry.get_enrichment_configs()
    if not enrichment_configs:
        return

    mcp_client = await get_mcp_client(tool_registry=tool_registry)

    for enrichment in enrichment_configs:
        tool_config = tool_registry.get_tool(enrichment.tool)
        if not tool_config:
            logger.warning(f"enrichment tool '{enrichment.tool}' not found in registry")
            continue

        output_key = enrichment.output_key or enrichment.tool
        logger.info(
            f"running enrichment '{output_key}' via {tool_config.mcp_tool_name} "
            f"for {len(hypotheses)} hypotheses"
        )

        for hyp in hypotheses:
            input_value = getattr(hyp, enrichment.input_field, hyp.text)
            try:
                result = await mcp_client.call_tool(
                    tool_config.mcp_tool_name,
                    topic=input_value,
                    max_results=enrichment.max_results,
                )
                parsed = json.loads(result) if isinstance(result, str) else result
                # extract nested array via results_path (e.g., "results" for NvdSearchResponse)
                if enrichment.results_path and isinstance(parsed, dict):
                    parsed = parsed.get(enrichment.results_path, parsed)
                hyp.enrichments[output_key] = parsed
            except Exception as e:
                logger.warning(f"enrichment '{output_key}' failed for hypothesis: {e}")
                hyp.enrichments[output_key] = {"error": str(e)}


# main coordinator function


async def generate_hypotheses(state: WorkflowState) -> Dict[str, Any]:
    """
    Coordinate hypothesis generation using appropriate strategies

    Implements 3-condition strategy:
    - Condition (a): lit review + tools → 50% tool-based + 50% debate-with-lit
    - Condition (b): no lit review → 100% debate-only
    - Condition (c): lit review but no tools → 100% debate-with-lit

    args:
        state: current workflow state

    returns:
        dict with hypotheses, debate_transcripts, metrics, and message
    """
    logger.info("Starting hypothesis generation")

    supervisor_guidance = state.get("supervisor_guidance")
    articles_with_reasoning = state.get("articles_with_reasoning")
    mcp_available = state.get("mcp_available", False)
    enable_tool_calling = state.get("enable_tool_calling_generation", False)
    total_count = state["initial_hypotheses_count"]

    if not supervisor_guidance:
        raise ValueError("No supervisor_guidance in state for node=generation")

    has_literature = _check_literature_availability(articles_with_reasoning, mcp_available)
    counts = _determine_generation_counts(state, total_count, has_literature, enable_tool_calling)

    reference_index = build_reference_index(
        articles=state.get("articles"),
        context_enrichment_sources=state.get("context_enrichment_sources"),
    )
    if not reference_index.is_empty():
        logger.info(
            f"Built reference index: {sum(1 for k in reference_index.sources if k.startswith('P'))} paper(s), "
            f"{sum(1 for k in reference_index.sources if k.startswith('KG'))} KG source(s)"
        )

    _log_generation_strategy(counts, total_count, mcp_available)
    await _emit_start_progress(state, counts, total_count)

    try:
        results = await _execute_generation_tasks(state, counts, articles_with_reasoning, reference_index)

        if counts.is_degraded_mode:
            _apply_degraded_mode_fallback(results.debate_only_hypotheses)

        _log_generation_summary(results)
        await _emit_complete_progress(state, results, counts)

        all_hypotheses = (
            results.tools_hypotheses
            + results.debate_with_lit_hypotheses
            + results.debate_only_hypotheses
        )

        # run post-generation enrichments (e.g., NVD CVE lookup)
        await _enrich_hypotheses(all_hypotheses, state)

        parts = _build_summary_message_parts(results, counts)
        message_content = f"Generated {len(all_hypotheses)} hypotheses ({', '.join(parts)})"

        return {
            "hypotheses": all_hypotheses,
            "debate_transcripts": results.debate_transcripts,
            "hypothesis_count": len(all_hypotheses),
            "message": message_content,
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
