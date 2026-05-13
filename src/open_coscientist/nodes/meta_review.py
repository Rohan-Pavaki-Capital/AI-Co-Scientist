"""
Meta-review node - synthesize insights from all reviews.
"""

import json
import logging
from typing import Any, Dict

from ..constants import (
    THINKING_MAX_TOKENS,
    MEDIUM_TEMPERATURE,
    PROGRESS_META_REVIEW_START,
    PROGRESS_META_REVIEW_COMPLETE,
)
from ..llm import call_llm_json
from ..models import create_metrics_update
from ..prompts import get_meta_review_prompt
from ..state import WorkflowState

logger = logging.getLogger(__name__)


async def meta_review_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Synthesize insights from all reviews across all hypotheses.

    This node analyzes all the reviews collectively to identify:
    - Common strengths and weaknesses
    - Promising research directions
    - Areas needing improvement
    - Strategic guidance for evolution

    Args:
        state: Current workflow state

    Returns:
        Dictionary with updated state fields (meta_review)
    """
    hypotheses = state["hypotheses"]
    logger.info(f"Synthesizing meta-review from {len(hypotheses)} hypotheses")

    # Emit progress
    if state.get("progress_callback"):
        await state["progress_callback"](
            "meta_review_start",
            {
                "message": "Synthesizing insights from all reviews...",
                "progress": PROGRESS_META_REVIEW_START,
            },
        )

    # Collect all reviews
    all_reviews = []
    for i, hyp in enumerate(hypotheses):
        if not hyp.reviews:
            continue

        # Get the latest review for each hypothesis
        latest_review = hyp.reviews[-1]

        review_data = {
            "hypothesis_index": i,
            "hypothesis_text": hyp.text[:200] + "..." if len(hyp.text) > 200 else hyp.text,
            "overall_score": latest_review.overall_score,
            "review_summary": latest_review.review_summary,
            "scores": latest_review.scores,
            "constructive_feedback": latest_review.constructive_feedback,
            "elo_rating": hyp.elo_rating,
            "win_loss_record": f"{hyp.win_count}W-{hyp.loss_count}L",
        }
        all_reviews.append(review_data)

    if not all_reviews:
        logger.warning("No reviews available for meta-review")
        return {
            "meta_review": {
                "summary": "No reviews available",
                "common_strengths": [],
                "common_weaknesses": [],
                "strategic_recommendations": [],
            }
        }

    # Format all reviews for the LLM
    reviews_text = json.dumps(all_reviews, indent=2)

    # Get supervisor guidance from state
    supervisor_guidance = state.get("supervisor_guidance")

    # Call LLM to synthesize meta-review
    prompt, schema = get_meta_review_prompt(
        research_goal=state["research_goal"],
        all_reviews=reviews_text,
        supervisor_guidance=supervisor_guidance,
        instructions=None,  # for the future
        tool_registry=state.get("tool_registry"),
    )

    # save prompt to disk for debugging
    from ..prompts import save_prompt_to_disk

    save_prompt_to_disk(
        run_id=state.get("run_id", "unknown"),
        prompt_name="meta_review",
        content=prompt,
        metadata={
            "prompt_length_chars": len(prompt),
            "hypotheses_count": len(hypotheses),
            "reviews_count": len(all_reviews),
        },
    )

    response = await call_llm_json(
        prompt=prompt,
        model_name=state["model_name"],
        max_tokens=THINKING_MAX_TOKENS,  # Meta-review needs more space for aggregating all reviews
        temperature=MEDIUM_TEMPERATURE,
        json_schema=schema,
    )

    def _coerce_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if isinstance(value, str):
            return [value]
        return []

    summary = (
        response.get("synthesis_summary")
        or response.get("meta_review_summary")
        or response.get("summary")
        or ""
    )

    common_strengths = _coerce_str_list(
        response.get("common_strengths") or response.get("strengths")
    )
    common_weaknesses = _coerce_str_list(
        response.get("common_weaknesses") or response.get("weaknesses")
    )
    emerging_themes = _coerce_str_list(response.get("emerging_themes"))

    recurring = response.get("recurring_themes")
    if isinstance(recurring, dict):
        if not common_strengths:
            common_strengths = _coerce_str_list(recurring.get("common_strengths"))
        if not common_weaknesses:
            common_weaknesses = _coerce_str_list(recurring.get("common_weaknesses"))
        if not emerging_themes:
            emerging_themes = _coerce_str_list(
                recurring.get("recurring_feedback") or recurring.get("themes")
            )
    elif isinstance(recurring, list) and not emerging_themes:
        themes: list[str] = []
        for item in recurring:
            if isinstance(item, dict):
                theme = item.get("theme")
                description = item.get("description")
                if theme and description:
                    themes.append(f"{theme}: {description}")
                elif theme:
                    themes.append(str(theme))
                elif description:
                    themes.append(str(description))
            elif item is not None:
                themes.append(str(item))
        emerging_themes = themes

    strategic_recommendations = response.get("strategic_recommendations") or response.get(
        "recommendations"
    )
    if isinstance(strategic_recommendations, dict):
        strategic_recommendations = [strategic_recommendations]
    elif isinstance(strategic_recommendations, str):
        strategic_recommendations = [strategic_recommendations]
    elif not isinstance(strategic_recommendations, list):
        strategic_recommendations = []

    diversity_assessment = response.get("diversity_assessment") or ""
    top_performers_analysis = response.get("top_performers_analysis") or ""
    areas_for_improvement = _coerce_str_list(response.get("areas_for_improvement"))

    process_assessment = response.get("process_assessment")
    if isinstance(process_assessment, dict):
        if not diversity_assessment:
            diversity_assessment = str(process_assessment.get("generation_process", ""))
        if not top_performers_analysis:
            top_performers_analysis = str(process_assessment.get("review_process", ""))
        if not areas_for_improvement:
            areas_for_improvement = _coerce_str_list(
                process_assessment.get("evolution_process")
            )

    meta_review = {
        "summary": summary,
        "common_strengths": common_strengths,
        "common_weaknesses": common_weaknesses,
        "emerging_themes": emerging_themes,
        "strategic_recommendations": strategic_recommendations,
        "diversity_assessment": diversity_assessment,
        "top_performers_analysis": top_performers_analysis,
        "areas_for_improvement": areas_for_improvement,
    }

    logger.info("Meta-review complete")
    logger.info(f"Common strengths: {len(meta_review['common_strengths'])}")
    logger.info(f"Strategic recommendations: {len(meta_review['strategic_recommendations'])}")

    # Emit progress
    if state.get("progress_callback"):
        await state["progress_callback"](
            "meta_review_complete",
            {
                "message": "Meta-review synthesis complete",
                "progress": PROGRESS_META_REVIEW_COMPLETE,
                "strengths_count": len(meta_review["common_strengths"]),
                "recommendations_count": len(meta_review["strategic_recommendations"]),
            },
        )

    # Update metrics (deltas only, merge_metrics will add to existing state)
    metrics = create_metrics_update(llm_calls_delta=1)

    return {
        "meta_review": meta_review,
        "metrics": metrics,
        "messages": [
            {
                "role": "assistant",
                "content": "Synthesized meta-review from all hypotheses",
                "metadata": {
                    "phase": "meta_review",
                    "themes": len(meta_review.get("emerging_themes", [])),
                },
            }
        ],
    }
