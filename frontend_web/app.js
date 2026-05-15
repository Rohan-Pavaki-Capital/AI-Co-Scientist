const steps = [
  { key: "supervisor", label: "Supervisor plan" },
  { key: "literature_review", label: "Literature review" },
  { key: "generate", label: "Generate hypotheses" },
  { key: "reflection", label: "Reflection" },
  { key: "review", label: "Review" },
  { key: "ranking", label: "Ranking and tournament" },
  { key: "meta_review", label: "Meta review" },
  { key: "evolve", label: "Evolution" },
  { key: "proximity", label: "Deduplication" },
];

const phaseToStep = {
  supervisor_start: { step: "supervisor", status: "running" },
  supervisor_complete: { step: "supervisor", status: "done" },
  literature_review_start: { step: "literature_review", status: "running" },
  literature_review_complete: { step: "literature_review", status: "done" },
  generation_start: { step: "generate", status: "running" },
  generation_complete: { step: "generate", status: "done" },
  reflection_start: { step: "reflection", status: "running" },
  reflection_complete: { step: "reflection", status: "done" },
  review_start: { step: "review", status: "running" },
  review_complete: { step: "review", status: "done" },
  tournament_start: { step: "ranking", status: "running" },
  tournament_complete: { step: "ranking", status: "done" },
  meta_review_start: { step: "meta_review", status: "running" },
  meta_review_complete: { step: "meta_review", status: "done" },
  evolve_start: { step: "evolve", status: "running" },
  evolve_complete: { step: "evolve", status: "done" },
  proximity_start: { step: "proximity", status: "running" },
  proximity_complete: { step: "proximity", status: "done" },
};

const state = {
  running: false,
  eventSource: null,
  stepMap: new Map(),
  logs: [],
};

const dom = {
  runForm: document.getElementById("runForm"),
  researchGoal: document.getElementById("researchGoal"),
  citedPapers: document.getElementById("citedPapers"),
  maxIterations: document.getElementById("maxIterations"),
  initialCount: document.getElementById("initialCount"),
  evolutionCount: document.getElementById("evolutionCount"),
  modelName: document.getElementById("modelName"),
  enableLitReview: document.getElementById("enableLitReview"),
  enableToolCalling: document.getElementById("enableToolCalling"),
  runMode: document.getElementById("runMode"),
  apiBase: document.getElementById("apiBase"),
  apiBaseField: document.getElementById("apiBaseField"),
  goalCount: document.getElementById("goalCount"),
  statusPill: document.getElementById("statusPill"),
  modePill: document.getElementById("modePill"),
  statusText: document.getElementById("statusText"),
  progressBar: document.getElementById("progressBar"),
  logList: document.getElementById("logList"),
  logCount: document.getElementById("logCount"),
  stepsList: document.getElementById("stepsList"),
  stepCount: document.getElementById("stepCount"),
  finalList: document.getElementById("finalList"),
  finalCount: document.getElementById("finalCount"),
  runButton: document.getElementById("runButton"),
  clearButton: document.getElementById("clearButton"),
};

function init() {
  initSteps();
  updateGoalCount();
  updateModePill();
  dom.researchGoal.addEventListener("input", updateGoalCount);
  dom.runMode.addEventListener("change", updateModePill);
  dom.runForm.addEventListener("submit", handleRun);
  dom.clearButton.addEventListener("click", resetUI);
  document.addEventListener("click", handleCopyClick);
}

function updateGoalCount() {
  dom.goalCount.textContent = `${dom.researchGoal.value.length} characters`;
}

function updateModePill() {
  const isLive = dom.runMode.value === "api";
  dom.modePill.textContent = isLive ? "Live mode" : "Demo mode";
  if (dom.apiBaseField) {
    dom.apiBaseField.style.display = isLive ? "block" : "none";
  }
}

function initSteps() {
  dom.stepsList.innerHTML = "";
  state.stepMap.clear();
  steps.forEach((step, index) => {
    const card = document.createElement("article");
    card.className = "step-card";
    card.dataset.step = step.key;
    card.dataset.status = "pending";
    card.style.setProperty("--step-index", index);
    card.innerHTML = `
      <div class="step-header">
        <div>
          <p class="step-name">${step.label}</p>
          <span class="step-key">${step.key.replace(/_/g, " ")}</span>
        </div>
        <span class="step-status">Pending</span>
      </div>
      <div class="step-output"><div class="step-placeholder">No output yet.</div></div>
    `;
    dom.stepsList.appendChild(card);
    state.stepMap.set(step.key, {
      card,
      statusEl: card.querySelector(".step-status"),
      outputEl: card.querySelector(".step-output"),
      status: "pending",
    });
  });
  updateStepCount();
}

function updateStepCount() {
  const total = steps.length;
  const done = Array.from(state.stepMap.values()).filter((item) => item.status === "done").length;
  dom.stepCount.textContent = `${done} of ${total} complete`;
}

function setStepStatus(stepKey, status, label) {
  const item = state.stepMap.get(stepKey);
  if (!item) {
    return;
  }
  item.status = status;
  item.card.dataset.status = status;
  item.statusEl.textContent = label || status.charAt(0).toUpperCase() + status.slice(1);
  updateStepCount();
}

function setStepOutput(stepKey, html) {
  const item = state.stepMap.get(stepKey);
  if (!item) {
    return;
  }
  item.outputEl.innerHTML = html;
}

function setStatus(text, tone) {
  dom.statusPill.textContent = text;
  dom.statusPill.style.background = tone === "error" ? "#fee2e2" : "#ffffff";
  dom.statusPill.style.borderColor = tone === "error" ? "#fca5a5" : "var(--line)";
  dom.statusText.textContent = text;
}

function setProgress(value) {
  const safe = Math.max(0, Math.min(100, value));
  dom.progressBar.style.width = `${safe}%`;
}

function addLog(phase, message) {
  const time = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `
    <span class="log-time">${time}</span>
    <span><strong>${escapeHtml(phase)}</strong> ${escapeHtml(message)}</span>
  `;
  if (dom.logList.querySelector(".log-empty")) {
    dom.logList.innerHTML = "";
  }
  dom.logList.appendChild(line);
  dom.logList.scrollTop = dom.logList.scrollHeight;
  state.logs.push({ phase, message });
  dom.logCount.textContent = state.logs.length;
}

function resetUI() {
  state.running = false;
  cleanupStream();
  state.logs = [];
  dom.logList.innerHTML = "<div class=\"log-empty\">Logs will appear here once a run starts.</div>";
  dom.logCount.textContent = "0";
  setStatus("Idle", "idle");
  setProgress(0);
  updateFinalResults([]);
  initSteps();
}

function cleanupStream() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

function buildPayload() {
  const papers = dom.citedPapers.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return {
    research_goal: dom.researchGoal.value.trim(),
    model_name: dom.modelName.value.trim() || "claude-haiku-4-5-20251001",
    max_iterations: parseInt(dom.maxIterations.value, 10),
    initial_hypotheses_count: parseInt(dom.initialCount.value, 10),
    evolution_max_count: parseInt(dom.evolutionCount.value, 10),
    enable_literature_review_node: dom.enableLitReview.checked,
    enable_tool_calling_generation: dom.enableToolCalling.checked,
    cited_papers: papers,
  };
}

async function handleRun(event) {
  event.preventDefault();
  if (state.running) {
    return;
  }
  const payload = buildPayload();
  if (!payload.research_goal) {
    setStatus("Research goal is required", "error");
    return;
  }
  resetUI();
  state.running = true;
  setStatus("Running", "running");
  addLog("startup", "Starting workflow run");
  if (dom.runMode.value === "demo") {
    runDemo();
  } else {
    await runApi(payload);
  }
}

async function runApi(payload) {
  const base = dom.apiBase.value.trim();
  const baseUrl = base || window.location.origin;
  try {
    const response = await fetch(`${baseUrl}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`Run failed (${response.status})`);
    }
    const data = await response.json();
    if (!data.run_id) {
      throw new Error("Missing run_id from server");
    }
    connectStream(baseUrl, data.run_id);
  } catch (error) {
    setStatus("Error", "error");
    addLog("error", error.message || "Unable to start run");
    state.running = false;
  }
}

function connectStream(baseUrl, runId) {
  cleanupStream();
  const streamUrl = `${baseUrl}/api/stream/${runId}`;
  const source = new EventSource(streamUrl);
  state.eventSource = source;

  source.addEventListener("log", (event) => {
    const payload = JSON.parse(event.data);
    handleLogEvent(payload);
  });

  source.addEventListener("step", (event) => {
    const payload = JSON.parse(event.data);
    handleStepEvent(payload);
  });

  source.addEventListener("done", (event) => {
    const payload = JSON.parse(event.data);
    handleDoneEvent(payload);
  });

  source.addEventListener("error", (event) => {
    if (event.data) {
      const payload = JSON.parse(event.data);
      handleErrorEvent(payload);
    } else {
      handleErrorEvent({ message: "Stream error" });
    }
  });
}

function handleLogEvent(payload) {
  const phase = payload.phase || "log";
  const data = payload.data || {};
  const message = data.message || phase;
  addLog(phase, message);
  if (typeof data.progress === "number") {
    setProgress(data.progress);
  }
  const mapping = phaseToStep[phase];
  if (mapping) {
    setStepStatus(mapping.step, mapping.status);
  }
}

function handleStepEvent(payload) {
  const node = payload.node;
  const stateData = payload.state || {};
  if (node) {
    setStepStatus(node, "done");
    const percent = ((steps.findIndex((step) => step.key === node) + 1) / steps.length) * 100;
    if (!Number.isNaN(percent)) {
      setProgress(percent);
    }
    setStepOutput(node, formatOutputForStep(node, stateData));
    if (node === "ranking" || node === "proximity") {
      updateFinalResults(stateData.hypotheses || []);
    }
  }
}

function handleDoneEvent(payload) {
  addLog("complete", "Run complete");
  setStatus("Completed", "done");
  setProgress(100);
  state.running = false;
  cleanupStream();
  if (payload.state) {
    setStepOutput("ranking", formatOutputForStep("ranking", payload.state));
    setStepOutput("meta_review", formatOutputForStep("meta_review", payload.state));
    updateFinalResults(payload.state.hypotheses || []);
  }
}

function handleErrorEvent(payload) {
  addLog("error", payload.message || "Run failed");
  setStatus("Error", "error");
  state.running = false;
  cleanupStream();
}

function runDemo() {
  const demo = buildDemoStates();
  const timeline = [
    { delay: 300, type: "log", data: { phase: "supervisor_start", data: { message: "Analyzing research goal", progress: 8 } } },
    { delay: 1200, type: "step", data: { node: "supervisor", state: demo.supervisor } },
    { delay: 1500, type: "log", data: { phase: "literature_review_start", data: { message: "Scanning literature sources", progress: 18 } } },
    { delay: 2400, type: "step", data: { node: "literature_review", state: demo.literature_review } },
    { delay: 2700, type: "log", data: { phase: "generation_start", data: { message: "Generating initial hypotheses", progress: 32 } } },
    { delay: 3900, type: "step", data: { node: "generate", state: demo.generate } },
    { delay: 4200, type: "log", data: { phase: "reflection_start", data: { message: "Comparing hypotheses to literature", progress: 44 } } },
    { delay: 5200, type: "step", data: { node: "reflection", state: demo.reflection } },
    { delay: 5600, type: "log", data: { phase: "review_start", data: { message: "Running peer review", progress: 58 } } },
    { delay: 6700, type: "step", data: { node: "review", state: demo.review } },
    { delay: 7100, type: "log", data: { phase: "tournament_start", data: { message: "Ranking with tournament", progress: 72 } } },
    { delay: 8300, type: "step", data: { node: "ranking", state: demo.ranking } },
    { delay: 8700, type: "log", data: { phase: "meta_review_start", data: { message: "Synthesizing meta review", progress: 84 } } },
    { delay: 9500, type: "step", data: { node: "meta_review", state: demo.meta_review } },
    { delay: 9900, type: "log", data: { phase: "evolve_start", data: { message: "Refining top hypotheses", progress: 92 } } },
    { delay: 11000, type: "step", data: { node: "evolve", state: demo.evolve } },
    { delay: 11600, type: "log", data: { phase: "proximity_start", data: { message: "Checking for duplicates", progress: 97 } } },
    { delay: 12600, type: "step", data: { node: "proximity", state: demo.proximity } },
    { delay: 13200, type: "done", data: { state: demo.proximity } },
  ];

  timeline.forEach((event) => {
    setTimeout(() => {
      if (!state.running) {
        return;
      }
      if (event.type === "log") {
        handleLogEvent(event.data);
      } else if (event.type === "step") {
        handleStepEvent(event.data);
      } else if (event.type === "done") {
        handleDoneEvent(event.data);
      }
    }, event.delay);
  });
}

function updateFinalResults(hypotheses) {
  if (!dom.finalList || !dom.finalCount) {
    return;
  }
  if (!Array.isArray(hypotheses) || hypotheses.length === 0) {
    dom.finalList.innerHTML = "<div class=\"final-empty\">No final hypotheses yet.</div>";
    dom.finalCount.textContent = "0";
    return;
  }
  const sorted = [...hypotheses].sort((a, b) => (b.elo_rating || 0) - (a.elo_rating || 0));
  dom.finalList.innerHTML = sorted
    .map((hyp, index) => buildFinalCard(hyp, index))
    .join("");
  dom.finalCount.textContent = sorted.length;
}

function buildDemoStates() {
  const researchPlan = {
    research_goal_analysis: {
      goal_summary: "Identify early biomarkers for Parkinsons disease using multi-omics and imaging signals.",
      key_areas: ["omics signatures", "imaging phenotypes", "longitudinal cohorts"],
      constraints_identified: ["small sample sizes", "medication confounders"],
      success_criteria: ["replicable markers", "actionable experiments", "clear validation path"],
    },
    workflow_plan: {
      generation_phase: {
        focus_areas: ["blood markers", "sleep metrics", "neuroimaging"],
        diversity_targets: "cover distinct modalities and stages",
        quantity_target: "5 to 7 hypotheses",
      },
      review_phase: {
        critical_criteria: ["biological plausibility", "testability", "novelty"],
        review_depth: "deep review with cross-checks",
      },
      ranking_phase: {
        ranking_approach: "balance novelty with feasibility",
        selection_criteria: ["impact", "validation readiness", "data availability"],
      },
      evolution_phase: {
        refinement_priorities: ["reduce confounders", "clarify metrics"],
        iteration_strategy: "refine top hypotheses in one cycle",
      },
    },
    performance_assessment: {
      current_status: "ready to generate",
      bottlenecks_identified: ["limited cohort diversity"],
      agent_performance: {
        generation_agent: "ready",
        reflection_agent: "pending",
        ranking_agent: "pending",
        evolution_agent: "pending",
        proximity_agent: "pending",
        meta_review_agent: "pending",
      },
    },
    adjustment_recommendations: [
      {
        aspect: "data sources",
        adjustment: "add two cohorts",
        justification: "improve generalizability",
      },
    ],
    output_preparation: {
      hypothesis_selection_strategy: "prioritize high testability",
      presentation_format: "bullet summary with experiments",
      key_insights_to_highlight: ["novel markers", "validation path", "expected effect size"],
    },
  };

  const baseHypotheses = [
    {
      text: "A plasma lipid ratio shift in the prodromal phase predicts Parkinsons onset 18 months prior to motor symptoms.",
      explanation: "Early lipid metabolism changes may precede neuronal decline and can be measured in blood.",
      literature_grounding: "C1, C3",
      experiment: "Track lipid panels in a longitudinal cohort and compare with matched controls.",
      score: 0.0,
      elo_rating: 1200,
      reviews: [],
      reflection_notes: "",
    },
    {
      text: "REM sleep fragmentation plus subtle substantia nigra signal drift predicts disease conversion in at-risk patients.",
      explanation: "Sleep pattern disruptions may correlate with early neurodegeneration.",
      literature_grounding: "C2",
      experiment: "Combine sleep wearables with MRI follow-ups every six months.",
      score: 0.0,
      elo_rating: 1200,
      reviews: [],
      reflection_notes: "",
    },
    {
      text: "Gut microbiome diversity loss interacts with inflammatory markers to accelerate alpha-synuclein aggregation.",
      explanation: "Microbiome shifts can influence systemic inflammation and protein aggregation pathways.",
      literature_grounding: "C4, C5",
      experiment: "Measure microbiome profiles alongside inflammatory panels in a new cohort.",
      score: 0.0,
      elo_rating: 1200,
      reviews: [],
      reflection_notes: "",
    },
  ];

  const reflectionHypotheses = baseHypotheses.map((hyp, index) => ({
    ...hyp,
    reflection_notes: `Classification: ${index === 1 ? "missing piece" : "neutral"}. Reasoning: ${index === 2 ? "Microbiome signals appear under-explored." : "Evidence is suggestive but not definitive."}`,
  }));

  const reviewedHypotheses = reflectionHypotheses.map((hyp, index) => ({
    ...hyp,
    score: 7.8 - index * 0.6,
    reviews: [
      {
        review_summary: index === 0 ? "Strong biomarker signal with clear validation path." : "Promising but needs tighter cohort design.",
        scores: { novelty: 8 - index, feasibility: 7 - index, impact: 8 - index * 0.5 },
        overall_score: 7.8 - index * 0.6,
      },
    ],
  }));

  const rankedHypotheses = reviewedHypotheses.map((hyp, index) => ({
    ...hyp,
    elo_rating: 1260 - index * 30,
  }));

  return {
    supervisor: {
      research_plan: researchPlan,
      hypotheses: [],
      meta_review: {},
      tournament_matchups: [],
      evolution_details: [],
      similarity_clusters: [],
    },
    literature_review: {
      research_plan: researchPlan,
      articles_with_reasoning: "Key themes: lipid metabolism, sleep disruption, gut inflammation. Gaps: limited multi-modal validation.",
      hypotheses: [],
      meta_review: {},
      tournament_matchups: [],
      evolution_details: [],
      similarity_clusters: [],
    },
    generate: {
      research_plan: researchPlan,
      articles_with_reasoning: "Key themes: lipid metabolism, sleep disruption, gut inflammation.",
      hypotheses: baseHypotheses,
      meta_review: {},
      tournament_matchups: [],
      evolution_details: [],
      similarity_clusters: [],
    },
    reflection: {
      research_plan: researchPlan,
      articles_with_reasoning: "Key themes: lipid metabolism, sleep disruption, gut inflammation.",
      hypotheses: reflectionHypotheses,
      meta_review: {},
      tournament_matchups: [],
      evolution_details: [],
      similarity_clusters: [],
    },
    review: {
      research_plan: researchPlan,
      articles_with_reasoning: "Key themes: lipid metabolism, sleep disruption, gut inflammation.",
      hypotheses: reviewedHypotheses,
      meta_review: {},
      tournament_matchups: [],
      evolution_details: [],
      similarity_clusters: [],
    },
    ranking: {
      research_plan: researchPlan,
      articles_with_reasoning: "Key themes: lipid metabolism, sleep disruption, gut inflammation.",
      hypotheses: rankedHypotheses,
      meta_review: {},
      tournament_matchups: [
        {
          hypothesis_a: rankedHypotheses[0].text,
          hypothesis_b: rankedHypotheses[1].text,
          winner: "a",
          reasoning: "Hypothesis A has stronger validation path",
          winner_elo_before: 1200,
          winner_elo_after: 1260,
          loser_elo_before: 1200,
          loser_elo_after: 1180,
        },
      ],
      evolution_details: [],
      similarity_clusters: [],
    },
    meta_review: {
      research_plan: researchPlan,
      hypotheses: rankedHypotheses,
      meta_review: {
        summary: "The strongest hypotheses leverage measurable biomarkers with clear validation pathways.",
        common_strengths: ["clear experimental plan", "multi-modal data usage"],
        common_weaknesses: ["cohort diversity", "confounding variables"],
        strategic_recommendations: ["expand cohorts", "prioritize longitudinal signals"],
      },
      tournament_matchups: [],
      evolution_details: [],
      similarity_clusters: [],
    },
    evolve: {
      research_plan: researchPlan,
      hypotheses: rankedHypotheses,
      meta_review: {
        summary: "Refined top hypotheses to reduce confounders.",
      },
      tournament_matchups: [],
      evolution_details: [
        {
          original: rankedHypotheses[0].text,
          evolved: "Integrate lipid ratio shifts with inflammatory markers to reduce medication bias.",
          rationale: "Combines metabolic signal with a control marker for confounding.",
        },
      ],
      similarity_clusters: [],
    },
    proximity: {
      research_plan: researchPlan,
      hypotheses: rankedHypotheses,
      meta_review: {
        summary: "No major duplicates detected.",
      },
      tournament_matchups: [],
      evolution_details: [],
      similarity_clusters: [
        {
          cluster_id: "cluster-1",
          cluster_name: "Metabolic biomarkers",
          central_theme: "lipid shifts",
          similar_hypotheses: [rankedHypotheses[0].text],
        },
      ],
    },
  };
}

function formatOutputForStep(stepKey, stateData) {
  switch (stepKey) {
    case "supervisor":
      return formatSupervisorOutput(stateData.research_plan);
    case "literature_review":
      return formatLiteratureOutput(stateData);
    case "generate":
      return formatHypothesesOutput(stateData.hypotheses, { showDetails: true });
    case "reflection":
      return formatReflectionOutput(stateData.hypotheses);
    case "review":
      return formatReviewOutput(stateData.hypotheses);
    case "ranking":
      return formatRankingOutput(stateData.hypotheses, stateData.tournament_matchups || []);
    case "meta_review":
      return formatMetaReviewOutput(stateData.meta_review || {});
    case "evolve":
      return formatEvolutionOutput(stateData.evolution_details || []);
    case "proximity":
      return formatProximityOutput(stateData.similarity_clusters || []);
    default:
      return "<div class=\"step-placeholder\">No output yet.</div>";
  }
}

function formatSupervisorOutput(plan) {
  if (!plan || Object.keys(plan).length === 0) {
    return "<div class=\"step-placeholder\">No supervisor plan yet.</div>";
  }
  const analysis = plan.research_goal_analysis || {};
  const workflow = plan.workflow_plan || {};
  const performance = plan.performance_assessment || {};
  const adjustments = plan.adjustment_recommendations || [];
  const prep = plan.output_preparation || {};

  const content = `
    <div>
      <strong>Goal summary</strong>
      <div>${escapeHtml(analysis.goal_summary || "")}</div>
      ${formatList("Key areas", analysis.key_areas)}
      ${formatList("Constraints", analysis.constraints_identified)}
      ${formatList("Success criteria", analysis.success_criteria)}
      <strong>Workflow plan</strong>
      ${formatSubSection("Generation", workflow.generation_phase)}
      ${formatSubSection("Review", workflow.review_phase)}
      ${formatSubSection("Ranking", workflow.ranking_phase)}
      ${formatSubSection("Evolution", workflow.evolution_phase)}
      <strong>Performance</strong>
      <div>${escapeHtml(performance.current_status || "")}</div>
      ${formatList("Bottlenecks", performance.bottlenecks_identified)}
      ${formatList("Adjustments", adjustments.map((adj) => `${adj.aspect}: ${adj.adjustment}`))}
      <strong>Output preparation</strong>
      <div>${escapeHtml(prep.hypothesis_selection_strategy || "")}</div>
      ${formatList("Key insights", prep.key_insights_to_highlight)}
    </div>
  `;
  return wrapCopyBlock(content);
}

function formatLiteratureOutput(stateData) {
  const text = stateData.articles_with_reasoning;
  if (!text || text === "__LIT_REVIEW_FAILED__") {
    return wrapCopyBlock(`<div class="step-placeholder" style="color:#6b7280;font-style:italic;">No relevant most recent literature found. The workflow will continue with hypothesis generation based on the model's existing knowledge.</div>`);
  }
  return wrapCopyBlock(`<pre>${escapeHtml(text)}</pre>`);
}

function formatHypothesesOutput(hypotheses, options = {}) {
  if (!Array.isArray(hypotheses) || hypotheses.length === 0) {
    return "<div class=\"step-placeholder\">No hypotheses yet.</div>";
  }
  return hypotheses
    .map((hyp, index) => {
      const details = options.showDetails
        ? buildHypothesisDetails(hyp)
        : "";
      return buildHypothesisCard(hyp, index, details);
    })
    .join("");
}

function formatReflectionOutput(hypotheses) {
  if (!Array.isArray(hypotheses) || hypotheses.length === 0) {
    return "<div class=\"step-placeholder\">No reflection notes yet.</div>";
  }
  return hypotheses
    .map((hyp, index) => {
      const details = `<div><strong>Reflection</strong> ${escapeHtml(hyp.reflection_notes || "Not available")}</div>`;
      return buildHypothesisCard(hyp, index, details);
    })
    .join("");
}

function formatReviewOutput(hypotheses) {
  if (!Array.isArray(hypotheses) || hypotheses.length === 0) {
    return "<div class=\"step-placeholder\">No reviews yet.</div>";
  }
  return hypotheses
    .map((hyp, index) => {
      const review = hyp.reviews && hyp.reviews.length > 0 ? hyp.reviews[hyp.reviews.length - 1] : null;
      const details = `
        <div><strong>Summary</strong> ${escapeHtml(review ? review.review_summary : "Review pending")}</div>
        ${review ? formatScoreRow(review.scores || {}) : ""}
      `;
      return buildHypothesisCard(hyp, index, details);
    })
    .join("");
}

function formatRankingOutput(hypotheses, matchups) {
  if (!Array.isArray(hypotheses) || hypotheses.length === 0) {
    return "<div class=\"step-placeholder\">No rankings yet.</div>";
  }
  const sorted = [...hypotheses].sort((a, b) => (b.elo_rating || 0) - (a.elo_rating || 0));
  const rankingList = sorted
    .slice(0, 5)
    .map((hyp, index) => {
      return `<div><strong>#${index + 1}</strong> ${escapeHtml(hyp.text || "")} (elo ${hyp.elo_rating || 0})</div>`;
    })
    .join("");

  const matchupList = matchups.length
    ? matchups
        .slice(0, 2)
        .map((match, index) => {
          return `<div><strong>Matchup ${index + 1}</strong> ${escapeHtml(match.reasoning || "")}</div>`;
        })
        .join("")
    : "<div>No matchups captured yet.</div>";

  const content = `
    <div>
      <strong>Top rankings</strong>
      ${rankingList}
      <strong>Tournament highlights</strong>
      ${matchupList}
    </div>
  `;
  return wrapCopyBlock(content);
}

function formatMetaReviewOutput(metaReview) {
  if (!metaReview || Object.keys(metaReview).length === 0) {
    return "<div class=\"step-placeholder\">No meta review yet.</div>";
  }
  const summary = metaReview.summary || metaReview.meta_review_summary || metaReview.synthesis_summary || "";
  const content = `
    <div>
      <strong>Summary</strong>
      <div>${escapeHtml(summary)}</div>
      ${formatList("Common strengths", metaReview.common_strengths || metaReview.strengths)}
      ${formatList("Common weaknesses", metaReview.common_weaknesses || metaReview.weaknesses)}
      ${formatList("Recommendations", metaReview.strategic_recommendations)}
    </div>
  `;
  return wrapCopyBlock(content);
}

function formatEvolutionOutput(details) {
  if (!Array.isArray(details) || details.length === 0) {
    return "<div class=\"step-placeholder\">No evolution changes yet.</div>";
  }
  return details
    .map((detail, index) => {
      return `
        <div class="hypothesis-card copy-wrapper">
          <div class="hypothesis-header">
            <h4>Evolution ${index + 1}</h4>
            <button class="copy-button" type="button">Copy</button>
          </div>
          <div class="copy-content">
            <div><strong>Original</strong> ${escapeHtml(detail.original || "")}</div>
            <div><strong>Evolved</strong> ${escapeHtml(detail.evolved || "")}</div>
            <div><strong>Rationale</strong> ${escapeHtml(detail.rationale || "")}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

function formatProximityOutput(clusters) {
  if (!Array.isArray(clusters) || clusters.length === 0) {
    return "<div class=\"step-placeholder\">No proximity clusters reported.</div>";
  }
  return clusters
    .map((cluster) => {
      return `
        <div class="hypothesis-card copy-wrapper">
          <div class="hypothesis-header">
            <h4>${escapeHtml(cluster.cluster_name || "Cluster")}</h4>
            <button class="copy-button" type="button">Copy</button>
          </div>
          <div class="copy-content">
            <div>${escapeHtml(cluster.central_theme || "")}</div>
            ${formatList("Similar hypotheses", cluster.similar_hypotheses)}
          </div>
        </div>
      `;
    })
    .join("");
}

function buildHypothesisCard(hyp, index, detailsHtml) {
  return `
    <div class="hypothesis-card copy-wrapper">
      <div class="hypothesis-header">
        <h4>Hypothesis ${index + 1}</h4>
        <button class="copy-button" type="button">Copy</button>
      </div>
      <div class="copy-content">
        <div>${escapeHtml(hyp.text || "")}</div>
        ${detailsHtml || ""}
      </div>
    </div>
  `;
}

function buildHypothesisDetails(hyp) {
  return `
    ${hyp.explanation ? `<div><strong>Explanation</strong> ${escapeHtml(hyp.explanation)}</div>` : ""}
    ${hyp.experiment ? `<div><strong>Experiment</strong> ${escapeHtml(hyp.experiment)}</div>` : ""}
    ${hyp.literature_grounding ? `<div><strong>Literature grounding</strong> ${escapeHtml(hyp.literature_grounding)}</div>` : ""}
  `;
}

function buildFinalCard(hyp, index) {
  const stats = `
    <div class="score-row">
      <span class="score-chip">score: ${escapeHtml(formatNumber(hyp.score))}</span>
      <span class="score-chip">elo: ${escapeHtml(formatNumber(hyp.elo_rating))}</span>
      <span class="score-chip">win rate: ${escapeHtml(formatNumber(hyp.win_rate))}%</span>
    </div>
  `;
  const details = `
    ${buildHypothesisDetails(hyp)}
    ${stats}
  `;
  return buildHypothesisCard(hyp, index, details);
}

function wrapCopyBlock(innerHtml) {
  return `
    <div class="copy-wrapper">
      <button class="copy-button" type="button">Copy</button>
      <div class="copy-content">${innerHtml}</div>
    </div>
  `;
}

function handleCopyClick(event) {
  const button = event.target.closest(".copy-button");
  if (!button) {
    return;
  }
  const wrapper = button.closest(".copy-wrapper");
  if (!wrapper) {
    return;
  }
  const content = wrapper.querySelector(".copy-content");
  if (!content) {
    return;
  }
  const text = content.innerText.trim();
  if (!text) {
    return;
  }
  copyToClipboard(text)
    .then(() => {
      const original = button.textContent;
      button.textContent = "Copied";
      button.classList.add("copied");
      setTimeout(() => {
        button.textContent = original;
        button.classList.remove("copied");
      }, 1200);
    })
    .catch(() => {
      button.textContent = "Copy failed";
      setTimeout(() => {
        button.textContent = "Copy";
      }, 1200);
    });
}

async function copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "0";
  }
  if (typeof value === "number") {
    return value.toFixed(1).replace(/\.0$/, "");
  }
  return String(value);
}

function formatSubSection(title, section) {
  if (!section) {
    return "";
  }
  const content = Object.entries(section)
    .map(([key, value]) => {
      if (Array.isArray(value)) {
        return `<div><strong>${escapeHtml(labelize(key))}</strong> ${escapeHtml(value.join(", "))}</div>`;
      }
      return `<div><strong>${escapeHtml(labelize(key))}</strong> ${escapeHtml(String(value))}</div>`;
    })
    .join("");
  return `<div><strong>${escapeHtml(title)}</strong>${content}</div>`;
}

function formatList(title, items) {
  if (!Array.isArray(items) || items.length === 0) {
    return "";
  }
  const list = items.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("");
  return `<div><strong>${escapeHtml(title)}</strong><ul>${list}</ul></div>`;
}

function formatScoreRow(scores) {
  const entries = Object.entries(scores);
  if (!entries.length) {
    return "";
  }
  return `
    <div class="score-row">
      ${entries
        .map(([key, value]) => `<span class="score-chip">${escapeHtml(key)}: ${escapeHtml(String(value))}</span>`)
        .join("")}
    </div>
  `;
}

function labelize(text) {
  return text.replace(/_/g, " ");
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

init();
