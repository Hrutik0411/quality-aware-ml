import { useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  CircleAlert,
  Database,
  FileSpreadsheet,
  FlaskConical,
  Gauge,
  LoaderCircle,
  MessageCircle,
  Send,
  ShieldCheck,
  Trophy,
  Upload,
  Workflow,
  Play,
} from "lucide-react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./App.css";


const API_URL = import.meta.env.VITE_API_URL || "https://quality-aware-ml.onrender.com";

const CLASSIFICATION_METRICS = [
  { label: "Accuracy", key: "Accuracy", cvKey: "accuracy", higher: true },
  {
    label: "Macro Precision",
    key: "Precision_macro",
    cvKey: "precision_macro",
    higher: true,
  },
  {
    label: "Macro Recall",
    key: "Recall_macro",
    cvKey: "recall_macro",
    higher: true,
  },
  {
    label: "Macro F1",
    key: "F1_macro",
    cvKey: "f1_macro",
    higher: true,
  },
];

const REGRESSION_METRICS = [
  { label: "RMSE", key: "RMSE", cvKey: "rmse", higher: false },
  { label: "MAE", key: "MAE", cvKey: "mae", higher: false },
  { label: "R² Score", key: "R2", cvKey: "r2", higher: true },
];

const NAV_ITEMS = [
  { label: "Overview", icon: Gauge, id: "overview" },
  { label: "Dataset analysis", icon: Database, id: "dataset-analysis" },
  { label: "Agent workflow", icon: Workflow, id: "agent-workflow" },
  { label: "Experiments", icon: FlaskConical, id: "experiments" },
  { label: "Verification", icon: ShieldCheck, id: "verification" },
];

const AGENTS = [
  {
    key: "dataset_profiler",
    title: "Dataset Profiler",
    stage: "Dataset profiling",
    description:
      "Understand dataset dimensions, columns, data types, and statistical characteristics.",
    icon: Database,
    reportField: "profile",
    suggestions: [
      "Explain the dataset profile.",
      "What do the columns tell us?",
    ],
  },
  {
    key: "quality_analyst",
    title: "Quality Analyst",
    stage: "Data quality",
    description:
      "Understand missing values, duplicates, outliers, and other data-quality findings.",
    icon: ShieldCheck,
    reportField: "quality_report",
    suggestions: [
      "Explain the quality issues.",
      "Which data-quality concerns matter most?",
    ],
  },
  {
    key: "impact_analyst",
    title: "Impact Analyst",
    stage: "Impact analysis",
    description:
      "Explore how quality issues may affect model training and evaluation.",
    icon: Activity,
    reportField: "impact_report",
    suggestions: [
      "Explain the impact assessment.",
      "How could these issues affect predictions?",
    ],
  },
  {
    key: "pipeline_planner",
    title: "Pipeline Planner",
    stage: "Pipeline planning",
    description:
      "Understand the pipeline, prediction target, and evaluation approach.",
    icon: Workflow,
    reportField: "pipeline_plan",
    suggestions: [
      "Explain the planned pipeline.",
      "Why was this problem type selected?",
    ],
  },
  {
    key: "experiment_agent",
    title: "Experiment Agent",
    stage: "Model experiments",
    description:
      "Understand model comparisons, performance metrics, and cross-validation results.",
    icon: FlaskConical,
    reportField: "experiment_report",
    suggestions: [
      "Compare the model results.",
      "Explain the evaluation metrics.",
    ],
  },
  {
    key: "critic_agent",
    title: "Critic Agent",
    stage: "Verification",
    description:
      "Ask about verification checks, evidence, and experiment limitations.",
    icon: ShieldCheck,
    reportField: "verification_report",
    suggestions: [
      "Explain the verification result.",
      "What limitations should I consider?",
    ],
  },
  {
    key: "feedback_agent",
    title: "Feedback Agent",
    stage: "Feedback",
    description:
      "Understand the recommended next action and the reasoning behind it.",
    icon: BrainCircuit,
    reportField: "feedback_report",
    suggestions: [
      "Explain the recommended next action.",
      "What should I improve next?",
    ],
  },
];

function App() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");
  const [columns, setColumns] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedMetric, setSelectedMetric] = useState("F1_macro");
  const [activeNav, setActiveNav] = useState("overview");

  const scrollToSection = (id) => {
    setActiveNav(id);

    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  const handleFileChange = async (event) => {
    const selectedFile = event.target.files?.[0];

    setResult(null);
    setError("");
    setColumns([]);
    setTarget("");
    setFile(null);

    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a CSV file.");
      return;
    }

    try {
      const text = await selectedFile.text();
      const firstLine = text.split(/\r?\n/, 1)[0];

      if (!firstLine?.trim()) {
        setError("The selected CSV file is empty.");
        return;
      }

      const headers = [];
      let current = "";
      let insideQuotes = false;

      for (let i = 0; i < firstLine.length; i++) {
        const char = firstLine[i];

        if (
          char === '"' &&
          firstLine[i + 1] === '"' &&
          insideQuotes
        ) {
          current += '"';
          i++;
        } else if (char === '"') {
          insideQuotes = !insideQuotes;
        } else if (char === "," && !insideQuotes) {
          headers.push(current.trim());
          current = "";
        } else {
          current += char;
        }
      }

      headers.push(current.trim());

      const cleanedHeaders = headers.map((header, index) =>
        index === 0 ? header.replace(/^\uFEFF/, "") : header
      );

      if (cleanedHeaders.length < 2) {
        setError("The CSV must contain at least two columns.");
        return;
      }

      setFile(selectedFile);
      setColumns(cleanedHeaders);
      setTarget(cleanedHeaders[cleanedHeaders.length - 1]);
    } catch {
      setError("Unable to read this CSV file.");
    }
  };

  const handleAnalyze = async () => {
    if (!file || !target) {
      setError("Upload a CSV file and select its target column.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("target_column", target);

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Dataset analysis failed.");
      }

      setResult(data);
      setActiveNav("dataset-analysis");
    } catch (err) {
      setError(
        err.message ||
          "Cannot connect to FastAPI. Check that the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const dataset = result?.dataset;
  const profile = result?.profile;
  const quality = result?.quality_report;
  const impact = result?.impact_report;
  const plan = result?.pipeline_plan;
  const experiments = result?.experiment_report;
  const verification = result?.verification_report;
  const feedback = result?.feedback_report;

  const allResults = experiments?.results || [];

  const successfulResults = allResults.filter(
    (item) =>
      item.status === "success" &&
      item.metrics &&
      Object.keys(item.metrics).length > 0
  );

  const problemType = String(
    plan?.problem_type || experiments?.problem_type || ""
  ).toLowerCase();

  const isClassification = problemType.includes("class");

  const availableMetrics = isClassification
    ? CLASSIFICATION_METRICS
    : REGRESSION_METRICS;

  const activeMetric =
    availableMetrics.find((metric) => metric.key === selectedMetric) ||
    availableMetrics[0];

  const baselineResults = successfulResults.filter(
    (item) => item.experiment === "baseline"
  );

  const rankedModels = useMemo(() => {
    const bestByModel = new Map();

    for (const item of baselineResults) {
      const testValue = Number(item.metrics?.[activeMetric.key]);

      const cvMetric =
        item.cross_validation?.metrics?.[activeMetric.cvKey];

      const cvValue = Number(cvMetric?.mean);

      const cvIsValid =
        item.cross_validation?.status === "success" &&
        Number.isFinite(cvValue);

      const score = cvIsValid
        ? cvValue
        : Number.isFinite(testValue)
          ? testValue
          : null;

      if (score === null) continue;

      const entry = {
        model: item.model || "Unnamed model",
        score,
        source: cvIsValid
          ? "Cross-validation mean"
          : "Held-out test score",
        cv: cvIsValid ? cvMetric : null,
        testScore: Number.isFinite(testValue) ? testValue : null,
      };

      const previous = bestByModel.get(entry.model);

      if (
        !previous ||
        (activeMetric.higher
          ? entry.score > previous.score
          : entry.score < previous.score)
      ) {
        bestByModel.set(entry.model, entry);
      }
    }

    return [...bestByModel.values()].sort((a, b) =>
      activeMetric.higher
        ? b.score - a.score
        : a.score - b.score
    );
  }, [baselineResults, activeMetric]);

  const bestModel = rankedModels[0] || null;

  const chartData = useMemo(
    () =>
      baselineResults.flatMap((item) => {
        const value = Number(item.metrics?.[activeMetric.key]);

        if (!Number.isFinite(value)) return [];

        const cvMetric =
          item.cross_validation?.metrics?.[activeMetric.cvKey];

        const cvMean =
          item.cross_validation?.status === "success" &&
          Number.isFinite(Number(cvMetric?.mean))
            ? Number(cvMetric.mean)
            : null;

        return [
          {
            model: item.model || "Unnamed model",
            value,
            displayValue: cvMean ?? value,
            scoreSource: cvMean === null ? "Test score" : "CV mean",
          },
        ];
      }),
    [baselineResults, activeMetric]
  );

  const workflowSteps = [
    { name: "Dataset profiling", data: profile },
    { name: "Quality assessment", data: quality },
    { name: "Impact analysis", data: impact },
    { name: "Pipeline planning", data: plan },
    { name: "ML experiments", data: experiments },
    { name: "Critic verification", data: verification },
    { name: "Feedback decision", data: feedback },
  ];
    return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <BrainCircuit size={25} />
          </div>
          <div>
            <h2>QualiMind AI</h2>
            <p>Agentic ML Platform</p>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(({ label, icon: Icon, id }) => (
            <button
              key={id}
              type="button"
              className={`nav-item ${activeNav === id ? "active" : ""}`}
              onClick={() => scrollToSection(id)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
  <div className="status-dot" />
  <span>ML backend configured</span>
  <p>API: {API_URL}</p>
</div>
      </aside>

      <main className="main-content">
        <header className="topbar" id="overview">
          <div>
            <div className="eyebrow">AI / MACHINE LEARNING</div>
            <h1>Dataset Intelligence</h1>
            <p className="subtitle">
              Understand data quality. Evaluate models. Make evidence-based
              pipeline decisions.
            </p>
          </div>
          <div className="api-badge">
            <span className="status-dot" /> Local API
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <span className="hero-tag">AGENTIC AI FRAMEWORK</span>
            <h2>Turn raw datasets into reliable ML insights.</h2>
            <p>
              Upload a dataset and let your analysis agents assess its quality,
              plan experiments, compare models, and recommend the next action.
            </p>
            <div className="hero-points">
              <span><CheckCircle2 size={15} /> Quality-aware</span>
              <span><CheckCircle2 size={15} /> Experiment-driven</span>
              <span><CheckCircle2 size={15} /> Critic-verified</span>
            </div>
          </div>

          <div className="hero-art">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="brain-core"><BrainCircuit size={62} /></div>
            <div className="floating-node node-one"><Database size={19} /></div>
            <div className="floating-node node-two"><Activity size={19} /></div>
            <div className="floating-node node-three"><ShieldCheck size={19} /></div>
          </div>
        </section>

        <section className="section-heading" id="dataset-analysis">
          <div>
            <h2>Analyze a dataset</h2>
            <p>Choose a CSV file and its prediction target.</p>
          </div>

          {dataset && (
            <span className="success-pill">
              <CheckCircle2 size={15} /> Analysis completed
            </span>
          )}
        </section>

        <section className="upload-card">
          <label className="drop-zone">
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileChange}
            />
            <div className="upload-icon"><Upload size={27} /></div>
            <h3>{file ? file.name : "Upload your dataset"}</h3>
            <p>
              {file
                ? `${(file.size / 1024).toFixed(1)} KB · CSV file selected`
                : "Click here to browse your CSV dataset."}
            </p>
            <span className="browse-button">Choose CSV file</span>
          </label>

          <div className="upload-controls">
            <label htmlFor="target-select">Prediction target</label>
            <select
              id="target-select"
              value={target}
              onChange={(event) => setTarget(event.target.value)}
              disabled={!columns.length || loading}
            >
              {!columns.length && (
                <option value="">Upload a CSV first</option>
              )}

              {columns.map((column, index) => (
                <option key={`${column}-${index}`} value={column}>
                  {column}
                </option>
              ))}
            </select>

            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={!file || !target || loading}
            >
              {loading ? (
                <>
                  <LoaderCircle className="spin" size={18} />
                  Analyzing dataset...
                </>
              ) : (
                <>
                  <Play size={17} fill="currentColor" />
                  Run AI analysis
                </>
              )}
            </button>

            <p className="control-note">
              Your file is sent to the deployed FastAPI backend for analysis.
            </p>
          </div>
        </section>

        {error && (
          <div className="error-banner">
            <CircleAlert size={20} />
            <div>
              <strong>Something needs attention</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {result && (
          <>
            <section className="section-heading results-heading">
              <div>
                <h2>Analysis results</h2>
                <p>Results returned by your existing Python agents.</p>
              </div>
              <span className="neutral-pill">
                {dataset?.filename || file?.name}
              </span>
            </section>

            <section className="stats-grid">
              <StatCard
                icon={<Database size={20} />}
                label="Dataset rows"
                value={dataset?.rows?.toLocaleString()}
              />
              <StatCard
                icon={<FileSpreadsheet size={20} />}
                label="Features + target"
                value={dataset?.columns}
              />
              <StatCard
                icon={<FlaskConical size={20} />}
                label="Successful experiments"
                value={successfulResults.length}
              />
              <StatCard
                icon={<ShieldCheck size={20} />}
                label="Critic status"
                value={verification?.status || "Unavailable"}
                compact
              />
            </section>

            {/* Agent workflow and model performance */}
            <section className="results-grid" id="agent-workflow">
              <div className="panel">
                <div className="panel-heading">
                  <div>
                    <h2><Workflow size={19} /> Agent workflow</h2>
                    <p>Results available from each stage.</p>
                  </div>
                </div>

                <div className="workflow-list">
                  {workflowSteps.map((step, index) => (
                    <div className="workflow-step" key={step.name}>
                      <div className="step-number">
                        {step.data ? <CheckCircle2 size={17} /> : index + 1}
                      </div>
                      <div className="step-copy">
                        <strong>{step.name}</strong>
                        <span>
                          {step.data ? "Result received" : "No result available"}
                        </span>
                      </div>
                      <span className={step.data ? "step-done" : "step-pending"}>
                        {step.data ? "Ready" : "Pending"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel">
                <div className="panel-heading">
                  <div>
                    <h2><Activity size={19} /> Model performance</h2>
                    <p>Compare baseline models using the selected metric.</p>
                  </div>
                </div>

                <div className="chart-controls">
                  <label htmlFor="metric-select">Metric to compare</label>
                  <select
                    id="metric-select"
                    value={activeMetric.key}
                    onChange={(event) => setSelectedMetric(event.target.value)}
                  >
                    {availableMetrics.map((metric) => (
                      <option key={metric.key} value={metric.key}>
                        {metric.label}
                      </option>
                    ))}
                  </select>
                </div>

                {chartData.length ? (
                  <div className="chart-area">
                    <ResponsiveContainer width="100%" height={330}>
                      <BarChart
                        data={chartData}
                        margin={{ top: 12, right: 12, left: 0, bottom: 75 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey="model"
                          angle={-25}
                          textAnchor="end"
                          interval={0}
                          height={95}
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis
                          domain={isClassification ? [0, 1] : ["auto", "auto"]}
                          tick={{ fontSize: 11 }}
                        />
                        <Tooltip
                          formatter={(value, _, item) => [
                            Number(value).toFixed(4),
                            item.payload.scoreSource,
                          ]}
                        />
                        <Bar
                          dataKey="displayValue"
                          name={activeMetric.label}
                          radius={[5, 5, 0, 0]}
                        >
                          {chartData.map((item, index) => (
                            <Cell
                              key={`${item.model}-${index}`}
                              fill={
                                bestModel?.model === item.model
                                  ? "#16a085"
                                  : "#6d64ee"
                              }
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <EmptyState message="No successful baseline models have a numeric score for this metric." />
                )}

                <p className="chart-caption">
                  Green highlights the current top-ranked baseline model.
                </p>
              </div>
            </section>

            {/* 1. Dataset Profiler */}
            <section className="panel detail-panel">
              <div className="panel-heading">
                <div>
                  <h2><Database size={19} /> Dataset profile</h2>
                  <p>Dataset structure and statistical characteristics.</p>
                </div>
                <AgentChat agent={AGENTS[0]} result={result} />
              </div>

              <JsonSummary data={profile} />
            </section>

            {/* 2. Quality Analyst and 3. Impact Analyst */}
            <section className="results-grid">
              <div className="panel agent-report-panel">
                <div className="panel-heading">
                  <div>
                    <h2><ShieldCheck size={19} /> Quality assessment</h2>
                    <p>Findings from the quality analyst.</p>
                  </div>
                  <AgentChat agent={AGENTS[1]} result={result} />
                </div>
                <JsonSummary data={quality} />
              </div>

              <div className="panel agent-report-panel">
                <div className="panel-heading">
                  <div>
                    <h2><Activity size={19} /> Impact assessment</h2>
                    <p>Potential impacts on training and evaluation.</p>
                  </div>
                  <AgentChat agent={AGENTS[2]} result={result} />
                </div>
                <JsonSummary data={impact} />
              </div>
            </section>

            {/* 4. Pipeline Planner */}
            <section className="panel detail-panel">
              <div className="panel-heading">
                <div>
                  <h2><Gauge size={19} /> Pipeline plan</h2>
                  <p>Planning output from the pipeline planner.</p>
                </div>
                <AgentChat agent={AGENTS[3]} result={result} />
              </div>

              <div className="detail-grid">
                <Detail
                  label="Problem type"
                  value={plan?.problem_type || experiments?.problem_type}
                />
                <Detail label="Target column" value={plan?.target_column || target} />
                <Detail label="Detection confidence" value={plan?.detection_confidence} />
                <Detail label="Evaluation metrics" value={plan?.evaluation_metrics} />
              </div>

              {plan?.detection_reason && (
                <p className="reason-text">{plan.detection_reason}</p>
              )}
            </section>

            {/* 5. Experiment Agent */}
            <section className="panel detail-panel" id="experiments">
              <div className="panel-heading">
                <div>
                  <h2><FlaskConical size={19} /> Experiment details</h2>
                  <p>
                    Successful and failed model runs, including cross-validation.
                  </p>
                </div>
                <AgentChat agent={AGENTS[4]} result={result} />
              </div>

              {allResults.length ? (
                <div className="experiment-grid">
                  {allResults.map((item, index) => (
                    <div
                      className="experiment-card"
                      key={`${item.experiment}-${item.model}-${index}`}
                    >
                      <div className="experiment-title">
                        <strong>{item.model || "Unnamed model"}</strong>
                        <span>{item.experiment || "experiment"}</span>
                      </div>

                      <div className={`experiment-status status-${item.status || "unknown"}`}>
                        {item.status || "unknown"}
                      </div>

                      {item.status === "success" &&
                        item.metrics &&
                        Object.entries(item.metrics).map(([name, value]) => (
                          <div className="metric-row" key={name}>
                            <span>{name.replaceAll("_", " ")}</span>
                            <strong>
                              {typeof value === "number"
                                ? value.toFixed(4)
                                : String(value)}
                            </strong>
                          </div>
                        ))}

                      {item.cross_validation && (
                        <details className="cv-details">
                          <summary>Cross-validation results</summary>
                          <JsonSummary data={item.cross_validation} />
                        </details>
                      )}

                      {item.comparison && (
                        <details className="cv-details">
                          <summary>Comparison with baseline</summary>
                          <JsonSummary data={item.comparison} />
                        </details>
                      )}

                      {item.reason && <p className="reason-text">{item.reason}</p>}
                      {item.error && <p className="reason-text">Error: {item.error}</p>}
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState message="No experiment results were returned." />
              )}
            </section>

            {/* 6. Critic Agent and 7. Feedback Agent */}
            <section className="results-grid" id="verification">
              <div className="panel agent-report-panel">
                <div className="panel-heading">
                  <div>
                    <h2><ShieldCheck size={19} /> Critic verification</h2>
                    <p>Checks on experiment evidence and limitations.</p>
                  </div>
                  <AgentChat agent={AGENTS[5]} result={result} />
                </div>

                <div className="verification-status">
                  {verification?.status || "No status returned"}
                </div>
                <JsonSummary data={verification} />
              </div>

              <div className="panel agent-report-panel">
                <div className="panel-heading">
                  <div>
                    <h2><BrainCircuit size={19} /> Next action</h2>
                    <p>Recommendation from the feedback agent.</p>
                  </div>
                  <AgentChat agent={AGENTS[6]} result={result} />
                </div>

                <div className="action-name">
                  {feedback?.action || "No action returned"}
                </div>
                <p className="reason-text">
                  {feedback?.reason || "No reason returned."}
                </p>
                <JsonSummary data={feedback} />
              </div>
            </section>

            {/* Final model recommendation */}
            <section className="panel best-model-panel">
              <div className="best-model-heading">
                <div className="best-model-icon"><Trophy size={26} /></div>
                <div>
                  <div className="eyebrow">FINAL MODEL RECOMMENDATION</div>
                  <h2>Best model to use</h2>
                  <p>Selected from the successful baseline models.</p>
                </div>
              </div>

              {bestModel ? (
                <>
                  <div className="best-model-name">{bestModel.model}</div>

                  <div className="best-model-score">
                    <span>{activeMetric.label}</span>
                    <strong>{bestModel.score.toFixed(4)}</strong>
                  </div>

                  <p className="reason-text">
                    Ranked first by {activeMetric.label} using {bestModel.source.toLowerCase()}.
                    {bestModel.cv
                      ? ` Cross-validation standard deviation: ${Number(bestModel.cv.std).toFixed(4)}.`
                      : " Cross-validation was unavailable, so the held-out test score was used as a fallback."}
                  </p>

                  <h3>Model ranking</h3>

                  <div className="ranking-list">
                    {rankedModels.map((item, index) => (
                      <div className="ranking-row" key={item.model}>
                        <span className="ranking-position">#{index + 1}</span>
                        <span className="ranking-model">{item.model}</span>
                        <strong>{item.score.toFixed(4)}</strong>
                      </div>
                    ))}
                  </div>

                  <p className="chart-caption">
                    The winner is the best candidate among models successfully
                    evaluated, not a guarantee of future performance.
                  </p>
                </>
              ) : (
                <EmptyState message="No successful baseline model has a usable score for the selected metric." />
              )}
            </section>

            <button
              className="secondary-button"
              onClick={() => {
                setResult(null);
                setError("");
                setFile(null);
                setColumns([]);
                setTarget("");
                scrollToSection("dataset-analysis");
              }}
            >
              Analyze another dataset
            </button>
          </>
        )}

        <footer className="footer">
          <BrainCircuit size={17} />
          <span>QualiMind AI · Quality-Aware ML Pipeline</span>
          <span>Local development</span>
        </footer>
      </main>
    </div>
  );
}
function AgentChat({ agent, result }) {
  const [isOpen, setIsOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatError, setChatError] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const askQuestion = async (event, suggestedQuestion) => {
    event?.preventDefault();

    const userQuestion = (suggestedQuestion ?? question).trim();

    if (!userQuestion || chatLoading) return;

    setIsOpen(true);
    setChatError("");
    setQuestion("");

    setMessages((previous) => [
      ...previous,
      { role: "user", content: userQuestion },
    ]);

    setChatLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          agent_name: agent.key,
          question: userQuestion,
          context: result,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to get a chatbot response.");
      }

      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: data.answer },
      ]);
    } catch (err) {
      setChatError(err.message || "Could not connect to the chatbot.");
    } finally {
      setChatLoading(false);
    }
  };

  const Icon = agent.icon;

  return (
    <div className="compact-agent-chat">
      <button
        type="button"
        className={`agent-chat-toggle ${isOpen ? "chat-open" : ""}`}
        onClick={() => setIsOpen((previous) => !previous)}
        aria-expanded={isOpen}
        aria-label={`${isOpen ? "Close" : "Open"} ${agent.title} chatbot`}
        title={`Chat with ${agent.title}`}
      >
        {isOpen ? (
          <span className="chat-close-symbol">×</span>
        ) : (
          <MessageCircle size={21} />
        )}
        {!isOpen && <span className="chat-toggle-label">Ask AI</span>}
      </button>

      {isOpen && (
        <section className="compact-chat-window">
          <header className="compact-chat-header">
            <div className="compact-chat-agent-icon">
              <Icon size={19} />
            </div>

            <div className="compact-chat-title">
              <strong>{agent.title}</strong>
              <span>Ask about this report</span>
            </div>

            <button
              type="button"
              className="compact-chat-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close chatbot"
            >
              ×
            </button>
          </header>

          <div className="compact-chat-suggestions">
            {agent.suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                disabled={chatLoading}
                onClick={(event) => askQuestion(event, suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>

          <div className="compact-chat-messages" aria-live="polite">
            {messages.length === 0 && (
              <div className="compact-chat-empty">
                <MessageCircle size={22} />
                <strong>What would you like to understand?</strong>
                <p>
                  Ask a question about the findings in this agent's report.
                </p>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={`agent-chat-message ${
                  message.role === "user"
                    ? "user-message"
                    : "assistant-message"
                }`}
              >
                <strong>
                  {message.role === "user" ? "You" : agent.title}
                </strong>
                <p>{message.content}</p>
              </div>
            ))}

            {chatLoading && (
              <div className="agent-chat-loading">
                <LoaderCircle className="spin" size={16} />
                <span>Thinking...</span>
              </div>
            )}
          </div>

          {chatError && (
            <div className="agent-chat-error" role="alert">
              {chatError}
            </div>
          )}

          <form
            className="compact-chat-form"
            onSubmit={(event) => askQuestion(event)}
          >
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={`Ask ${agent.title}...`}
              aria-label={`Question for ${agent.title}`}
              maxLength={4000}
              disabled={chatLoading}
            />

            <button
              type="submit"
              disabled={chatLoading || !question.trim()}
              aria-label="Send message"
            >
              {chatLoading ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <Send size={17} />
              )}
            </button>
          </form>
        </section>
      )}
    </div>
  );
}

function StatCard({ icon, label, value, compact = false }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <p>{label}</p>
      <strong className={compact ? "stat-value compact-value" : "stat-value"}>
        {value ?? "—"}
      </strong>
    </div>
  );
}

function Detail({ label, value }) {
  const displayValue =
    value == null
      ? "Not provided"
      : typeof value === "object"
        ? JSON.stringify(value)
        : String(value);

  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{displayValue}</strong>
    </div>
  );
}

function EmptyState({ message }) {
  return <div className="empty-state">{message}</div>;
}

function JsonSummary({ data }) {
  if (data == null) {
    return <EmptyState message="No information returned." />;
  }

  return (
    <details className="json-details">
      <summary>View full report</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

export default App;