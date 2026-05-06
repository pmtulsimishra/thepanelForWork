
import { useState, useRef, useCallback } from "react";
import "./App.css";

function trimTrailingSlash(value) {
  return String(value || "").replace(/\/+$/, "");
}

function extractErrorMessage(error) {
  return String(error?.message || error || "Unknown error");
}

function isProviderUnavailableMessage(message) {
  const text = String(message || "").toLowerCase();
  return (
    text.includes("unauthorized") ||
    text.includes("forbidden") ||
    text.includes("invalid api key") ||
    text.includes("api key") ||
    text.includes("github models is disabled") ||
    text.includes("disabled") ||
    text.includes("401") ||
    text.includes("403") ||
    text.includes("insufficient_quota") ||
    text.includes("deploymentnotfound") ||
    text.includes("no model providers configured")
  );
}

function isCloudUnavailableError(error) {
  if (error?.unavailableOnly) return true;
  const message = String(error?.message || "").toLowerCase();
  return (
    message.includes("github models is disabled") ||
    message.includes("unauthorized") ||
    message.includes("401") ||
    message.includes("forbidden") ||
    message.includes("403") ||
    message.includes("invalid api key") ||
    message.includes("no model providers configured") ||
    message.includes("all configured model providers are unavailable")
  );
}

function buildProviders() {
  const providers = [];

  const azureEndpoint = trimTrailingSlash(process.env.REACT_APP_AZURE_OPENAI_ENDPOINT);
  const azureKey = process.env.REACT_APP_AZURE_OPENAI_API_KEY;
  const azureDeployment = process.env.REACT_APP_AZURE_OPENAI_DEPLOYMENT;
  const azureApiVersion = process.env.REACT_APP_AZURE_OPENAI_API_VERSION || "2024-06-01";
  if (azureEndpoint && azureKey && azureDeployment) {
    providers.push({
      id: "azure-openai",
      label: "Azure OpenAI",
      model: azureDeployment,
      call: async (systemPrompt, userMessage) => {
        const url = `${azureEndpoint}/openai/deployments/${azureDeployment}/chat/completions?api-version=${encodeURIComponent(azureApiVersion)}`;
        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "api-key": azureKey,
          },
          body: JSON.stringify({
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: userMessage },
            ],
            max_tokens: 1000,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Azure OpenAI failed (${response.status}): ${errorText || "Unknown error"}`);
        }

        const data = await response.json();
        const content = data.choices?.[0]?.message?.content;
        if (!content) {
          throw new Error("Azure OpenAI returned an empty response.");
        }
        return content;
      },
    });
  }

  const openAIKey = process.env.REACT_APP_OPENAI_API_KEY;
  const openAIBase = trimTrailingSlash(process.env.REACT_APP_OPENAI_BASE_URL || "https://api.openai.com/v1");
  const openAIModel = process.env.REACT_APP_OPENAI_MODEL || "gpt-4o-mini";
  if (openAIKey) {
    providers.push({
      id: "openai",
      label: "OpenAI",
      model: openAIModel,
      call: async (systemPrompt, userMessage) => {
        const response = await fetch(`${openAIBase}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${openAIKey}`,
          },
          body: JSON.stringify({
            model: openAIModel,
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: userMessage },
            ],
            max_tokens: 1000,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`OpenAI failed (${response.status}): ${errorText || "Unknown error"}`);
        }

        const data = await response.json();
        const content = data.choices?.[0]?.message?.content;
        if (!content) {
          throw new Error("OpenAI returned an empty response.");
        }
        return content;
      },
    });
  }

  const githubToken = process.env.REACT_APP_GITHUB_TOKEN;
  const githubModel = process.env.REACT_APP_GITHUB_MODEL || "gpt-4o";
  const githubEndpoint = process.env.REACT_APP_GITHUB_MODELS_ENDPOINT || "https://models.inference.ai.azure.com/chat/completions";
  if (githubToken) {
    providers.push({
      id: "github-models",
      label: "GitHub Models",
      model: githubModel,
      call: async (systemPrompt, userMessage) => {
        const response = await fetch(githubEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${githubToken}`,
          },
          body: JSON.stringify({
            model: githubModel,
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: userMessage },
            ],
            max_tokens: 1000,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`GitHub Models failed (${response.status}): ${errorText || "Unknown error"}`);
        }

        const data = await response.json();
        const content = data.choices?.[0]?.message?.content;
        if (!content) {
          throw new Error("GitHub Models returned an empty response.");
        }
        return content;
      },
    });
  }

  const priority = (process.env.REACT_APP_MODEL_PROVIDER_PRIORITY || "azure-openai,openai,github-models")
    .split(",")
    .map(v => v.trim().toLowerCase())
    .filter(Boolean);

  providers.sort((a, b) => {
    const ai = priority.indexOf(a.id);
    const bi = priority.indexOf(b.id);
    const aRank = ai === -1 ? Number.MAX_SAFE_INTEGER : ai;
    const bRank = bi === -1 ? Number.MAX_SAFE_INTEGER : bi;
    return aRank - bRank;
  });

  return providers;
}

function inferContextFallback(documentText) {
  const text = documentText || "";
  const words = text.trim().split(/\s+/).filter(Boolean);
  const sentenceCount = text.split(/[.!?]+/).filter(s => s.trim()).length;
  const lower = text.toLowerCase();

  const audience = lower.includes("team")
    ? "Internal team or stakeholders"
    : lower.includes("customer") || lower.includes("user")
    ? "External customers or end users"
    : "General professional audience";

  const goal = lower.includes("proposal")
    ? "Persuade readers to approve a proposal"
    : lower.includes("plan")
    ? "Outline a plan and align execution"
    : "Inform readers and drive a next step";

  const stage = words.length < 200
    ? "Early draft"
    : words.length < 800
    ? "Working draft"
    : "Detailed draft";

  const domain = lower.includes("api") || lower.includes("architecture")
    ? "Software/Technical"
    : lower.includes("budget") || lower.includes("revenue")
    ? "Business/Finance"
    : "General";

  const tone = lower.includes("must") || lower.includes("required")
    ? "Directive"
    : lower.includes("maybe") || lower.includes("could")
    ? "Exploratory"
    : "Neutral";

  const keyRisks = sentenceCount < 6
    ? "The document may be too brief to fully justify claims or decisions. Add concrete evidence and specifics."
    : "Potential risks include unclear assumptions, missing evidence, and insufficiently explicit next actions.";

  return { audience, goal, stage, domain, tone, keyRisks };
}

function runReviewFallback(persona, documentText, context) {
  const text = documentText || "";
  const words = text.trim().split(/\s+/).filter(Boolean);
  const sentenceCount = text.split(/[.!?]+/).filter(s => s.trim()).length;
  const hasNumbers = /\d/.test(text);
  const hasBullets = /(^|\n)\s*[-*•]/.test(text);

  const strengths = [];
  if (hasBullets) strengths.push("Structured sections or bullet points improve scanability.");
  if (hasNumbers) strengths.push("Use of numbers suggests some claims are grounded in specifics.");
  if (words.length > 400) strengths.push("Depth is sufficient for a substantive review.");
  if (strengths.length === 0) strengths.push("The draft is concise and easy to skim.");

  const risks = [];
  if (words.length < 250) risks.push("The draft is short and may not provide enough supporting evidence.");
  if (!hasNumbers) risks.push("Key claims are not quantified; add data points where possible.");
  if (sentenceCount > 0 && words.length / sentenceCount > 26) risks.push("Sentences may be dense; shorten or split for readability.");
  if (risks.length === 0) risks.push("No major structural risks detected, but validate assumptions with a domain expert.");

  const personaFocus = `Focus: ${persona.name}. Review perspective based on this persona prompt: ${persona.prompt}`;
  const contextSummary = context && Object.keys(context).length
    ? `\n\nContext summary\n- Audience: ${context.audience || "unknown"}\n- Goal: ${context.goal || "unknown"}\n- Stage: ${context.stage || "unknown"}\n- Domain: ${context.domain || "unknown"}\n- Tone: ${context.tone || "unknown"}`
    : "";

  return `${personaFocus}${contextSummary}\n\nStrengths\n- ${strengths.join("\n- ")}\n\nRisks / Gaps\n- ${risks.join("\n- ")}\n\nRecommended revisions\n- Make the objective explicit in the opening paragraph.\n- Add concrete examples or metrics to support major claims.\n- End with clear next steps, owners, and timeline.\n\nQuestions to resolve\n- Which audience decision should this document influence?\n- What assumptions are currently unstated?\n- What proof would make skeptical readers say yes?\n\nNote: This review was generated in offline fallback mode because cloud model access is unavailable.`;
}

async function callModel(systemPrompt, userMessage) {
  const providers = buildProviders();
  if (!providers.length) {
    throw new Error("No model providers configured. Set Azure OpenAI, OpenAI, or GitHub Models env vars.");
  }

  const failures = [];
  for (const provider of providers) {
    try {
      const content = await provider.call(systemPrompt, userMessage);
      return {
        content,
        providerLabel: `${provider.label} / ${provider.model}`,
      };
    } catch (error) {
      const message = extractErrorMessage(error);
      failures.push({
        provider: provider.label,
        message,
        unavailable: isProviderUnavailableMessage(message),
      });
    }
  }

  const unavailableOnly = failures.length > 0 && failures.every(f => f.unavailable);
  const details = failures.map(f => `[${f.provider}] ${f.message}`).join(" | ");
  const aggregateError = new Error(
    unavailableOnly
      ? `All configured model providers are unavailable. ${details}`
      : `All configured model providers failed. ${details}`
  );
  aggregateError.unavailableOnly = unavailableOnly;
  throw aggregateError;
}

const DEFAULT_PERSONAS = [
  {
    id: 1,
    name: "Devil's Advocate",
    emoji: "😈",
    color: "#c0392b",
    prompt: "You are a sharp devil's advocate. Challenge assumptions, find logical flaws, and push back on weak arguments. Be direct and critical but constructive. Point out what doesn't hold up and why.",
  },
  {
    id: 2,
    name: "Clarity Coach",
    emoji: "🔍",
    color: "#16a085",
    prompt: "You are a writing clarity coach. Focus on readability, structure, flow, and whether ideas are communicated effectively. Identify confusing sections, jargon, or unclear transitions. Suggest concrete improvements.",
  },
  {
    id: 3,
    name: "Domain Expert",
    emoji: "🎓",
    color: "#7d3c98",
    prompt: "You are a rigorous domain expert. Evaluate accuracy, depth, and completeness. Flag anything oversimplified, incorrect, or missing important nuance. Ask probing questions about the subject matter.",
  },
  {
    id: 4,
    name: "End User",
    emoji: "👤",
    color: "#d4860a",
    prompt: "You are the target audience reading this document. React naturally — what resonates? What confuses you? What questions do you have? What makes you more or less engaged or persuaded?",
  },
];

async function inferContext(documentText) {
  try {
    const result = await callModel(
      `Analyze this document and infer its context. Respond ONLY with a JSON object, no markdown or explanation:
{
  "audience": "who this is written for",
  "goal": "what this document is trying to accomplish",
  "stage": "how complete it seems",
  "domain": "subject area or industry",
  "tone": "current tone of the writing",
  "keyRisks": "1-2 sentence summary of main weaknesses"
}`,
      documentText.slice(0, 3000)
    );
    try {
      return JSON.parse(result.content.replace(/```json|```/g, "").trim());
    } catch {
      return {};
    }
  } catch (error) {
    if (isCloudUnavailableError(error)) {
      return inferContextFallback(documentText);
    }
    throw error;
  }
}

async function runReview(persona, documentText, context) {
  const ctx = context && Object.keys(context).length
    ? `\n\n--- DOCUMENT CONTEXT ---
Audience: ${context.audience || "unknown"}
Goal: ${context.goal || "unknown"}
Stage: ${context.stage || "unknown"}
Domain: ${context.domain || "unknown"}
Tone: ${context.tone || "unknown"}
Key Risks: ${context.keyRisks || "none identified"}
---`
    : "";
  try {
    const result = await callModel(persona.prompt + ctx, `Please review this document:\n\n${documentText}`);
    return {
      text: result.content,
      source: "cloud",
      providerLabel: result.providerLabel,
    };
  } catch (error) {
    if (isCloudUnavailableError(error)) {
      return {
        text: runReviewFallback(persona, documentText, context),
        source: "offline",
        providerLabel: null,
      };
    }
    throw error;
  }
}
async function extractTextFromFile(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".txt") || name.endsWith(".md") || name.endsWith(".csv")) {
    return await file.text();
  }
  
  if (name.endsWith(".pdf")) {
    const pdfjsLib = await import("pdfjs-dist");
    pdfjsLib.GlobalWorkerOptions.workerSrc = 
      `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let text = "";
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map(item => item.str).join(" ") + "\n";
    }
    return text;
  }
  return await file.text();
}

export default function ThePanel() {
  const [docText, setDocText] = useState("");
  const [fileName, setFileName] = useState(null);
  const [personas, setPersonas] = useState(DEFAULT_PERSONAS);
  const [reviews, setReviews] = useState({});
  const [loading, setLoading] = useState({});
  const [context, setContext] = useState(null);
  const [inferring, setInferring] = useState(false);
  const [runningAll, setRunningAll] = useState(false);
  const [activeTab, setActiveTab] = useState(null);
  const [showContext, setShowContext] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editForm, setEditForm] = useState({});
  const [addingNew, setAddingNew] = useState(false);
  const [newForm, setNewForm] = useState({ name: "", emoji: "🤖", color: "#4a90e2", prompt: "" });
  const [errorMessage, setErrorMessage] = useState("");
  const [modeNotice, setModeNotice] = useState("");
  const fileInputRef = useRef(null);
  const contextRef = useRef(null);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    setErrorMessage("");
    setModeNotice("");
    setFileName(file.name);
    setContext(null);
    contextRef.current = null;
    setReviews({});
    setActiveTab(null);
    try {
      const text = await extractTextFromFile(file);
      setDocText(text);
    } catch {
      setDocText("Error reading file. Try pasting the text directly.");
      setErrorMessage("Could not read that file. Try a different file or paste text directly.");
    }
  }, []);

  const handleDocChange = (text) => {
    setErrorMessage("");
    setModeNotice("");
    setDocText(text);
    setFileName(null);
    setContext(null);
    contextRef.current = null;
    setReviews({});
    setActiveTab(null);
  };

  const ensureContext = async (text) => {
    if (contextRef.current) return contextRef.current;
    setInferring(true);
    try {
      const ctx = await inferContext(text);
      contextRef.current = ctx;
      setContext(ctx);
      return ctx;
    } finally {
      setInferring(false);
    }
  };

  const reviewOne = async (persona) => {
    if (!docText.trim() || loading[persona.id]) return;
    setErrorMessage("");
    setModeNotice("");
    setLoading(l => ({ ...l, [persona.id]: true }));
    setActiveTab(persona.id);
    try {
      const ctx = await ensureContext(docText);
      const result = await runReview(persona, docText, ctx);
      if (result.source === "offline") {
        setModeNotice("Cloud model unavailable. Showing offline fallback reviews.");
      } else if (result.providerLabel) {
        setModeNotice(`Using ${result.providerLabel}`);
      }
      setReviews(r => ({ ...r, [persona.id]: result.text }));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to run reviewer.");
    } finally {
      setLoading(l => ({ ...l, [persona.id]: false }));
    }
  };

  const reviewAll = async () => {
    if (!docText.trim() || runningAll) return;
    setErrorMessage("");
    setModeNotice("");
    setRunningAll(true);
    try {
      const ctx = await ensureContext(docText);
      const results = await Promise.all(
        personas.map(p => runReview(p, docText, ctx).then(result => ({ id: p.id, result })))
      );
      const anyOffline = results.some(({ result }) => result.source === "offline");
      const providerLabels = [...new Set(results.map(({ result }) => result.providerLabel).filter(Boolean))];
      if (anyOffline) {
        setModeNotice("Cloud model unavailable. Showing offline fallback reviews.");
      } else if (providerLabels.length) {
        setModeNotice(`Using ${providerLabels.join(", ")}`);
      }
      const newReviews = {};
      results.forEach(({ id, result }) => { newReviews[id] = result.text; });
      setReviews(prev => ({ ...prev, ...newReviews }));
      setActiveTab(personas[0]?.id);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to run all reviewers.");
    } finally {
      setRunningAll(false);
    }
  };

  const startEdit = (p) => {
    setEditingId(p.id);
    setEditForm({ name: p.name, emoji: p.emoji, color: p.color, prompt: p.prompt });
  };

  const saveEdit = () => {
    setPersonas(ps => ps.map(p => p.id === editingId ? { ...p, ...editForm } : p));
    setEditingId(null);
  };

  const deletePersona = (id) => {
    setPersonas(ps => ps.filter(p => p.id !== id));
    setReviews(r => { const n = { ...r }; delete n[id]; return n; });
    if (activeTab === id) setActiveTab(null);
  };

  const saveNew = () => {
    if (!newForm.name.trim() || !newForm.prompt.trim()) return;
    const id = Date.now();
    setPersonas(ps => [...ps, { ...newForm, id }]);
    setAddingNew(false);
    setNewForm({ name: "", emoji: "🤖", color: "#4a90e2", prompt: "" });
  };

  const hasDoc = docText.trim().length > 0;
  const activePersona = personas.find(p => p.id === activeTab);
  const PRESET_COLORS = ["#c0392b","#16a085","#7d3c98","#d4860a","#2471a3","#1e8449","#884ea0","#616a6b"];

  return (
    <div className="app-container">
      {/* Header */}
      <div className="app-header">
        <div className="app-header-title">
          <div className="app-header-subtitle">Document Review Suite</div>
          <h1>The Panel</h1>
        </div>
        <div className="app-header-controls">
          {inferring && <span className="status-indicator">inferring context…</span>}
          {context && !inferring && (
            <button 
              onClick={() => setShowContext(v => !v)} 
              className="btn btn-secondary"
              style={{ padding: "6px 12px", fontSize: "11px" }}
            >
              {showContext ? "hide context" : "view context"}
            </button>
          )}
          <span className="status-indicator">{personas.length} reviewers</span>
        </div>
      </div>

      {/* Context Bar */}
      {showContext && context && (
        <div className="context-bar">
          {Object.entries(context).map(([k, v]) => v && (
            <div key={k} className="context-item">
              <div className="context-label">{k}</div>
              <div className="context-value">{v}</div>
            </div>
          ))}
        </div>
      )}

      {/* Error Notice */}
      {errorMessage && (
        <div className="notice-bar notice-error">
          {errorMessage}
        </div>
      )}

      {/* Mode Notice */}
      {modeNotice && (
        <div className="notice-bar notice-info">
          {modeNotice}
        </div>
      )}

      <div className="main-layout">
        {/* Left Sidebar */}
        <div className="sidebar">
          {/* Document input section */}
          <div className="sidebar-section">
            <div className="sidebar-label">Document</div>

            <div
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
              onClick={() => fileInputRef.current?.click()}
              className={`file-drop-zone ${dragging ? "dragging" : ""}`}
            >
              <div className="file-drop-zone-icon">📎</div>
              <div className="file-drop-zone-text">
                {fileName || "drop file or click to upload"}
              </div>
              <div className="file-drop-zone-hint">pdf · txt · md</div>
            </div>
            <input 
              ref={fileInputRef} 
              type="file" 
              accept=".txt,.md,.pdf,.csv" 
              className="file-input" 
              onChange={e => handleFile(e.target.files[0])} 
            />

            <textarea
              value={docText}
              onChange={e => handleDocChange(e.target.value)}
              placeholder="…or paste text here"
              className="text-input document-textarea"
            />
            {hasDoc && (
              <button 
                onClick={() => { setDocText(""); setFileName(null); setContext(null); contextRef.current = null; setReviews({}); setActiveTab(null); setErrorMessage(""); setModeNotice(""); }}
                className="btn-clear"
              >
                clear
              </button>
            )}
          </div>

          {/* Run All Button */}
          <div className="sidebar-section" style={{ padding: "14px 20px", borderBottom: "1px solid #222" }}>
            <button
              onClick={reviewAll}
              disabled={!hasDoc || runningAll}
              className="btn btn-primary"
              style={{ fontSize: "11px" }}
            >
              {runningAll ? "reviewing…" : "run all reviewers →"}
            </button>
          </div>

          {/* Personas list */}
          <div className="sidebar-section">
            <div className="sidebar-label">Reviewers</div>

            <div className="personas-list">
              {personas.map(p => (
                <div 
                  key={p.id}
                  onClick={() => { if (reviews[p.id]) setActiveTab(p.id); }}
                  className={`persona-item ${activeTab === p.id ? "active" : ""}`}
                  style={{
                    borderColor: activeTab === p.id ? p.color + "55" : undefined,
                    background: activeTab === p.id ? p.color + "10" : undefined,
                    cursor: reviews[p.id] ? "pointer" : "default",
                  }}
                >
                  <div className="persona-info">
                    <span className="persona-emoji">{p.emoji}</span>
                    <span className="persona-name">{p.name}</span>
                  </div>
                  <div className="persona-actions">
                    {loading[p.id] && <span className="status-dot">…</span>}
                    {reviews[p.id] && !loading[p.id] && <span className="status-dot" style={{ color: p.color }}>✓</span>}
                    <button 
                      onClick={e => { e.stopPropagation(); reviewOne(p); }}
                      disabled={!hasDoc || !!loading[p.id]}
                      className="btn-icon"
                    >
                      run
                    </button>
                    <button 
                      onClick={e => { e.stopPropagation(); startEdit(p); }}
                      className="btn-icon"
                    >
                      edit
                    </button>
                    <button 
                      onClick={e => { e.stopPropagation(); deletePersona(p.id); }}
                      className="btn-icon"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {!addingNew ? (
              <button 
                onClick={() => setAddingNew(true)}
                className="btn-add-reviewer"
              >
                + add reviewer
              </button>
            ) : (
              <div className="add-persona-form">
                <div className="form-row">
                  <input 
                    value={newForm.emoji} 
                    onChange={e => setNewForm(f => ({ ...f, emoji: e.target.value }))} 
                    className="text-input form-input"
                    style={{ width: "50px", textAlign: "center" }}
                    placeholder="🤖" 
                  />
                  <input 
                    value={newForm.name} 
                    onChange={e => setNewForm(f => ({ ...f, name: e.target.value }))} 
                    className="text-input form-input"
                    placeholder="Reviewer name" 
                  />
                </div>
                <div className="color-picker-row">
                  {PRESET_COLORS.map(c => (
                    <div 
                      key={c} 
                      onClick={() => setNewForm(f => ({ ...f, color: c }))}
                      className={`color-swatch ${newForm.color === c ? "selected" : ""}`}
                      style={{ backgroundColor: c }}
                    />
                  ))}
                </div>
                <textarea 
                  value={newForm.prompt} 
                  onChange={e => setNewForm(f => ({ ...f, prompt: e.target.value }))}
                  placeholder="Describe this reviewer's perspective…"
                  className="text-input"
                  style={{ minHeight: "70px", marginBottom: "8px" }}
                />
                <div className="form-buttons">
                  <button onClick={saveNew} className="btn btn-primary" style={{ fontSize: "11px" }}>save</button>
                  <button onClick={() => setAddingNew(false)} className="btn btn-secondary" style={{ fontSize: "11px" }}>cancel</button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Content Area */}
        <div className="content-area">
          {/* Review Tabs */}
          {Object.keys(reviews).length > 0 && (
            <div className="tabs-bar">
              {personas.filter(p => reviews[p.id]).map(p => (
                <button 
                  key={p.id} 
                  onClick={() => setActiveTab(p.id)}
                  className={`tab ${activeTab === p.id ? "active" : ""}`}
                  style={{
                    borderBottomColor: activeTab === p.id ? p.color : undefined,
                  }}
                >
                  {p.emoji} {p.name}
                </button>
              ))}
            </div>
          )}

          {/* Main Content */}
          <div className="content-main">
            {!hasDoc && (
              <div className="empty-state">
                <div className="empty-state-icon">📄</div>
                <div className="empty-state-text">upload or paste a document to begin</div>
              </div>
            )}

            {hasDoc && Object.keys(reviews).length === 0 && !runningAll && (
              <div className="empty-state">
                <div className="empty-state-icon">👥</div>
                <div className="empty-state-text">run all reviewers or pick one from the left</div>
              </div>
            )}

            {runningAll && Object.keys(reviews).length === 0 && (
              <div className="loading-state">
                <div className="loading-text">the panel is reviewing your document…</div>
              </div>
            )}

            {activePersona && reviews[activePersona.id] && (
              <div className="review-result">
                <div className="review-header">
                  <span className="review-emoji">{activePersona.emoji}</span>
                  <div>
                    <div className="review-title">{activePersona.name}</div>
                    <div className="review-status" style={{ color: activePersona.color }}>review complete</div>
                  </div>
                </div>
                <div className="review-body">
                  {reviews[activePersona.id]}
                </div>
              </div>
            )}

            {activeTab && loading[activeTab] && !reviews[activeTab] && (
              <div className="loading-state">
                <div className="loading-text">
                  {activePersona?.emoji} {activePersona?.name} is reviewing…
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Edit Modal */}
      {editingId && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-title">Edit Reviewer</div>
            <div style={{ display: "flex", gap: "8px", marginBottom: "10px" }}>
              <input 
                value={editForm.emoji} 
                onChange={e => setEditForm(f => ({ ...f, emoji: e.target.value }))} 
                className="text-input"
                style={{ width: "50px", textAlign: "center" }}
              />
              <input 
                value={editForm.name} 
                onChange={e => setEditForm(f => ({ ...f, name: e.target.value }))} 
                className="text-input"
                placeholder="Name" 
              />
            </div>
            <div className="color-picker-row" style={{ marginBottom: "10px" }}>
              {PRESET_COLORS.map(c => (
                <div 
                  key={c} 
                  onClick={() => setEditForm(f => ({ ...f, color: c }))}
                  className={`color-swatch ${editForm.color === c ? "selected" : ""}`}
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
            <textarea 
              value={editForm.prompt} 
              onChange={e => setEditForm(f => ({ ...f, prompt: e.target.value }))}
              className="text-input"
              style={{ minHeight: "100px", marginBottom: "12px" }}
            />
            <div className="modal-buttons">
              <button onClick={saveEdit} className="btn btn-primary" style={{ fontSize: "11px" }}>save</button>
              <button onClick={() => setEditingId(null)} className="btn btn-secondary" style={{ fontSize: "11px" }}>cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
