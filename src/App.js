
import { useState, useRef, useCallback } from "react";

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

  const inputStyle = {
    background: "#111", border: "1px solid #252525", borderRadius: "3px",
    color: "#ccc", padding: "8px 10px", fontSize: "12px", fontFamily: "monospace",
    outline: "none", width: "100%", boxSizing: "border-box",
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f0f", color: "#e0e0e0", fontFamily: "'Georgia', serif", display: "flex", flexDirection: "column" }}>

      {/* Header */}
      <div style={{ borderBottom: "1px solid #1c1c1c", padding: "18px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, zIndex: 20, background: "#0f0f0f" }}>
        <div>
          <div style={{ fontSize: "9px", letterSpacing: "4px", color: "#3a3a3a", textTransform: "uppercase", fontFamily: "monospace", marginBottom: "3px" }}>Document Review Suite</div>
          <h1 style={{ margin: 0, fontSize: "20px", fontWeight: "400", color: "#f0f0f0", letterSpacing: "-0.3px" }}>The Panel</h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {inferring && <span style={{ fontSize: "10px", color: "#444", fontFamily: "monospace", animation: "pulse 1.5s infinite" }}>inferring context…</span>}
          {context && !inferring && (
            <button onClick={() => setShowContext(v => !v)} style={{ background: "none", border: "1px solid #222", borderRadius: "3px", color: "#555", padding: "5px 12px", fontSize: "10px", cursor: "pointer", fontFamily: "monospace", letterSpacing: "1px" }}>
              {showContext ? "hide context" : "view context"}
            </button>
          )}
          <span style={{ fontSize: "10px", color: "#2a2a2a", fontFamily: "monospace" }}>{personas.length} reviewers</span>
        </div>
      </div>

      {/* Context Bar */}
      {showContext && context && (
        <div style={{ background: "#111", borderBottom: "1px solid #1c1c1c", padding: "14px 28px", display: "flex", gap: "28px", flexWrap: "wrap" }}>
          {Object.entries(context).map(([k, v]) => v && (
            <div key={k}>
              <div style={{ fontSize: "9px", letterSpacing: "2px", color: "#3a3a3a", textTransform: "uppercase", fontFamily: "monospace", marginBottom: "3px" }}>{k}</div>
              <div style={{ fontSize: "12px", color: "#888", maxWidth: "220px", lineHeight: "1.4" }}>{v}</div>
            </div>
          ))}
        </div>
      )}

      {errorMessage && (
        <div style={{ background: "#2a1515", borderBottom: "1px solid #472424", color: "#f2b6b6", padding: "10px 28px", fontSize: "11px", fontFamily: "monospace" }}>
          {errorMessage}
        </div>
      )}

      {modeNotice && (
        <div style={{ background: "#1a1d28", borderBottom: "1px solid #2d3552", color: "#b7c7ff", padding: "10px 28px", fontSize: "11px", fontFamily: "monospace" }}>
          {modeNotice}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", flex: 1 }}>

        {/* Left Column */}
        <div style={{ borderRight: "1px solid #1c1c1c", display: "flex", flexDirection: "column" }}>

          {/* Document input */}
          <div style={{ padding: "20px", borderBottom: "1px solid #1c1c1c" }}>
            <div style={{ fontSize: "9px", letterSpacing: "2px", color: "#3a3a3a", textTransform: "uppercase", fontFamily: "monospace", marginBottom: "10px" }}>Document</div>

            <div
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `1px dashed ${dragging ? "#444" : "#222"}`,
                borderRadius: "3px", padding: "12px", marginBottom: "8px",
                cursor: "pointer", textAlign: "center", background: dragging ? "#141414" : "transparent",
                transition: "all 0.15s",
              }}
            >
              <div style={{ fontSize: "16px", marginBottom: "3px" }}>📎</div>
              <div style={{ fontSize: "10px", color: fileName ? "#777" : "#3a3a3a", fontFamily: "monospace" }}>
                {fileName || "drop file or click to upload"}
              </div>
              <div style={{ fontSize: "9px", color: "#2a2a2a", marginTop: "2px", fontFamily: "monospace" }}>pdf · txt · md</div>
            </div>
            <input ref={fileInputRef} type="file" accept=".txt,.md,.pdf,.csv" style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />

            <textarea
              value={docText}
              onChange={e => handleDocChange(e.target.value)}
              placeholder="…or paste text here"
              style={{ ...inputStyle, minHeight: "120px", resize: "vertical", lineHeight: "1.6", fontSize: "12px", fontFamily: "inherit" }}
            />
            {hasDoc && (
              <button onClick={() => { setDocText(""); setFileName(null); setContext(null); contextRef.current = null; setReviews({}); setActiveTab(null); setErrorMessage(""); setModeNotice(""); }}
                style={{ background: "none", border: "none", color: "#2a2a2a", cursor: "pointer", fontSize: "10px", fontFamily: "monospace", marginTop: "4px", padding: 0 }}>
                clear
              </button>
            )}
          </div>

          {/* Run All */}
          <div style={{ padding: "14px 20px", borderBottom: "1px solid #1c1c1c" }}>
            <button
              onClick={reviewAll}
              disabled={!hasDoc || runningAll}
              style={{
                width: "100%", padding: "10px", borderRadius: "3px", border: "none",
                background: hasDoc && !runningAll ? "#efefef" : "#181818",
                color: hasDoc && !runningAll ? "#0f0f0f" : "#2a2a2a",
                fontSize: "11px", fontFamily: "monospace", cursor: hasDoc && !runningAll ? "pointer" : "default",
                letterSpacing: "1px", transition: "all 0.15s",
              }}
            >
              {runningAll ? "reviewing…" : "run all reviewers →"}
            </button>
          </div>

          {/* Personas list */}
          <div style={{ flex: 1, overflowY: "auto", padding: "14px 20px" }}>
            <div style={{ fontSize: "9px", letterSpacing: "2px", color: "#3a3a3a", textTransform: "uppercase", fontFamily: "monospace", marginBottom: "10px" }}>Reviewers</div>

            {personas.map(p => (
              <div key={p.id}
                onClick={() => { if (reviews[p.id]) setActiveTab(p.id); }}
                style={{
                  marginBottom: "6px", padding: "10px 12px", borderRadius: "3px",
                  border: `1px solid ${activeTab === p.id ? p.color + "55" : "#1c1c1c"}`,
                  background: activeTab === p.id ? p.color + "10" : "#111",
                  cursor: reviews[p.id] ? "pointer" : "default",
                  transition: "all 0.15s",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{ fontSize: "15px" }}>{p.emoji}</span>
                    <span style={{ fontSize: "12px", color: "#ccc" }}>{p.name}</span>
                  </div>
                  <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                    {loading[p.id] && <span style={{ fontSize: "9px", color: "#444", fontFamily: "monospace", animation: "pulse 1.5s infinite" }}>…</span>}
                    {reviews[p.id] && !loading[p.id] && <span style={{ fontSize: "9px", color: p.color, fontFamily: "monospace" }}>✓</span>}
                    <button onClick={e => { e.stopPropagation(); reviewOne(p); }}
                      disabled={!hasDoc || !!loading[p.id]}
                      style={{ background: "none", border: "none", color: hasDoc ? "#555" : "#222", cursor: hasDoc ? "pointer" : "default", fontSize: "10px", fontFamily: "monospace", padding: "0 2px" }}>
                      run
                    </button>
                    <button onClick={e => { e.stopPropagation(); startEdit(p); }}
                      style={{ background: "none", border: "none", color: "#333", cursor: "pointer", fontSize: "10px", fontFamily: "monospace", padding: "0 2px" }}>
                      edit
                    </button>
                    <button onClick={e => { e.stopPropagation(); deletePersona(p.id); }}
                      style={{ background: "none", border: "none", color: "#2a2a2a", cursor: "pointer", fontSize: "10px", fontFamily: "monospace", padding: "0 2px" }}>
                      ✕
                    </button>
                  </div>
                </div>
              </div>
            ))}

            {!addingNew ? (
              <button onClick={() => setAddingNew(true)}
                style={{ width: "100%", marginTop: "6px", padding: "8px", background: "none", border: "1px dashed #1c1c1c", borderRadius: "3px", color: "#333", cursor: "pointer", fontSize: "10px", fontFamily: "monospace", letterSpacing: "1px" }}>
                + add reviewer
              </button>
            ) : (
              <div style={{ marginTop: "8px", padding: "12px", background: "#111", border: "1px solid #1c1c1c", borderRadius: "3px" }}>
                <div style={{ display: "flex", gap: "6px", marginBottom: "6px" }}>
                  <input value={newForm.emoji} onChange={e => setNewForm(f => ({ ...f, emoji: e.target.value }))} style={{ ...inputStyle, width: "40px", textAlign: "center" }} placeholder="🤖" />
                  <input value={newForm.name} onChange={e => setNewForm(f => ({ ...f, name: e.target.value }))} style={{ ...inputStyle, flex: 1 }} placeholder="Reviewer name" />
                </div>
                <div style={{ display: "flex", gap: "4px", marginBottom: "6px" }}>
                  {PRESET_COLORS.map(c => (
                    <div key={c} onClick={() => setNewForm(f => ({ ...f, color: c }))}
                      style={{ width: "16px", height: "16px", borderRadius: "50%", background: c, cursor: "pointer", outline: newForm.color === c ? "2px solid #fff" : "none", outlineOffset: "1px" }} />
                  ))}
                </div>
                <textarea value={newForm.prompt} onChange={e => setNewForm(f => ({ ...f, prompt: e.target.value }))}
                  placeholder="Describe this reviewer's perspective and what they focus on…"
                  style={{ ...inputStyle, minHeight: "70px", resize: "vertical", lineHeight: "1.5", marginBottom: "6px" }} />
                <div style={{ display: "flex", gap: "6px" }}>
                  <button onClick={saveNew} style={{ flex: 1, padding: "7px", background: "#efefef", color: "#0f0f0f", border: "none", borderRadius: "3px", fontSize: "10px", fontFamily: "monospace", cursor: "pointer" }}>save</button>
                  <button onClick={() => setAddingNew(false)} style={{ flex: 1, padding: "7px", background: "none", border: "1px solid #222", borderRadius: "3px", color: "#444", fontSize: "10px", fontFamily: "monospace", cursor: "pointer" }}>cancel</button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Review Output */}
        <div style={{ display: "flex", flexDirection: "column" }}>

          {Object.keys(reviews).length > 0 && (
            <div style={{ display: "flex", borderBottom: "1px solid #1c1c1c", overflowX: "auto", background: "#0f0f0f", flexShrink: 0 }}>
              {personas.filter(p => reviews[p.id]).map(p => (
                <button key={p.id} onClick={() => setActiveTab(p.id)}
                  style={{
                    padding: "12px 18px", background: "none", border: "none",
                    borderBottom: `2px solid ${activeTab === p.id ? p.color : "transparent"}`,
                    color: activeTab === p.id ? "#f0f0f0" : "#444", cursor: "pointer", fontSize: "11px",
                    fontFamily: "monospace", whiteSpace: "nowrap", transition: "all 0.15s",
                  }}>
                  {p.emoji} {p.name}
                </button>
              ))}
            </div>
          )}

          <div style={{ flex: 1, overflowY: "auto", padding: "28px 36px" }}>
            {!hasDoc && (
              <div style={{ textAlign: "center", paddingTop: "80px", color: "#2a2a2a" }}>
                <div style={{ fontSize: "32px", marginBottom: "16px" }}>📄</div>
                <div style={{ fontSize: "12px", fontFamily: "monospace", letterSpacing: "1px" }}>upload or paste a document to begin</div>
              </div>
            )}

            {hasDoc && Object.keys(reviews).length === 0 && !runningAll && (
              <div style={{ textAlign: "center", paddingTop: "80px", color: "#2a2a2a" }}>
                <div style={{ fontSize: "32px", marginBottom: "16px" }}>👥</div>
                <div style={{ fontSize: "12px", fontFamily: "monospace", letterSpacing: "1px" }}>run all reviewers or pick one from the left</div>
              </div>
            )}

            {runningAll && Object.keys(reviews).length === 0 && (
              <div style={{ textAlign: "center", paddingTop: "80px", color: "#333" }}>
                <div style={{ fontSize: "12px", fontFamily: "monospace", letterSpacing: "1px", animation: "pulse 1.5s infinite" }}>the panel is reviewing your document…</div>
              </div>
            )}

            {activePersona && reviews[activePersona.id] && (
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "24px" }}>
                  <span style={{ fontSize: "22px" }}>{activePersona.emoji}</span>
                  <div>
                    <div style={{ fontSize: "16px", color: "#f0f0f0", fontWeight: "400" }}>{activePersona.name}</div>
                    <div style={{ fontSize: "10px", color: activePersona.color, fontFamily: "monospace", marginTop: "2px" }}>review complete</div>
                  </div>
                </div>
                <div style={{ fontSize: "14px", lineHeight: "1.8", color: "#bbb", whiteSpace: "pre-wrap" }}>
                  {reviews[activePersona.id]}
                </div>
              </div>
            )}

            {activeTab && loading[activeTab] && !reviews[activeTab] && (
              <div style={{ textAlign: "center", paddingTop: "60px", color: "#333" }}>
                <div style={{ fontSize: "11px", fontFamily: "monospace", animation: "pulse 1.5s infinite" }}>
                  {activePersona?.emoji} {activePersona?.name} is reviewing…
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Edit Modal */}
      {editingId && (
        <div style={{ position: "fixed", inset: 0, background: "#000000cc", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ background: "#111", border: "1px solid #222", borderRadius: "4px", padding: "24px", width: "360px" }}>
            <div style={{ fontSize: "10px", letterSpacing: "2px", color: "#444", fontFamily: "monospace", marginBottom: "14px", textTransform: "uppercase" }}>Edit Reviewer</div>
            <div style={{ display: "flex", gap: "8px", marginBottom: "10px" }}>
              <input value={editForm.emoji} onChange={e => setEditForm(f => ({ ...f, emoji: e.target.value }))} style={{ ...inputStyle, width: "44px", textAlign: "center" }} />
              <input value={editForm.name} onChange={e => setEditForm(f => ({ ...f, name: e.target.value }))} style={{ ...inputStyle, flex: 1 }} placeholder="Name" />
            </div>
            <div style={{ display: "flex", gap: "4px", marginBottom: "10px" }}>
              {PRESET_COLORS.map(c => (
                <div key={c} onClick={() => setEditForm(f => ({ ...f, color: c }))}
                  style={{ width: "18px", height: "18px", borderRadius: "50%", background: c, cursor: "pointer", outline: editForm.color === c ? "2px solid #fff" : "none", outlineOffset: "1px" }} />
              ))}
            </div>
            <textarea value={editForm.prompt} onChange={e => setEditForm(f => ({ ...f, prompt: e.target.value }))}
              style={{ ...inputStyle, minHeight: "100px", resize: "vertical", lineHeight: "1.5", marginBottom: "12px" }} />
            <div style={{ display: "flex", gap: "8px" }}>
              <button onClick={saveEdit} style={{ flex: 1, padding: "9px", background: "#efefef", color: "#0f0f0f", border: "none", borderRadius: "3px", fontSize: "11px", fontFamily: "monospace", cursor: "pointer" }}>save</button>
              <button onClick={() => setEditingId(null)} style={{ flex: 1, padding: "9px", background: "none", border: "1px solid #222", borderRadius: "3px", color: "#555", fontSize: "11px", fontFamily: "monospace", cursor: "pointer" }}>cancel</button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
      `}</style>
    </div>
  );
}
