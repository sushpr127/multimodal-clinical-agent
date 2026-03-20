import { useState, useRef, useEffect } from "react";

const QUERY_TYPE_LABELS = {
  narrative: { label: "narrative", color: "#085041", bg: "#E1F5EE" },
  numerical: { label: "numerical", color: "#633806", bg: "#FAEEDA" },
  visual:    { label: "visual",    color: "#3C3489", bg: "#EEEDFE" },
  mixed:     { label: "mixed",     color: "#5F5E5A", bg: "#F1EFE8" },
};

const CHUNK_COLORS = {
  text:  "#378ADD",
  table: "#EF9F27",
  chart: "#7F77DD",
};

const SUGGESTED = [
  "Why do 90% of clinical drug development programs fail?",
  "What does the STAR diagram show about drug classes?",
  "What are the dosage recommendations for this drug?",
  "What are the main adverse reactions reported?",
];

export default function Chat({ activeDoc, messages, setMessages, onQuery }) {
  const [input, setInput]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [openTrace, setOpenTrace] = useState(null);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(query) {
    const q = query || input.trim();
    if (!q || loading) return;
    setInput("");
    setOpenTrace(null);

    const userMsg = { role: "user", content: q, id: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const result = await onQuery(q);
      setMessages(prev => [...prev, {
        role: "assistant",
        id: Date.now() + 1,
        content: result.answer,
        query_type: result.query_type,
        citations: result.citations,
        reasoning_trace: result.reasoning_trace,
      }]);
    } catch (e) {
      const isTimeout = e.name === "AbortError";
      setMessages(prev => [...prev, {
        role: "assistant",
        id: Date.now() + 1,
        content: isTimeout
          ? "Request timed out after 30 seconds. The document may be large — try a more specific question."
          : "Backend unreachable. Make sure the API server is running on port 8000.",
        query_type: "",
        citations: [],
        reasoning_trace: "",
        error: true,
      }]);
    
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <main className="chat-area">
      <div className="chat-header">
        <div className="chat-title">Clinical document Q&A</div>
        {activeDoc && (
          <div className="chat-scope">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{marginRight: 5}}>
              <rect x="1" y="1" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1"/>
              <path d="M3 4h6M3 6h4M3 8h5" stroke="currentColor" strokeWidth="1" strokeLinecap="round"/>
            </svg>
            {activeDoc.replace(".pdf", "")}
          </div>
        )}
      </div>

      <div className="messages">
        {isEmpty && (
          <div className="empty-state">
            <div className="empty-icon">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <rect width="32" height="32" rx="8" fill="#E6F1FB"/>
                <path d="M8 10h16M8 14h12M8 18h10M8 22h14" stroke="#185FA5" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="empty-title">
              {activeDoc ? `Ask about ${activeDoc.replace(".pdf","")}` : "Select a document to get started"}
            </div>
            <div className="empty-sub">
              Questions are scoped to the selected document. Citations show exact page numbers.
            </div>
            <div className="suggestions">
              {SUGGESTED.map((s, i) => (
                <button key={i} className="suggestion-btn" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`msg msg-${msg.role}`}>
            <div className="msg-avatar">
              {msg.role === "user" ? "You" : "AI"}
            </div>
            <div className="msg-body">
              {msg.role === "assistant" && msg.query_type && (
                <div
                  className="query-type-badge"
                  style={{
                    background: QUERY_TYPE_LABELS[msg.query_type]?.bg || "#F1EFE8",
                    color: QUERY_TYPE_LABELS[msg.query_type]?.color || "#5F5E5A",
                  }}
                >
                  {msg.query_type} query
                </div>
              )}

              <div className={`msg-bubble ${msg.error ? "bubble-error" : ""}`}>
                {msg.content.split("\n").map((line, i) => (
                  <span key={i}>{line}{i < msg.content.split("\n").length - 1 && <br/>}</span>
                ))}
              </div>

              {msg.citations?.length > 0 && (
                <div className="citations">
                  {msg.citations.slice(0, 4).map((c, i) => (
                    <div key={i} className="citation">
                      <div className="citation-dot" style={{ background: CHUNK_COLORS[c.chunk_type] || "#888" }}/>
                      <span className="citation-type">{c.chunk_type}</span>
                      <span className="citation-sep">·</span>
                      <span>{c.source_file.replace(".pdf","")}</span>
                      <span className="citation-sep">·</span>
                      <span>p.{c.page_number}</span>
                    </div>
                  ))}
                </div>
              )}

              {msg.reasoning_trace && (
                <div className="trace-wrap">
                  <button
                    className="trace-toggle"
                    onClick={() => setOpenTrace(openTrace === msg.id ? null : msg.id)}
                  >
                    {openTrace === msg.id ? "hide" : "show"} reasoning trace
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{marginLeft: 4}}>
                      <path
                        d={openTrace === msg.id ? "M2 6l3-3 3 3" : "M2 4l3 3 3-3"}
                        stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"
                      />
                    </svg>
                  </button>
                  {openTrace === msg.id && (
                    <div className="trace-body">
                      {msg.reasoning_trace.trim().split("\n").map((line, i) => (
                        <div key={i}>{line}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="msg msg-assistant">
            <div className="msg-avatar">AI</div>
            <div className="msg-body">
              <div className="msg-bubble loading-bubble">
                <div className="dot-pulse">
                  <div /><div /><div />
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="input-row">
        <textarea
          ref={inputRef}
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={activeDoc ? `Ask about ${activeDoc.replace(".pdf","")}...` : "Select a document first..."}
          rows={1}
          disabled={loading || !activeDoc}
        />
        <button
          className={`send-btn ${(!input.trim() || loading || !activeDoc) ? "send-disabled" : ""}`}
          onClick={() => send()}
          disabled={!input.trim() || loading || !activeDoc}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M2 8h12M10 4l4 4-4 4" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    </main>
  );
}