import { useState, useEffect } from "react";
import Sidebar from "./Sidebar";
import Chat from "./Chat";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const [documents, setDocuments]     = useState([]);
  const [activeDoc, setActiveDoc]     = useState(null);
  const [uploading, setUploading]     = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [chatHistories, setChatHistories] = useState({});

  useEffect(() => { fetchDocuments(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function fetchDocuments() {
    try {
      const res  = await fetch(`${API}/documents`);
      const data = await res.json();
      setDocuments(data);
      if (data.length > 0 && !activeDoc) setActiveDoc(data[0].filename);
    } catch (e) {
      console.error("Failed to load documents:", e);
    }
  }

  async function handleUpload(file) {
    setUploading(true);
    setUploadError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res  = await fetch(`${API}/ingest`, { method: "POST", body: form });
      const data = await res.json();
      if (data.status === "skipped") {
        setUploadError(`${file.name} is already indexed.`);
      } else {
        setActiveDoc(data.filename);
      }
      await fetchDocuments();
    } catch (e) {
      setUploadError("Upload failed. Is the backend running?");
    } finally {
      setUploading(false);
    }
  }

  async function handleQuery(query) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000); // 30s timeout
    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, source_file: activeDoc || "" }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error("Query failed");
      return res.json();
    } finally {
      clearTimeout(timeout);
    }
  }

  const activeMessages = chatHistories[activeDoc] || [];

  function setActiveMessages(updater) {
    setChatHistories(prev => ({
      ...prev,
      [activeDoc]: typeof updater === "function"
        ? updater(prev[activeDoc] || [])
        : updater,
    }));
  }

  function handleSelectDoc(filename) {
    setActiveDoc(filename);
    setUploadError("");
  }

  return (
    <div className="app-shell">
      <Sidebar
        documents={documents}
        activeDoc={activeDoc}
        onSelect={handleSelectDoc}
        onUpload={handleUpload}
        uploading={uploading}
        uploadError={uploadError}
      />
      <Chat
        key={activeDoc}
        activeDoc={activeDoc}
        messages={activeMessages}
        setMessages={setActiveMessages}
        onQuery={handleQuery}
      />
    </div>
  );
}