import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

export default function Sidebar({ documents, activeDoc, onSelect, onUpload, uploading, uploadError }) {

  const onDrop = useCallback(accepted => {
    if (accepted[0]) onUpload(accepted[0]);
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
    disabled: uploading,
  });

  const totalChunks = docs => docs.text_count + docs.table_count + docs.chart_count;

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <rect width="20" height="20" rx="5" fill="#185FA5"/>
            <path d="M5 7h10M5 10h7M5 13h8" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          <span>ClinicalDoc AI</span>
        </div>
        <div className="sidebar-sub">{documents.length} document{documents.length !== 1 ? "s" : ""} indexed</div>
      </div>

      <div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? "drag-active" : ""} ${uploading ? "uploading" : ""}`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <>
            <div className="upload-spinner" />
            <div className="upload-text">Processing PDF...</div>
            <div className="upload-sub">Extracting text, tables, charts</div>
          </>
        ) : (
          <>
            <div className="upload-icon">
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                <path d="M9 12V4M6 7l3-3 3 3" stroke="#185FA5" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M3 14h12" stroke="#185FA5" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="upload-text">
              {isDragActive ? "Drop PDF here" : "Upload PDF"}
            </div>
            <div className="upload-sub">Click or drag to select</div>
          </>
        )}
      </div>

      {uploadError && (
        <div className="upload-error">{uploadError}</div>
      )}

      <div className="doc-section-label">Documents</div>

      <div className="doc-list">
        {documents.length === 0 && (
          <div className="doc-empty">No documents yet. Upload a PDF to get started.</div>
        )}
        {documents.map(doc => (
          <div
            key={doc.filename}
            className={`doc-item ${activeDoc === doc.filename ? "active" : ""}`}
            onClick={() => onSelect(doc.filename)}
          >
            <div className="doc-icon">PDF</div>
            <div className="doc-info">
              <div className="doc-name" title={doc.filename}>
                {doc.filename.replace(".pdf", "")}
              </div>
              <div className="doc-meta">{totalChunks(doc)} chunks</div>
              <div className="doc-badges">
                {doc.text_count > 0 && (
                  <span className="badge badge-text">{doc.text_count} text</span>
                )}
                {doc.table_count > 0 && (
                  <span className="badge badge-table">{doc.table_count} tables</span>
                )}
                {doc.chart_count > 0 && (
                  <span className="badge badge-chart">{doc.chart_count} charts</span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="footer-row">
          <div className="footer-dot" style={{background: "#639922"}} />
          <span>Backend connected</span>
        </div>
      </div>
    </aside>
  );
}