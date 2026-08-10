"use client";

import { ChangeEvent, FormEvent, ReactNode, useEffect, useRef, useState } from "react";

type DocumentItem = {
  source: string;
  name: string;
  meta: string;
  kind: "pdf" | "md" | "txt" | "xlsx";
  category: string;
  active: boolean;
};

type DocumentResponse = {
  source: string;
  name: string;
  kind: string;
  category?: string;
  chunks: number;
  updated_at: string;
};

type ChatItem = {
  id: number;
  title: string;
  date: string;
  exchanges: Exchange[];
};

type SourceItem = {
  number: number;
  document: string;
  chunk: number;
  score: number;
  content: string;
};

type Exchange = {
  id: number;
  question: string;
  answer: string;
  sources: SourceItem[];
  elapsedSeconds: number;
};

type AskResponse = {
  answer: string;
  sources: SourceItem[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_LOCAL_RAG_API_URL ?? "http://127.0.0.1:8000";
const CHAT_STORAGE_KEY = "nivora-chat-history-v1";

const suggestions = [
  "How does Corrective RAG work?",
  "How does RAGAS measure faithfulness?",
  "Compare agentic and traditional RAG.",
];

function renderAnswerText(
  text: string,
  sources: SourceItem[],
  onSource: (sourceNumber: number) => void,
): ReactNode {
  return text.split(/\n{2,}/).filter(Boolean).map((paragraph, paragraphIndex) => (
    <p key={`${paragraphIndex}-${paragraph.slice(0, 24)}`}>
      {paragraph.split(/(\[\d+\])/g).filter(Boolean).map((part, partIndex) => {
        const match = part.match(/^\[(\d+)]$/);
        const sourceNumber = match ? Number(match[1]) : null;
        if (sourceNumber && sources.some((source) => source.number === sourceNumber)) {
          return (
            <button
              type="button"
              className="citation"
              key={`${part}-${partIndex}`}
              onClick={() => onSource(sourceNumber)}
            >
              {part}
            </button>
          );
        }
        return <span key={`${partIndex}-${part.slice(0, 12)}`}>{part}</span>;
      })}
    </p>
  ));
}

function toDocumentItem(document: DocumentResponse): DocumentItem {
  const kind = document.kind.toLowerCase();
  const category = document.category?.trim() || "RAG Research";
  return {
    source: document.source,
    name: document.name,
    meta: `${document.chunks} ${document.chunks === 1 ? "chunk" : "chunks"}`,
    kind: kind === "pdf" || kind === "txt" || kind === "xlsx" ? kind : "md",
    category,
    active: true,
  };
}

function chatTitle(question: string): string {
  const clean = question.trim().replace(/\s+/g, " ");
  return clean.length > 42 ? `${clean.slice(0, 39)}…` : clean;
}

export default function Home() {
  const [chats, setChats] = useState<ChatItem[]>([]);
  const [activeChatId, setActiveChatId] = useState<number | null>(null);
  const [chatHistoryLoaded, setChatHistoryLoaded] = useState(false);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [activeSources, setActiveSources] = useState<SourceItem[]>([]);
  const [selectedSourceNumber, setSelectedSourceNumber] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [notice, setNotice] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState("qwen2.5-1.5b");
  const [selectedLanguage, setSelectedLanguage] = useState("Auto-detect");
  const [openDropdown, setOpenDropdown] = useState<"model" | "language" | null>(null);
  const [answerLength, setAnswerLength] = useState("Normal");
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({
    "RAG Research": true,
  });
  const [categoryDialogOpen, setCategoryDialogOpen] = useState(false);
  const [categoryName, setCategoryName] = useState("RAG Research");
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadCategoryRef = useRef("RAG Research");
  const selectedSource = activeSources.find((source) => source.number === selectedSourceNumber) ?? null;
  const categories = Array.from(
    documents.reduce((groups, document) => {
      const category = document.category?.trim() || "RAG Research";
      const group = groups.get(category) ?? [];
      group.push(document);
      groups.set(category, group);
      return groups;
    }, new Map<string, DocumentItem[]>())
  ).sort(([left], [right]) => String(left).localeCompare(String(right)));
  const existingCategories = categories.map(([category]) => category);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(CHAT_STORAGE_KEY);
      if (!stored) return;
      const parsed = JSON.parse(stored) as ChatItem[];
      if (!Array.isArray(parsed)) return;
      const restored = parsed.map((chat) => ({
        ...chat,
        exchanges: Array.isArray(chat.exchanges) ? chat.exchanges : [],
      }));
      setChats(restored);
      const firstChat = restored[0];
      if (firstChat) {
        setActiveChatId(firstChat.id);
        setExchanges(firstChat.exchanges);
        const lastExchange = firstChat.exchanges.at(-1);
        setActiveSources(lastExchange?.sources ?? []);
        setSelectedSourceNumber(lastExchange?.sources[0]?.number ?? null);
      }
    } catch {
      window.localStorage.removeItem(CHAT_STORAGE_KEY);
    } finally {
      setChatHistoryLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!chatHistoryLoaded) return;
    window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chats.slice(0, 30)));
  }, [chats, chatHistoryLoaded]);

  useEffect(() => {
    let cancelled = false;
    async function loadDocuments() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/documents`);
        if (!response.ok) throw new Error(`Request failed with status ${response.status}.`);
        const result = await response.json() as DocumentResponse[];
        if (!cancelled) setDocuments(result.map(toDocumentItem));
      } catch (error) {
        if (!cancelled) {
          setNotice(
            error instanceof Error
              ? `Documents could not be loaded: ${error.message}`
              : "Documents could not be loaded."
          );
        }
      }
    }
    void loadDocuments();
    return () => {
      cancelled = true;
    };
  }, []);

  function showSource(sources: SourceItem[], sourceNumber: number) {
    setActiveSources(sources);
    setSelectedSourceNumber(sourceNumber);
    setSourcesOpen(true);
  }

  async function submitQuestion(event: FormEvent) {
    event.preventDefault();
    const question = query.trim();
    const chatIdAtSubmit = activeChatId;
    if (!question || isAsking) return;
    const selectedSources = documents
      .filter((document) => document.active)
      .map((document) => document.source);
    if (selectedSources.length === 0) {
      setNotice("Select at least one document to search.");
      return;
    }
    setIsAsking(true);
    setNotice("");
    setQuery("");
    const startedAt = performance.now();
    try {
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 2, sources: selectedSources }),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(error?.detail ?? `Request failed with status ${response.status}.`);
      }
      const result = await response.json() as AskResponse;
      const exchange: Exchange = {
        id: Date.now(),
        question,
        answer: result.answer,
        sources: result.sources,
        elapsedSeconds: (performance.now() - startedAt) / 1000,
      };
      setExchanges((current) => [...current, exchange]);
      if (chatIdAtSubmit === null) {
        const newChat: ChatItem = {
          id: exchange.id,
          title: chatTitle(question),
          date: "Today",
          exchanges: [exchange],
        };
        setChats((current) => [newChat, ...current].slice(0, 30));
        setActiveChatId(newChat.id);
      } else {
        setChats((current) =>
          current.map((chat) =>
            chat.id === chatIdAtSubmit
              ? { ...chat, exchanges: [...chat.exchanges, exchange] }
              : chat
          )
        );
      }
      if (result.sources.length > 0) {
        setActiveSources(result.sources);
        setSelectedSourceNumber(result.sources[0].number);
      }
    } catch (error) {
      setQuery(question);
      setNotice(
        error instanceof Error
          ? `The local RAG service is unavailable: ${error.message}`
          : "The local RAG service is unavailable."
      );
    } finally {
      setIsAsking(false);
    }
  }

  async function uploadDocument(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || isUploading) return;
    setIsUploading(true);
    setNotice(`Indexing ${file.name} locally…`);
    const body = new FormData();
    body.append("file", file);
    body.append("category", uploadCategoryRef.current);
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents`, { method: "POST", body });
      if (!response.ok) {
        const error = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(error?.detail ?? `Request failed with status ${response.status}.`);
      }
      const result = await response.json() as DocumentResponse;
      setDocuments((current) => [
        toDocumentItem(result),
        ...current.filter((document) => document.source !== result.source),
      ]);
      const resultCategory = result.category?.trim() || "RAG Research";
      setOpenCategories((current) => ({ ...current, [resultCategory]: true }));
      setNotice(`${result.name} was indexed locally in ${result.chunks} chunks.`);
    } catch (error) {
      setNotice(error instanceof Error ? `Upload failed: ${error.message}` : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  function chooseUploadCategory(event: FormEvent) {
    event.preventDefault();
    const cleanCategory = categoryName.trim().replace(/\s+/g, " ");
    if (!cleanCategory) return;
    uploadCategoryRef.current = cleanCategory;
    setCategoryName(cleanCategory);
    setCategoryDialogOpen(false);
    fileInputRef.current?.click();
  }

  async function deleteDocument(document: DocumentItem) {
    try {
      const encodedSource = document.source.split("/").map(encodeURIComponent).join("/");
      const response = await fetch(`${API_BASE_URL}/api/documents/${encodedSource}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const error = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(error?.detail ?? `Request failed with status ${response.status}.`);
      }
      setDocuments((current) => current.filter((item) => item.source !== document.source));
      setNotice(`${document.name} was removed from the local index.`);
    } catch (error) {
      setNotice(error instanceof Error ? `Delete failed: ${error.message}` : "Delete failed.");
    }
  }

  function deleteChat(id: number) {
    const chat = chats.find((item) => item.id === id);
    const remainingChats = chats.filter((item) => item.id !== id);
    setChats(remainingChats);
    if (activeChatId === id) {
      const nextChat = remainingChats[0];
      setActiveChatId(nextChat?.id ?? null);
      setExchanges(nextChat?.exchanges ?? []);
      const lastExchange = nextChat?.exchanges.at(-1);
      setActiveSources(lastExchange?.sources ?? []);
      setSelectedSourceNumber(lastExchange?.sources[0]?.number ?? null);
    }
    if (chat) setNotice(`“${chat.title}” was removed from chat history.`);
  }

  function startNewChat() {
    setActiveChatId(null);
    setExchanges([]);
    setActiveSources([]);
    setSelectedSourceNumber(null);
    setSourcesOpen(false);
    setQuery("");
    setNotice("A new chat was started.");
  }

  function openChat(chat: ChatItem) {
    setActiveChatId(chat.id);
    setExchanges(chat.exchanges);
    const lastExchange = chat.exchanges.at(-1);
    setActiveSources(lastExchange?.sources ?? []);
    setSelectedSourceNumber(lastExchange?.sources[0]?.number ?? null);
    setSourcesOpen(false);
    setNotice("");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <img src="/assistant-icon.png" alt="" className="brand-mark" />
          <div>
            <strong>Nivora</strong>
            <span>Ask Nivi. Find it locally.</span>
          </div>
        </div>
        <div className="top-actions">
          <button className="model-button" onClick={() => setSettingsOpen(true)}>
            {selectedModel} <span>⌄</span>
          </button>
        </div>
      </header>

      <div className={`workspace ${libraryOpen ? "left-open" : ""} ${sourcesOpen ? "right-open" : ""}`}>
        {libraryOpen && <aside className="library-panel">
          <button
            className="sidebar-toggle inside-toggle left-inside direction-left"
            type="button"
            onClick={() => setLibraryOpen(false)}
            aria-label="Close left sidebar"
            title="Close left sidebar"
          >
            <img src="/sidebar-toggle.svg" alt="" />
          </button>
          <section className="sidebar-section chat-history-section">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Chat History</span>
              </div>
              <button className="small-icon" aria-label="Start a new chat" onClick={startNewChat}>＋</button>
            </div>
            <nav className="chat-history" aria-label="Chat history">
              {chats.map((chat) => (
                <div className={`history-row ${activeChatId === chat.id ? "is-current" : ""}`} key={chat.id}>
                  <button
                    type="button"
                    className="history-main"
                    onClick={() => openChat(chat)}
                    aria-label={`Open ${chat.title}`}
                  >
                    <span className="history-icon">◌</span>
                    <span className="history-copy"><b>{chat.title}</b><time>{chat.date}</time></span>
                  </button>
                  <button
                    type="button"
                    className="delete-chat"
                    aria-label={`Delete ${chat.title}`}
                    title={`Delete ${chat.title}`}
                    onClick={() => deleteChat(chat.id)}
                  >
                    ×
                  </button>
                </div>
              ))}
              {chatHistoryLoaded && chats.length === 0 && (
                <p className="empty-history">Your saved chats will appear here.</p>
              )}
            </nav>
          </section>

          <section className="sidebar-section documents-section">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">My Documents</span>
              </div>
              <input
                ref={fileInputRef}
                className="file-input"
                type="file"
                accept=".pdf,.md,.txt,.xlsx"
                onChange={uploadDocument}
              />
              <button
                className="heading-upload"
                aria-label="Add file"
                disabled={isUploading}
                onClick={() => {
                  setCategoryName(existingCategories[0] ?? "RAG Research");
                  setCategoryDialogOpen(true);
                }}
              >
                {isUploading ? "…" : "＋"}
              </button>
            </div>

            <div className="document-groups">
              {categories.map(([category, categoryDocuments]) => {
                const isOpen = openCategories[category] ?? true;
                const allCategoryDocumentsSelected = categoryDocuments.every(
                  (document) => document.active
                );
                const someCategoryDocumentsSelected = categoryDocuments.some(
                  (document) => document.active
                );
                return (
                  <section className="document-category" key={category}>
                    <div className="collection-row">
                      <span>
                        <b>{category}</b>
                        <small>{categoryDocuments.length} {categoryDocuments.length === 1 ? "document" : "documents"}</small>
                      </span>
                      <span className="category-actions">
                        <label className="category-select-all">
                          <input
                            type="checkbox"
                            checked={allCategoryDocumentsSelected}
                            ref={(element) => {
                              if (element) {
                                element.indeterminate =
                                  someCategoryDocumentsSelected && !allCategoryDocumentsSelected;
                              }
                            }}
                            onChange={() =>
                              setDocuments((current) =>
                                current.map((document) =>
                                  document.category === category
                                    ? { ...document, active: !allCategoryDocumentsSelected }
                                    : document
                                )
                              )
                            }
                          />
                          <small>Select all</small>
                        </label>
                        <button
                          className={`category-toggle ${isOpen ? "is-open" : ""}`}
                          type="button"
                          aria-label={isOpen ? `Collapse ${category}` : `Expand ${category}`}
                          aria-expanded={isOpen}
                          onClick={() =>
                            setOpenCategories((current) => ({
                              ...current,
                              [category]: !isOpen,
                            }))
                          }
                        >
                          <img src="/category-dropdown.svg" alt="" />
                        </button>
                      </span>
                    </div>

                    {isOpen && <div className="document-list">
                      {categoryDocuments.map((document) => (
                  <div className={`document-row ${document.active ? "is-active" : ""}`} key={document.source}>
                    <input
                      type="checkbox"
                      checked={document.active}
                      aria-label={`Use ${document.name} in search`}
                      onChange={() =>
                        setDocuments((current) =>
                          current.map((item) =>
                            item.source === document.source ? { ...item, active: !item.active } : item
                          )
                        )
                      }
                    />
                    <span className={`file-type ${document.kind}`}>{document.kind.toUpperCase()}</span>
                    <span className="document-copy"><b>{document.name}</b><small>{document.meta}</small></span>
                    <span className="ready-dot" aria-label={document.active ? "Active" : "Inactive"} />
                    <button
                      type="button"
                      className="delete-document"
                      aria-label={`Delete ${document.name}`}
                      title={`Delete ${document.name}`}
                      onClick={() => void deleteDocument(document)}
                    >
                      ×
                    </button>
                  </div>
                      ))}
                    </div>}
                  </section>
                );
              })}
            </div>

          </section>
        </aside>}

        <section className="chat-panel">
          {!libraryOpen && (
            <button
              className="sidebar-toggle collapsed-toggle left-collapsed direction-right"
              type="button"
              onClick={() => setLibraryOpen(true)}
              aria-label="Open left sidebar"
              title="Open left sidebar"
            >
              <img src="/sidebar-toggle.svg" alt="" />
            </button>
          )}
          {!sourcesOpen && (
            <button
              className="sidebar-toggle collapsed-toggle right-collapsed direction-left"
              type="button"
              onClick={() => setSourcesOpen(true)}
              aria-label="Open sources panel"
              title="Open sources panel"
            >
              <img src="/sidebar-toggle.svg" alt="" />
            </button>
          )}
          <div className="chat-content">
            <div className="assistant-message">
              <img src="/assistant-icon.png" alt="Nivi chatbot" />
              <div className="message-bubble welcome">
                <strong>Hello, I&apos;m Nivi.</strong>
                <p>I search your selected documents locally and support my answers with verifiable sources. What would you like to explore today?</p>
              </div>
            </div>

            <div className="suggestions">
              {suggestions.map((suggestion) => (
                <button key={suggestion} onClick={() => setQuery(suggestion)}>{suggestion}</button>
              ))}
            </div>

            {exchanges.map((exchange) => (
              <div className="exchange" key={exchange.id}>
                <div className="user-message">{exchange.question}</div>
                <div className="assistant-message">
                  <img src="/assistant-icon.png" alt="Nivi chatbot" />
                  <div className="message-bubble answer">
                    <div className="answer-label">
                      <span>Source-grounded answer</span>
                      <small>{exchange.elapsedSeconds.toFixed(1)} sec</small>
                    </div>
                    {renderAnswerText(
                      exchange.answer,
                      exchange.sources,
                      (sourceNumber) => showSource(exchange.sources, sourceNumber),
                    )}
                    <div className="message-actions">
                      <button onClick={() => navigator.clipboard.writeText(exchange.answer).then(() => setNotice("The answer was copied to the clipboard."))}>Copy</button>
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {isAsking && (
              <div className="assistant-message is-thinking" aria-live="polite">
                <img src="/assistant-icon.png" alt="Nivi chatbot" />
                <div className="message-bubble answer">
                  <div className="answer-label"><span>Searching local documents</span><small>Please wait</small></div>
                  <p className="thinking-line"><span /><span /><span /></p>
                </div>
              </div>
            )}
          </div>

          <form className="composer" onSubmit={submitQuestion}>
            {notice && <div className="demo-notice">{notice}<button type="button" onClick={() => setNotice("")}>×</button></div>}
            <div className="composer-box">
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Ask a question about your documents..."
                aria-label="Question"
                rows={1}
              />
              <div className="composer-controls">
                <span>Search the <b>local RAG index</b></span>
                <button type="submit" className="send-button" aria-label="Send question" disabled={isAsking || !query.trim()}>
                  {isAsking ? "Thinking…" : "Send"}
                </button>
              </div>
            </div>
            <small>Answers are generated only from selected documents. Verify important information against the sources.</small>
          </form>
        </section>

        {sourcesOpen && <aside className="source-panel">
          <button
            className="sidebar-toggle inside-toggle right-inside direction-right"
            type="button"
            onClick={() => setSourcesOpen(false)}
            aria-label="Close sources panel"
            title="Close sources panel"
          >
            <img src="/sidebar-toggle.svg" alt="" />
          </button>
          <div className="source-heading">
            <div><span className="eyebrow">Sources</span></div>
            <span className="source-count">{activeSources.length}</span>
          </div>

          <div className="source-content">
          {activeSources.length > 0 ? (
            <>
              <div className="source-tabs" style={{ gridTemplateColumns: `repeat(${activeSources.length}, 1fr)` }}>
                {activeSources.map((source) => (
                  <button
                    className={selectedSourceNumber === source.number ? "active" : ""}
                    key={`${source.document}-${source.number}`}
                    onClick={() => setSelectedSourceNumber(source.number)}
                  >
                    Source {source.number}
                  </button>
                ))}
              </div>
              {selectedSource && (
                <article className="source-card">
                  <div className="source-file">
                    <span className="file-type pdf">DOC</span>
                    <span><b>{selectedSource.document}</b><small>Chunk {selectedSource.chunk}</small></span>
                  </div>
                  <div className="confidence">
                    <span>Relevance</span>
                    <b>{Math.round(selectedSource.score * 100)}%</b>
                    <i><em style={{ width: `${Math.round(selectedSource.score * 100)}%` }} /></i>
                  </div>
                  <blockquote>{selectedSource.content}</blockquote>
                </article>
              )}
            </>
          ) : (
            <div className="empty-sources">
              <strong>No sources selected</strong>
              <p>Ask a question and open a citation to inspect its supporting document passage.</p>
            </div>
          )}
          </div>

        </aside>}
      </div>

      {categoryDialogOpen && (
        <div className="modal-backdrop" onMouseDown={() => setCategoryDialogOpen(false)}>
          <form
            className="settings-modal category-modal"
            onMouseDown={(event) => event.stopPropagation()}
            onSubmit={chooseUploadCategory}
            aria-modal="true"
            role="dialog"
            aria-label="Choose document category"
          >
            <div className="modal-heading">
              <div><span className="eyebrow">Add document</span><h2>Choose a category</h2></div>
              <button type="button" aria-label="Close category dialog" onClick={() => setCategoryDialogOpen(false)}>×</button>
            </div>
            <label className="settings-field">
              <span>Category name</span>
              <input
                autoFocus
                list="document-category-options"
                maxLength={80}
                value={categoryName}
                onChange={(event) => setCategoryName(event.target.value)}
                placeholder="e.g. Course Notes"
              />
            </label>
            <datalist id="document-category-options">
              {existingCategories.map((category) => <option value={category} key={category} />)}
            </datalist>
            <p className="category-help">Select an existing category or enter a new one.</p>
            <div className="category-actions">
              <button type="button" className="cancel-category" onClick={() => setCategoryDialogOpen(false)}>Cancel</button>
              <button type="submit" className="save-settings" disabled={!categoryName.trim()}>Choose file</button>
            </div>
          </form>
        </div>
      )}

      {settingsOpen && (
        <div className="modal-backdrop" onMouseDown={() => { setSettingsOpen(false); setOpenDropdown(null); }}>
          <section className="settings-modal" onMouseDown={(event) => event.stopPropagation()} aria-modal="true" role="dialog" aria-label="Answer settings">
            <div className="modal-heading"><div><span className="eyebrow">Local mode</span><h2>Answer settings</h2></div><button aria-label="Close settings" onClick={() => { setSettingsOpen(false); setOpenDropdown(null); }}>×</button></div>

            <label className="settings-field">
              <span>Model</span>
              <div className={`custom-select ${openDropdown === "model" ? "is-open" : ""}`}>
                <button
                  type="button"
                  className="select-trigger"
                  aria-haspopup="listbox"
                  aria-expanded={openDropdown === "model"}
                  onClick={() => setOpenDropdown((current) => current === "model" ? null : "model")}
                >
                  <span>{selectedModel}</span><i>⌄</i>
                </button>
                {openDropdown === "model" && (
                  <div className="select-menu" role="listbox" aria-label="Model options">
                    {["qwen2.5-1.5b", "qwen2.5-0.5b"].map((model) => (
                      <button
                        type="button"
                        role="option"
                        aria-selected={selectedModel === model}
                        className={selectedModel === model ? "is-selected" : ""}
                        key={model}
                        onClick={() => { setSelectedModel(model); setOpenDropdown(null); }}
                      >
                        <span>{model}</span><b>{selectedModel === model ? "✓" : ""}</b>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </label>

            <label className="settings-field">
              <span>Answer language</span>
              <div className={`custom-select ${openDropdown === "language" ? "is-open" : ""}`}>
                <button
                  type="button"
                  className="select-trigger"
                  aria-haspopup="listbox"
                  aria-expanded={openDropdown === "language"}
                  onClick={() => setOpenDropdown((current) => current === "language" ? null : "language")}
                >
                  <span>{selectedLanguage}</span><i>⌄</i>
                </button>
                {openDropdown === "language" && (
                  <div className="select-menu" role="listbox" aria-label="Answer language options">
                    {["Auto-detect", "English", "Turkish"].map((language) => (
                      <button
                        type="button"
                        role="option"
                        aria-selected={selectedLanguage === language}
                        className={selectedLanguage === language ? "is-selected" : ""}
                        key={language}
                        onClick={() => { setSelectedLanguage(language); setOpenDropdown(null); }}
                      >
                        <span>{language}</span><b>{selectedLanguage === language ? "✓" : ""}</b>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </label>
            <fieldset><legend>Answer length</legend>{["Short", "Normal", "Detailed"].map((item) => <button key={item} className={answerLength === item ? "active" : ""} onClick={() => setAnswerLength(item)}>{item}</button>)}</fieldset>
            <div className="privacy-note"><b>Privacy protection is on</b><p>Documents and questions never leave this device.</p></div>
            <button className="save-settings" onClick={() => { setSettingsOpen(false); setOpenDropdown(null); setNotice("Demo settings were updated."); }}>Apply settings</button>
          </section>
        </div>
      )}
    </main>
  );
}
