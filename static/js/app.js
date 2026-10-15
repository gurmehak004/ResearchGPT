// ResearchGPT Frontend Logic - Clean Professional Console

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const fileList = document.getElementById("file-list");
const docCount = document.getElementById("doc-count");
const totalChunksStat = document.getElementById("total-chunks-stat");
const resetBtn = document.getElementById("reset-btn");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const chatMessages = document.getElementById("chat-messages");
const headerStatus = document.getElementById("header-status");
const systemStatusIndicator = document.getElementById("system-status-indicator");
const systemStatusText = document.getElementById("system-status-text");

// Initial Status Fetch
document.addEventListener("DOMContentLoaded", fetchStatus);

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    updateDocPanel(data);
  } catch (err) {
    console.error("Failed to fetch status:", err);
  }
}

function updateDocPanel(data) {
  const files = data.files || [];
  const totalChunks = data.total_chunks || 0;
  const paperChunks = data.paper_chunks || {};

  docCount.textContent = files.length;
  totalChunksStat.textContent = totalChunks;

  if (files.length > 0) {
    fileList.innerHTML = files.map(file => {
      const chunks = paperChunks[file] || 0;
      return `
        <div class="file-row">
          <span class="file-title" title="${file}">${escapeHtml(file)}</span>
          <span class="chunk-tag">${chunks} sections</span>
        </div>
      `;
    }).join("");

    chatInput.disabled = false;
    sendBtn.disabled = false;
    headerStatus.textContent = `${files.length} paper(s) loaded. Ask your first question below.`;
    systemStatusIndicator.classList.add("active");
    systemStatusText.textContent = "Ready";
  } else {
    fileList.innerHTML = `<div class="empty-state">No papers loaded yet. Upload PDFs above to begin.</div>`;
    chatInput.disabled = true;
    sendBtn.disabled = true;
    headerStatus.textContent = "Upload papers on the left to get started.";
    systemStatusIndicator.classList.remove("active");
    systemStatusText.textContent = "Idle";
  }
}

// Upload Handling
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFileUpload(e.dataTransfer.files);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileUpload(e.target.files);
  }
});

async function handleFileUpload(files) {
  const formData = new FormData();
  let validPdfCount = 0;

  for (let file of files) {
    if (file.name.toLowerCase().endsWith(".pdf")) {
      formData.append("files", file);
      validPdfCount++;
    }
  }

  if (validPdfCount === 0) {
    showStatus("Invalid file format. Only PDF files are supported.", "error");
    return;
  }

  showStatus(`Uploading ${validPdfCount} PDF document(s)…`, "info");
  systemStatusIndicator.classList.remove("active");
  systemStatusText.textContent = "Indexing...";

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    const contentType = res.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      throw new Error(`Upload was not accepted (HTTP ${res.status}). Try a smaller PDF and check the Render logs.`);
    }
    const result = await res.json();
    if (!res.ok) {
      throw new Error(result.error || "Failed to process files");
    }

    showStatus("Upload received. Extracting text and building the index…", "info");
    const completed = await waitForUpload(result.job_id);
    showStatus(completed.message, "success");
    await fetchStatus();
  } catch (err) {
    showStatus(`Processing error: ${err.message}`, "error");
    systemStatusText.textContent = "Error";
  }
}

async function waitForUpload(jobId) {
  while (true) {
    await new Promise(resolve => setTimeout(resolve, 2000));
    const res = await fetch(`/api/upload/${encodeURIComponent(jobId)}`);
    const contentType = res.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      throw new Error(`Upload status unavailable (HTTP ${res.status}). The server may have restarted; check Render logs.`);
    }
    const job = await res.json();
    if (!res.ok) {
      throw new Error(job.error || "Could not check upload progress");
    }
    if (job.status === "completed") return job.result;
    if (job.status === "failed") throw new Error(job.error || "PDF processing failed");
  }
}

function showStatus(msg, type) {
  uploadStatus.textContent = msg;
  uploadStatus.className = `status-box ${type}`;
  uploadStatus.classList.remove("hidden");
  if (type === "success") {
    setTimeout(() => {
      uploadStatus.classList.add("hidden");
    }, 5000);
  }
}

// Reset Handler
resetBtn.addEventListener("click", async () => {
  if (confirm("Reset current retrieval index and remove all loaded documents?")) {
    try {
      await fetch("/api/reset", { method: "POST" });
      fetchStatus();
      appendMessage("assistant", "All papers have been cleared. Upload new PDFs to continue.");
    } catch (e) {
      console.error(e);
    }
  }
});

// Chat Handling
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit"));
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;

  appendMessage("user", query);
  chatInput.value = "";
  chatInput.style.height = "auto";

  const loadingId = appendLoadingMessage();
  systemStatusText.textContent = "Thinking...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    // Safely parse response — guard against HTML proxy error pages
    let data;
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      data = await res.json();
    } else {
      const text = await res.text();
      console.error("Non-JSON chat response:", text.slice(0, 200));
      removeLoadingMessage(loadingId);
      appendMessage("assistant", "The server took too long to respond. Please try your query again.");
      return;
    }

    removeLoadingMessage(loadingId);

    if (!res.ok) {
      appendMessage("assistant", `Error: ${data.error || "Failed to generate answer"}`);
      return;
    }

    appendAssistantResponse(data);
  } catch (err) {
    removeLoadingMessage(loadingId);
    appendMessage("assistant", `Network error: ${err.message}`);
  } finally {
    systemStatusText.textContent = "Ready";
  }
});

function appendMessage(role, content) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;

  const tag = role === "user" ? "USER" : "ASSISTANT";
  const parsedContent = role === "assistant" ? marked.parse(content) : escapeHtml(content);

  msgDiv.innerHTML = `
    <div class="message-tag">${tag}</div>
    <div class="message-content">${parsedContent}</div>
  `;

  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendAssistantResponse(data) {
  const msgDiv = document.createElement("div");
  msgDiv.className = "message assistant";

  const parsedAnswer = marked.parse(data.answer);
  const confPct = Math.round((data.confidence || 0) * 100);

  const uniqueSources = data.unique_sources || [];
  const sourcesBadges = uniqueSources.map(s => `<span class="meta-chip source">${escapeHtml(s)}</span>`).join(" ");

  let citationsHtml = "";
  if (data.sources && data.sources.length > 0) {
    const items = data.sources.map(s => `
      <div class="citation-block">
        <div class="citation-source">${escapeHtml(s.source)}</div>
        <div class="citation-snippet">"${escapeHtml(s.snippet)}"</div>
      </div>
    `).join("");

    citationsHtml = `
      <div class="citation-panel">
        <button class="citation-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
          [View ${data.sources.length} Context Chunks]
        </button>
        <div class="citation-list hidden">${items}</div>
      </div>
    `;
  }

  msgDiv.innerHTML = `
    <div class="message-tag">ASSISTANT</div>
    <div class="message-content">
      ${parsedAnswer}
      <div class="meta-row">
        <span class="meta-chip green">Confidence: ${confPct}%</span>
        <span class="meta-chip">Latency: ${data.latency}s</span>
        ${sourcesBadges}
      </div>
      ${citationsHtml}
    </div>
  `;

  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendLoadingMessage() {
  const id = "loading-" + Date.now();
  const msgDiv = document.createElement("div");
  msgDiv.id = id;
  msgDiv.className = "message assistant";
  msgDiv.innerHTML = `
    <div class="message-tag">THINKING</div>
    <div class="message-content">
      <em>Reading your papers and generating a cited answer&hellip;</em>
    </div>
  `;
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return id;
}

function removeLoadingMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
