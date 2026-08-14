"use strict";

// DOM Elements
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseButton = document.getElementById("browseButton");
const uploadContent = document.getElementById("uploadContent");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeButton = document.getElementById("removeButton");
const dropOverlay = document.getElementById("dropOverlay");

const analyzeButton = document.getElementById("analyzeButton");
const buttonText = document.getElementById("buttonText");
const loadingSpinner = document.getElementById("loadingSpinner");

const errorMessage = document.getElementById("errorMessage");
const toastContainer = document.getElementById("toastContainer");

const results = document.getElementById("results");
const newAnalysisButton = document.getElementById("newAnalysisButton");

const probabilityRing = document.getElementById("probabilityRing");
const aiProbability = document.getElementById("aiProbability");
const humanProbability = document.getElementById("humanProbability");
const humanBar = document.getElementById("humanBar");
const confidenceBadge = document.getElementById("confidenceBadge");

const resultFilename = document.getElementById("resultFilename");
const resultFormat = document.getElementById("resultFormat");
const resultDimensions = document.getElementById("resultDimensions");
const resultFileSize = document.getElementById("resultFileSize");

const modelPredictions = document.getElementById("modelPredictions");
const signalsList = document.getElementById("signalsList");
const metadataList = document.getElementById("metadataList");
const colorSwatches = document.getElementById("colorSwatches");
const resultNote = document.getElementById("resultNote");

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
let selectedFile = null;
let objectUrl = null;
let abortController = null;

// ==================== File Selection ====================
browseButton.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", (e) => {
    if (e.target === removeButton || e.target === browseButton) return;
    fileInput.click();
});
dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });

// Drag & Drop
["dragenter", "dragover"].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropZone.classList.add("drag-over");
        dropOverlay.classList.remove("hidden");
    });
});
["dragleave", "drop"].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropZone.classList.remove("drag-over");
        dropOverlay.classList.add("hidden");
    });
});
dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

// ==================== File Handling ====================
function handleFile(file) {
    clearError();
    if (!file.type.startsWith("image/")) {
        showError("Please select a valid image file.");
        return;
    }
    if (file.size > MAX_FILE_SIZE) {
        showError("File is too large. Maximum allowed size is 10 MB.");
        return;
    }
    selectedFile = file;
    displayPreview(file);
    analyzeButton.disabled = false;
    results.classList.add("hidden");
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
}

function displayPreview(file) {
    imagePreview.src = objectUrl;
    imagePreview.alt = `Preview of ${file.name}`;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    uploadContent.classList.add("hidden");
    previewContainer.classList.remove("hidden");
}

removeButton.addEventListener("click", (e) => {
    e.stopPropagation();
    resetFileSelection();
});

function resetFileSelection() {
    if (abortController) abortController.abort();
    selectedFile = null;
    fileInput.value = "";
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    imagePreview.src = "";
    uploadContent.classList.remove("hidden");
    previewContainer.classList.add("hidden");
    analyzeButton.disabled = true;
    results.classList.add("hidden");
    clearError();
    document.title = "ImageGuard – AI Authenticity Detector";
}

// ==================== Analysis ====================
analyzeButton.addEventListener("click", analyzeImage);

// Keyboard shortcut
document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!analyzeButton.disabled) analyzeImage();
    }
});

async function analyzeImage() {
    if (!selectedFile) return;

    if (abortController) abortController.abort();
    abortController = new AbortController();

    setLoading(true);
    clearError();
    document.title = "Analyzing... | ImageGuard";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData,
            signal: abortController.signal
        });

        if (!response.ok) {
            let message = `Analysis failed with HTTP ${response.status}.`;
            try {
                const errorData = await response.json();
                if (errorData.detail) message = errorData.detail;
            } catch {}
            throw new Error(message);
        }

        const data = await response.json();
        renderResults(data);
        results.classList.remove("hidden");
        setTimeout(() => results.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
        showToast("Analysis complete", "success");
    } catch (error) {
        if (error.name === "AbortError") {
            showToast("Analysis cancelled", "error");
        } else {
            console.error("Analysis error:", error);
            showError(error.message || "Unable to analyze the image. Please try again.");
        }
    } finally {
        setLoading(false);
        document.title = "ImageGuard – AI Authenticity Detector";
        abortController = null;
    }
}

function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;
    loadingSpinner.classList.toggle("hidden", !isLoading);
    if (isLoading) buttonText.textContent = "Analyzing...";
    else {
        buttonText.textContent = "Analyze Image";
        analyzeButton.disabled = !selectedFile;
    }
}

// ==================== Render Results ====================
function renderResults(data) {
    renderProbability(data);
    renderFileInfo(data);
    renderModels(data.model_predictions);
    renderSignals(data.signals);
    renderMetadata(data.metadata_summary);
    renderColors(data.dominant_colors);
    renderHumanProbability(data.human_probability);
    resultNote.textContent = data.note || "This result is probabilistic and should not be treated as definitive proof.";
}

function renderProbability(data) {
    const prob = data.ai_probability;
    if (prob === null || prob === undefined || Number.isNaN(Number(prob))) {
        aiProbability.textContent = "N/A";
        probabilityRing.style.setProperty("--percentage", "0%");
        probabilityRing.style.setProperty("--gauge-color", "var(--accent)");
    } else {
        const percentage = clamp(Number(prob) * 100, 0, 100);
        aiProbability.textContent = `${percentage.toFixed(1)}%`;
        probabilityRing.style.setProperty("--percentage", `${percentage}%`);
        probabilityRing.style.setProperty("--gauge-color", getProbabilityColor(percentage));
    }
    renderConfidence(data.confidence);
}

function getProbabilityColor(p) {
    if (p > 80) return "var(--danger)";
    if (p >= 60) return "var(--warning)";
    if (p >= 40) return "var(--uncertain)";
    return "var(--real)";
}

function renderConfidence(conf) {
    confidenceBadge.className = "confidence-badge";
    if (!conf) { confidenceBadge.textContent = "UNKNOWN"; return; }
    const normalized = String(conf).toUpperCase();
    confidenceBadge.textContent = normalized;
    if (normalized === "HIGH") confidenceBadge.classList.add("confidence-high");
    else if (normalized === "MEDIUM") confidenceBadge.classList.add("confidence-medium");
    else if (normalized === "LOW") confidenceBadge.classList.add("confidence-low");
}

function renderFileInfo(data) {
    resultFilename.textContent = data.filename || "—";
    resultFormat.textContent = data.format || "—";
    resultDimensions.textContent = (data.width && data.height) ? `${data.width} × ${data.height}` : "—";
    resultFileSize.textContent = (data.file_size_bytes != null) ? formatBytes(data.file_size_bytes) : "—";
}

function renderModels(predictions) {
    modelPredictions.innerHTML = "";
    if (!predictions || Object.keys(predictions).length === 0) {
        modelPredictions.innerHTML = '<div class="signal-item"><span class="signal-icon">i</span><span>No model predictions available.</span></div>';
        return;
    }
    for (const [model, value] of Object.entries(predictions)) {
        const pct = clamp(Number(value) * 100, 0, 100);
        const displayName = model === "ateeq" ? "Ateeqq" : model === "wkaandemir" ? "Wkaandemir" : model;
        const card = document.createElement("div");
        card.className = "model-card";
        card.innerHTML = `
            <div class="model-top">
                <span class="model-name">${escapeHTML(displayName)}</span>
                <span class="model-value">${pct.toFixed(1)}%</span>
            </div>
            <div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct.toFixed(1)}">
                <div class="progress-bar" style="width: ${pct}%"></div>
            </div>`;
        modelPredictions.appendChild(card);
    }
}

function renderSignals(signals) {
    signalsList.innerHTML = "";
    if (!Array.isArray(signals) || signals.length === 0) {
        signalsList.innerHTML = '<div class="signal-item"><span class="signal-icon">i</span><span>No specific signals reported.</span></div>';
        return;
    }
    signals.forEach(signal => {
        const item = document.createElement("div");
        item.className = "signal-item";
        item.innerHTML = `<span class="signal-icon">✓</span><span>${escapeHTML(signal)}</span>`;
        signalsList.appendChild(item);
    });
}

function renderMetadata(metadata) {
    metadataList.innerHTML = "";
    if (!metadata || Object.keys(metadata).length === 0) {
        metadataList.innerHTML = '<div class="signal-item"><span class="signal-icon">i</span><span>No metadata available.</span></div>';
        return;
    }
    for (const [key, value] of Object.entries(metadata)) {
        const row = document.createElement("div");
        row.className = "metadata-row";
        row.innerHTML = `<span class="metadata-key">${escapeHTML(key)}</span><span class="metadata-value">${escapeHTML(value)}</span>`;
        metadataList.appendChild(row);
    }
}

function renderColors(colors) {
    colorSwatches.innerHTML = "";
    if (!Array.isArray(colors) || colors.length === 0) {
        colorSwatches.innerHTML = '<div class="signal-item"><span class="signal-icon">i</span><span>No dominant colour data available.</span></div>';
        return;
    }
    colors.forEach(color => {
        const safe = normalizeHexColor(color);
        const item = document.createElement("div");
        item.className = "color-swatch";
        item.innerHTML = `<span class="swatch" style="background-color:${safe}" title="${safe}"></span><span class="color-code">${safe}</span>`;
        colorSwatches.appendChild(item);
    });
}

function renderHumanProbability(val) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) {
        humanProbability.textContent = "N/A";
        humanBar.style.width = "0%";
        return;
    }
    const pct = clamp(Number(val) * 100, 0, 100);
    humanProbability.textContent = `${pct.toFixed(1)}%`;
    humanBar.style.width = `${pct}%`;
}

newAnalysisButton.addEventListener("click", () => {
    resetFileSelection();
    window.scrollTo({ top: 0, behavior: "smooth" });
});

// ==================== Helpers ====================
function clamp(v, min, max) { return Math.min(Math.max(v, min), max); }
function formatBytes(bytes) {
    const n = Number(bytes);
    if (!Number.isFinite(n) || n < 0) return "Unknown";
    if (n < 1024) return `${n} B`;
    if (n < 1024*1024) return `${(n/1024).toFixed(1)} KB`;
    if (n < 1024*1024*1024) return `${(n/(1024*1024)).toFixed(2)} MB`;
    return `${(n/(1024*1024*1024)).toFixed(2)} GB`;
}
function normalizeHexColor(color) {
    const val = String(color || "").trim();
    return val.match(/^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/) ? val : "#808080";
}
function escapeHTML(str) {
    return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;");
}
function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove("hidden");
}
function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}
function showToast(msg, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast ${type === "error" ? "error" : "success"}`;
    toast.textContent = msg;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}