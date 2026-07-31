/**
 * popup.js
 * --------
 * 1. Reads the video ID from the current YouTube tab.
 * 2. Pulls top-level comments via the YouTube Data API (GCP).
 * 3. Sends them to the Flask backend's /predict_batch endpoint.
 * 4. Renders summary stats + two Chart.js charts.
 */

const DEFAULTS = {
  backendUrl: "http://localhost:5001",
  maxComments: 200,
};

const SENTIMENT_COLORS = {
  positive: "#3ddc97",
  neutral: "#8b93a7",
  negative: "#ff5c72",
};

// ---------------------------------------------------------------------
// Element references
// ---------------------------------------------------------------------
const el = {
  settingsToggle: document.getElementById("settingsToggle"),
  settingsPanel: document.getElementById("settingsPanel"),
  apiKeyInput: document.getElementById("apiKeyInput"),
  backendUrlInput: document.getElementById("backendUrlInput"),
  maxCommentsInput: document.getElementById("maxCommentsInput"),
  saveSettingsBtn: document.getElementById("saveSettingsBtn"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusText: document.getElementById("statusText"),
  results: document.getElementById("results"),
  pulsePositive: document.getElementById("pulsePositive"),
  pulseNeutral: document.getElementById("pulseNeutral"),
  pulseNegative: document.getElementById("pulseNegative"),
  pctPositive: document.getElementById("pctPositive"),
  pctNeutral: document.getElementById("pctNeutral"),
  pctNegative: document.getElementById("pctNegative"),
  statTotalComments: document.getElementById("statTotalComments"),
  statTotalWords: document.getElementById("statTotalWords"),
  statPositive: document.getElementById("statPositive"),
  statNegative: document.getElementById("statNegative"),
  statNeutral: document.getElementById("statNeutral"),
  statAvgWords: document.getElementById("statAvgWords"),
  timeChartCanvas: document.getElementById("timeChart"),
  wordsChartCanvas: document.getElementById("wordsChart"),
};

let timeChart = null;
let wordsChart = null;

// ---------------------------------------------------------------------
// Settings: load / save (chrome.storage.local)
// ---------------------------------------------------------------------
function loadSettings() {
  chrome.storage.local.get(["apiKey", "backendUrl", "maxComments"], (stored) => {
    el.apiKeyInput.value = stored.apiKey || "";
    el.backendUrlInput.value = stored.backendUrl || DEFAULTS.backendUrl;
    el.maxCommentsInput.value = stored.maxComments || DEFAULTS.maxComments;
  });
}

function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["apiKey", "backendUrl", "maxComments"], (stored) => {
      resolve({
        apiKey: stored.apiKey || "",
        backendUrl: (stored.backendUrl || DEFAULTS.backendUrl).replace(/\/+$/, ""),
        maxComments: Number(stored.maxComments) || DEFAULTS.maxComments,
      });
    });
  });
}

el.settingsToggle.addEventListener("click", () => {
  const isHidden = el.settingsPanel.hasAttribute("hidden");
  if (isHidden) {
    el.settingsPanel.removeAttribute("hidden");
  } else {
    el.settingsPanel.setAttribute("hidden", "");
  }
  el.settingsToggle.setAttribute("aria-expanded", String(isHidden));
});

el.saveSettingsBtn.addEventListener("click", () => {
  const apiKey = el.apiKeyInput.value.trim();
  const backendUrl = el.backendUrlInput.value.trim() || DEFAULTS.backendUrl;
  const maxComments = Number(el.maxCommentsInput.value) || DEFAULTS.maxComments;

  chrome.storage.local.set({ apiKey, backendUrl, maxComments }, () => {
    setStatus("Settings saved.", "ok");
  });
});

// ---------------------------------------------------------------------
// Status helper
// ---------------------------------------------------------------------
function setStatus(message, kind) {
  el.statusText.textContent = message;
  el.statusText.classList.remove("status--error", "status--ok");
  if (kind === "error") el.statusText.classList.add("status--error");
  if (kind === "ok") el.statusText.classList.add("status--ok");
}

// ---------------------------------------------------------------------
// Video ID extraction from the active tab's URL
// ---------------------------------------------------------------------
function extractVideoId(url) {
  if (!url) return null;
  const match = url.match(
    /(?:youtube\.com\/(?:watch\?v=|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/
  );
  return match ? match[1] : null;
}

function getActiveTabUrl() {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (chrome.runtime.lastError || !tabs || !tabs[0]) {
        reject(new Error("Could not read the active tab."));
        return;
      }
      resolve(tabs[0].url);
    });
  });
}

// ---------------------------------------------------------------------
// YouTube Data API: fetch top-level comments (paginated)
// ---------------------------------------------------------------------
async function fetchComments(videoId, apiKey, maxComments) {
  const comments = [];
  let pageToken = "";

  while (comments.length < maxComments) {
    const url = new URL("https://www.googleapis.com/youtube/v3/commentThreads");
    url.searchParams.set("part", "snippet");
    url.searchParams.set("videoId", videoId);
    url.searchParams.set("key", apiKey);
    url.searchParams.set("maxResults", "100");
    url.searchParams.set("order", "time");
    url.searchParams.set("textFormat", "plainText");
    if (pageToken) url.searchParams.set("pageToken", pageToken);

    const response = await fetch(url.toString());
    const data = await response.json();

    if (!response.ok) {
      const reason = data?.error?.errors?.[0]?.reason;
      if (reason === "commentsDisabled") {
        throw new Error("Comments are disabled on this video.");
      }
      if (reason === "quotaExceeded") {
        throw new Error("YouTube API quota exceeded for this key.");
      }
      throw new Error(data?.error?.message || "YouTube API request failed.");
    }

    for (const item of data.items || []) {
      const snippet = item.snippet.topLevelComment.snippet;
      comments.push({
        text: snippet.textOriginal,
        timestamp: snippet.publishedAt,
      });
      if (comments.length >= maxComments) break;
    }

    pageToken = data.nextPageToken;
    if (!pageToken) break;
  }

  return comments;
}

// ---------------------------------------------------------------------
// Backend call
// ---------------------------------------------------------------------
async function predictSentiments(backendUrl, comments) {
  const response = await fetch(`${backendUrl}/predict_batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentences: comments }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error || "Backend request failed.");
  }
  return data.results || [];
}

// ---------------------------------------------------------------------
// Stats + rendering
// ---------------------------------------------------------------------
function countWords(text) {
  return (text || "").trim().split(/\s+/).filter(Boolean).length;
}

function renderPulseAndStats(results) {
  const total = results.length;
  const counts = { positive: 0, neutral: 0, negative: 0 };
  let totalWords = 0;

  for (const r of results) {
    if (counts[r.sentiment] !== undefined) counts[r.sentiment] += 1;
    totalWords += countWords(r.text);
  }

  const pct = (n) => (total ? Math.round((n / total) * 100) : 0);

  el.statTotalComments.textContent = total;
  el.statTotalWords.textContent = totalWords.toLocaleString();
  el.statPositive.textContent = counts.positive;
  el.statNegative.textContent = counts.negative;
  el.statNeutral.textContent = counts.neutral;
  el.statAvgWords.textContent = total ? Math.round(totalWords / total) : 0;

  el.pctPositive.textContent = `${pct(counts.positive)}%`;
  el.pctNeutral.textContent = `${pct(counts.neutral)}%`;
  el.pctNegative.textContent = `${pct(counts.negative)}%`;

  // Reset to 0 first so the width transition animates on reveal.
  el.pulsePositive.style.width = "0%";
  el.pulseNeutral.style.width = "0%";
  el.pulseNegative.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      el.pulsePositive.style.width = `${pct(counts.positive)}%`;
      el.pulseNeutral.style.width = `${pct(counts.neutral)}%`;
      el.pulseNegative.style.width = `${pct(counts.negative)}%`;
    });
  });
}

function dayKey(isoString) {
  return (isoString || "").slice(0, 10); // YYYY-MM-DD
}

function renderTimeChart(results) {
  const byDay = new Map();
  for (const r of results) {
    const key = dayKey(r.timestamp) || "unknown";
    if (!byDay.has(key)) byDay.set(key, { positive: 0, neutral: 0, negative: 0 });
    const bucket = byDay.get(key);
    if (bucket[r.sentiment] !== undefined) bucket[r.sentiment] += 1;
  }

  const days = [...byDay.keys()].sort();
  const datasetFor = (sentiment) => days.map((d) => byDay.get(d)[sentiment]);

  if (timeChart) timeChart.destroy();
  timeChart = new Chart(el.timeChartCanvas, {
    type: "line",
    data: {
      labels: days,
      datasets: [
        {
          label: "Positive",
          data: datasetFor("positive"),
          borderColor: SENTIMENT_COLORS.positive,
          backgroundColor: "rgba(61,220,151,0.15)",
          fill: true,
          tension: 0.35,
          pointRadius: 0,
        },
        {
          label: "Neutral",
          data: datasetFor("neutral"),
          borderColor: SENTIMENT_COLORS.neutral,
          backgroundColor: "rgba(139,147,167,0.1)",
          fill: true,
          tension: 0.35,
          pointRadius: 0,
        },
        {
          label: "Negative",
          data: datasetFor("negative"),
          borderColor: SENTIMENT_COLORS.negative,
          backgroundColor: "rgba(255,92,114,0.15)",
          fill: true,
          tension: 0.35,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#9aa0ac", boxWidth: 10, font: { size: 10.5 } } },
      },
      scales: {
        x: { ticks: { color: "#6b7180", font: { size: 9.5 } }, grid: { color: "#232733" } },
        y: {
          beginAtZero: true,
          ticks: { color: "#6b7180", font: { size: 9.5 }, precision: 0 },
          grid: { color: "#232733" },
        },
      },
    },
  });
}

const EXTRA_STOPWORDS = new Set([
  "im", "youre", "dont", "didnt", "doesnt", "cant", "isnt", "wasnt",
  "video", "comment", "comments", "youtube", "like", "just", "really",
]);

function renderWordsChart(results) {
  const freq = new Map();
  for (const r of results) {
    const words = (r.cleaned_text || "").split(/\s+/).filter(Boolean);
    for (const word of words) {
      if (word.length < 3 || EXTRA_STOPWORDS.has(word)) continue;
      freq.set(word, (freq.get(word) || 0) + 1);
    }
  }

  const top = [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);

  if (wordsChart) wordsChart.destroy();
  wordsChart = new Chart(el.wordsChartCanvas, {
    type: "bar",
    data: {
      labels: top.map(([word]) => word),
      datasets: [
        {
          data: top.map(([, count]) => count),
          backgroundColor: "#f2b705",
          borderRadius: 6,
          maxBarThickness: 18,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { color: "#6b7180", font: { size: 9.5 }, precision: 0 },
          grid: { color: "#232733" },
        },
        y: { ticks: { color: "#9aa0ac", font: { size: 11 } }, grid: { display: false } },
      },
    },
  });
}

// ---------------------------------------------------------------------
// Main flow
// ---------------------------------------------------------------------
el.analyzeBtn.addEventListener("click", async () => {
  el.analyzeBtn.disabled = true;
  el.results.setAttribute("hidden", "");

  try {
    const { apiKey, backendUrl, maxComments } = await getSettings();

    if (!apiKey) {
      setStatus("Add your YouTube Data API key in settings first.", "error");
      el.settingsPanel.removeAttribute("hidden");
      return;
    }

    setStatus("Finding the video...");
    const tabUrl = await getActiveTabUrl();
    const videoId = extractVideoId(tabUrl);
    if (!videoId) {
      setStatus("Open a YouTube video, then try again.", "error");
      return;
    }

    setStatus("Reading comments...");
    const comments = await fetchComments(videoId, apiKey, maxComments);
    if (!comments.length) {
      setStatus("No comments found on this video.", "error");
      return;
    }

    setStatus(`Scoring ${comments.length} comments...`);
    const results = await predictSentiments(backendUrl, comments);
    if (!results.length) {
      setStatus("The backend returned no results.", "error");
      return;
    }

    renderPulseAndStats(results);
    renderTimeChart(results);
    renderWordsChart(results);

    el.results.removeAttribute("hidden");
    setStatus(`Done — analyzed ${results.length} comments.`, "ok");
  } catch (err) {
    setStatus(err.message || "Something went wrong.", "error");
  } finally {
    el.analyzeBtn.disabled = false;
  }
});

loadSettings();
