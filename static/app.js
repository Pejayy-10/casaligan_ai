const chatEl = document.getElementById("chat");
const msgEl = document.getElementById("msg");
const sendBtn = document.getElementById("send");
const statusLine = document.getElementById("status-line");

const resultsEl = document.getElementById("results");
const resultsSub = document.getElementById("results-sub");
const pillCount = document.getElementById("pill-count");

const btnClear = document.getElementById("btn-clear");
const btnExamples = document.getElementById("btn-examples");
const tipsPanel = document.getElementById("tips-panel");
const btnCloseTips = document.getElementById("btn-close-tips");

const overlay = document.getElementById("modal-overlay");
const modalClose = document.getElementById("modal-close");
const modalTitle = document.getElementById("modal-title");
const modalSub = document.getElementById("modal-sub");
const modalBody = document.getElementById("modal-body");

let context = {
  preferences: {},
  needs_more_info: false,
  last_intent: ""
};

function escapeHtml(s) {
  return (s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function scrollChatToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setStatus(text) {
  statusLine.textContent = text;
}

function bubble(role, text) {
  const row = document.createElement("div");
  row.className = `row ${role}`;

  const b = document.createElement("div");
  b.className = `bubble ${role === "user" ? "user" : "assistant"}`;

  // Keep the assistant more natural: no forced templates in UI.
  // If backend returns multi-lines, show them nicely.
  const safe = escapeHtml(text).replaceAll("\n", "<br/>");
  b.innerHTML = safe;

  row.appendChild(b);
  chatEl.appendChild(row);
  scrollChatToBottom();
}

function typingIndicator(show) {
  const id = "typing-indicator";
  const existing = document.getElementById(id);

  if (!show) {
    if (existing) existing.remove();
    return;
  }
  if (existing) return;

  const row = document.createElement("div");
  row.id = id;
  row.className = "row assistant";

  const b = document.createElement("div");
  b.className = "bubble assistant";
  b.innerHTML = `
    <span class="typing">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </span>
  `;

  row.appendChild(b);
  chatEl.appendChild(row);
  scrollChatToBottom();
}

function formatProfileSub(r) {
  const bits = [];
  bits.push(`${r.gender}, ${r.age}`);
  bits.push(r.city);
  bits.push(`${r.experience_years} yr exp`);
  return bits.join(" • ");
}

function renderResults(results) {
  if (!results || results.length === 0) {
    resultsSub.textContent = "No results";
    pillCount.classList.add("hidden");
    resultsEl.innerHTML = `<div class="text-sm text-white/60">No recommendations yet.</div>`;
    return;
  }

  resultsSub.textContent = "Click a card to view packages";
  pillCount.textContent = `${results.length} shown`;
  pillCount.classList.remove("hidden");

  const cards = results.map((r) => {
    const skills = (r.skills || []).slice(0, 4).map(s => `<span class="badge">${escapeHtml(s)}</span>`).join(" ");
    const langs = (r.languages || []).slice(0, 3).map(s => `<span class="badge">${escapeHtml(s)}</span>`).join(" ");

    return `
      <button class="card w-full text-left" data-id="${escapeHtml(String(r.housekeeper_id))}">
        <div class="flex items-start justify-between gap-3">
          <div>
            <div class="text-sm font-semibold">${escapeHtml(r.name)}</div>
            <div class="mt-1 text-xs text-white/60">${escapeHtml(formatProfileSub(r))}</div>
          </div>
          <div class="text-right">
            <div class="text-xs text-white/60">Match</div>
            <div class="text-sm font-bold">${escapeHtml(String(r.match_score))}%</div>
          </div>
        </div>

        <div class="mt-3 flex flex-wrap gap-2">
          ${skills || `<span class="badge">no skills listed</span>`}
        </div>

        <div class="mt-2 flex flex-wrap gap-2 opacity-90">
          ${langs || `<span class="badge">no languages listed</span>`}
        </div>

        <div class="mt-3 text-xs text-white/60">
          Base price: <span class="text-white/80">${escapeHtml(String(r.base_price))}</span> • Package: <span class="text-white/80">${escapeHtml(String(r.package_type))}</span>
        </div>
      </button>
    `;
  }).join("");

  resultsEl.innerHTML = `<div class="grid gap-3">${cards}</div>`;

  // card click
  resultsEl.querySelectorAll("[data-id]").forEach(btn => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-id");
      const chosen = results.find(x => String(x.housekeeper_id) === String(id));
      if (chosen) openModal(chosen);
    });
  });
}

function openModal(r) {
  modalTitle.textContent = r.name;
  modalSub.textContent = formatProfileSub(r);

  const skills = (r.skills || []).map(s => `<span class="badge">${escapeHtml(s)}</span>`).join(" ");
  const langs = (r.languages || []).map(s => `<span class="badge">${escapeHtml(s)}</span>`).join(" ");

  const packages = (r.packages || []).map(p => `
    <div class="card">
      <div class="flex items-start justify-between gap-3">
        <div class="font-semibold text-sm">${escapeHtml(p.name)}</div>
        <div class="text-sm font-bold">${escapeHtml(String(p.price))}</div>
      </div>
      <div class="mt-1 text-xs text-white/60">Tap “Hire” in the full system (demo only)</div>
    </div>
  `).join("");

  modalBody.innerHTML = `
    <div class="grid gap-4">
      <div>
        <div class="text-sm font-semibold">Skills</div>
        <div class="mt-2 flex flex-wrap gap-2">${skills || `<span class="badge">none</span>`}</div>
      </div>

      <div>
        <div class="text-sm font-semibold">Languages</div>
        <div class="mt-2 flex flex-wrap gap-2">${langs || `<span class="badge">none</span>`}</div>
      </div>

      <div class="grid gap-2">
        <div class="text-sm font-semibold">Packages</div>
        <div class="grid gap-3">
          ${packages || `<div class="text-sm text-white/60">No packages available.</div>`}
        </div>
      </div>

      <div class="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/70">
        <div class="font-semibold text-white">Quick note</div>
        <div class="mt-1">
          Your match score is based on how many of your constraints were satisfied (skills, language, budget, dates, etc.).
        </div>
      </div>
    </div>
  `;

  overlay.classList.remove("hidden");
}

function closeModal() {
  overlay.classList.add("hidden");
}

overlay.addEventListener("click", (e) => {
  // click outside modal closes
  if (e.target === overlay) closeModal();
});
modalClose.addEventListener("click", closeModal);
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !overlay.classList.contains("hidden")) closeModal();
});

async function sendMessage() {
  const text = msgEl.value.trim();
  if (!text) return;

  bubble("user", text);
  msgEl.value = "";
  msgEl.focus();

  setStatus("Thinking…");
  typingIndicator(true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        context: context
      })
    });

    const data = await res.json();

    typingIndicator(false);
    setStatus("Ready");

    // keep assistant natural (no UI-injected templates)
    if (data.reply) bubble("assistant", data.reply);

    // store context for follow-up handling
    context = {
      preferences: data.preferences || {},
      needs_more_info: !!data.needs_more_info,
      last_intent: data.last_intent || data.intent || ""
    };

    // show results if any
    if (Array.isArray(data.results) && data.results.length) {
      renderResults(data.results);
    } else {
      // If no results, keep previous cards? (we clear only when intent is RECOMMEND and no results)
      if ((data.intent || "") === "RECOMMEND") {
        renderResults([]);
      }
    }

  } catch (err) {
    typingIndicator(false);
    setStatus("Error");
    bubble("assistant", "Oops—something went wrong. Try again.");
    console.error(err);
  }
}

sendBtn.addEventListener("click", sendMessage);
msgEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

// Clear chat
btnClear.addEventListener("click", () => {
  chatEl.innerHTML = "";
  renderResults([]);
  context = { preferences: {}, needs_more_info: false, last_intent: "" };
  setStatus("Ready");
  bubble("assistant", "Chat cleared. Tell me what you need and I’ll recommend matches.");
});

// Tips panel toggle
btnExamples.addEventListener("click", () => tipsPanel.classList.toggle("hidden"));
btnCloseTips.addEventListener("click", () => tipsPanel.classList.add("hidden"));

// Chips & quick links (fills input only — doesn’t send templates into chat)
document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    msgEl.value = btn.getAttribute("data-chip") || "";
    msgEl.focus();
  });
});
document.querySelectorAll("[data-quick]").forEach((btn) => {
  btn.addEventListener("click", () => {
    msgEl.value = btn.getAttribute("data-quick") || "";
    msgEl.focus();
  });
});

// Initial welcome (short, not templated)
bubble("assistant", "Hi! Tell me what you need (skill + location/date/budget if you have it) and I’ll recommend matches 😊");
renderResults([]);
