const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const typing = document.getElementById("typing");

const modalBackdrop = document.getElementById("modalBackdrop");
const modalContent = document.getElementById("modalContent");
const modalClose = document.getElementById("modalClose");

let lastResults = [];
let ctx = { needs_more_info: false, preferences: {} };

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function setTyping(on) {
  typing.classList.toggle("hidden", !on);
  if (on) scrollToBottom();
}

function el(tag, className, html) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function addMessage(role, text, results = null) {
  const isUser = role === "user";

  const wrap = el("div", "flex gap-3 my-3");
  wrap.classList.add("animate-pop");

  const avatar = el(
    "div",
    "h-9 w-9 rounded-2xl grid place-items-center font-bold border border-white/10 flex-none " +
      (isUser
        ? "bg-gradient-to-br from-fuchsia-500/25 via-purple-500/20 to-pink-500/20 text-white/90"
        : "bg-white/5 text-white/70")
  );
  avatar.textContent = isUser ? "U" : "C";

  const bubble = el(
    "div",
    "max-w-[980px] rounded-2xl border border-white/10 px-4 py-3 " +
      "shadow-[0_25px_80px_rgba(0,0,0,0.35)] " +
      (isUser
        ? "bg-gradient-to-br from-fuchsia-500/15 via-purple-500/10 to-pink-500/10"
        : "bg-white/5")
  );
  bubble.style.whiteSpace = "pre-wrap";
  bubble.textContent = text;

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chat.appendChild(wrap);

  if (results && results.length) {
    lastResults = results;

    // ✅ auto-rows-min + items-start prevents stretching
    const cards = el("div", "mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3 items-start auto-rows-min");

    results.forEach((r) => {
      // ✅ h-fit + self-start prevents tall cards
      const card = el(
        "div",
        "rounded-2xl border border-white/10 bg-black/30 hover:bg-white/5 transition cursor-pointer p-4 " +
          "hover:-translate-y-0.5 active:scale-[0.99] " +
          "h-fit self-start flex flex-col gap-3"
      );

      card.innerHTML = `
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <div class="font-semibold text-sm truncate">
              ${r.name} <span class="text-white/50">(${r.housekeeper_id})</span>
            </div>
            <div class="text-xs text-white/65 mt-1">
              ${r.gender}, ${r.age} • ${r.city} • ${r.experience_years} yrs exp
            </div>
          </div>

          <div class="shrink-0 text-[11px] px-3 py-1.5 rounded-full border border-white/10
                      bg-gradient-to-r from-fuchsia-500/20 via-purple-500/20 to-pink-500/20 text-white/85">
            ${r.match_score}% fit
          </div>
        </div>

        <div class="text-xs text-white/75 space-y-1">
          <div class="truncate">
            <span class="text-white/50">Languages:</span> ${r.languages.join(", ")}
          </div>

          <div class="line-clamp-2">
            <span class="text-white/50">Skills:</span> ${r.skills.join(", ")}
          </div>
        </div>

        <div class="text-xs text-white/55">
          Tap to view packages →
        </div>
      `;

      card.addEventListener("click", () => openProfile(r.housekeeper_id));
      cards.appendChild(card);
    });

    bubble.appendChild(cards);
  }

  scrollToBottom();
}

function openProfile(housekeeperId) {
  const r = lastResults.find(x => x.housekeeper_id === housekeeperId);
  if (!r) return;

  modalContent.innerHTML = `
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <div class="text-xl font-semibold">${r.name}</div>
        <div class="text-sm text-white/70 mt-1">
          ${r.gender}, ${r.age} • ${r.city} • ${r.experience_years} yrs experience
        </div>
      </div>

      <div class="shrink-0 text-xs px-3 py-1.5 rounded-full border border-white/10
                  bg-gradient-to-r from-fuchsia-500/25 via-purple-500/25 to-pink-500/25 text-white/85">
        Compatibility: ${r.match_score}%
      </div>
    </div>

    <div class="mt-4 flex flex-wrap gap-2 text-xs">
      <span class="px-3 py-1.5 rounded-full border border-white/10 bg-white/5 text-white/85">
        <b>Languages:</b> ${r.languages.join(", ")}
      </span>
      <span class="px-3 py-1.5 rounded-full border border-white/10 bg-white/5 text-white/85">
        <b>Skills:</b> ${r.skills.join(", ")}
      </span>
      <span class="px-3 py-1.5 rounded-full border border-white/10 bg-white/5 text-white/85">
        <b>Package:</b> ${r.package_type}
      </span>
      <span class="px-3 py-1.5 rounded-full border border-white/10 bg-white/5 text-white/85">
        <b>Base:</b> ₱${r.base_price}
      </span>
    </div>

    <div class="mt-5">
      <div class="text-sm font-semibold">Packages</div>
      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        ${r.packages.map(p => `
          <div class="rounded-2xl border border-white/10 bg-black/40 p-4">
            <div class="font-semibold text-sm">${p.name}</div>
            <div class="text-sm text-emerald-200/90 mt-1">₱${p.price}</div>
            <div class="text-xs text-white/60 mt-2">
              Demo only — in the real app this proceeds to direct hire.
            </div>
          </div>
        `).join("")}
      </div>
    </div>
  `;

  modalBackdrop.classList.remove("hidden");
}

function closeModal() {
  modalBackdrop.classList.add("hidden");
}

modalClose.addEventListener("click", closeModal);

// click outside modal closes
modalBackdrop.addEventListener("click", (e) => {
  const modal = document.getElementById("modal");
  if (!modal.contains(e.target)) closeModal();
});

// Esc closes
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !modalBackdrop.classList.contains("hidden")) {
    closeModal();
  }
});

// Sidebar chips
document.querySelectorAll(".chip").forEach(btn => {
  btn.addEventListener("click", () => {
    input.value = btn.dataset.prompt || "";
    input.focus();
  });
});

async function sendMessage(text) {
  addMessage("user", text);
  setTyping(true);

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ message: text, context: ctx })
  });

  const data = await res.json();
  setTyping(false);

  ctx.needs_more_info = !!data.needs_more_info;
  ctx.preferences = data.preferences || ctx.preferences || {};
  if (data.results && data.results.length) ctx.needs_more_info = false;

  addMessage("system", data.reply, data.results || []);
}

// initial message
addMessage("system", "Hi! I’m Casaligan Assistant 🤝 Tell me what you need and I’ll match you with available housekeepers.");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.focus();
  await sendMessage(text);
});
