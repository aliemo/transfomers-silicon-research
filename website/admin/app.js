const state = {
  items: [],
  selectedId: null,
  meta: null,
  mode: "edit", // edit | new
};

const $ = (id) => document.getElementById(id);

function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 2800);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText || "request failed");
  return data;
}

function formData() {
  const fd = new FormData($("editor"));
  const obj = Object.fromEntries(fd.entries());
  obj.year = Number(obj.year || 0);
  obj.model = String(obj.model || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
  obj.authors = String(obj.authors || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
  if (!obj.pdf || obj.pdf === "False") obj.pdf = false;
  return obj;
}

function renderReviewBox(paper) {
  const box = $("review-box");
  const r = paper.review || {};
  if (!state.meta?.review?.enabled) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  const checklist = (r.checklist || []).map((c) => `<li>${escapeHtml(c)}</li>`).join("");
  box.innerHTML = `
    <div class="review-pill">
      <span>${escapeHtml(r.current_label || paper.review_pass || "pass1")}</span>
      <span>P1=${escapeHtml(String(paper.review_pass1 || r.pass1 || "pending"))}</span>
      <span>P2=${escapeHtml(String(paper.review_pass2 || r.pass2 || "pending"))}</span>
      <span>P3=${escapeHtml(String(paper.review_pass3 || r.pass3 || "pending"))}</span>
    </div>
    <h3>${escapeHtml(r.current_label || "Review")}</h3>
    <p>${escapeHtml(r.description || "")}</p>
    <ul>${checklist || "<li>No checklist configured</li>"}</ul>
  `;
  const done = !!r.done || paper.review_pass === "done";
  $("btn-pass-accept").disabled = done;
  $("btn-pass-reject").disabled = done;
  $("btn-pass-skip").disabled = done;
}

function fillForm(paper) {
  const form = $("editor");
  form.title.value = paper.title || "";
  form.year.value = paper.year || "";
  form.type.value = paper.type || "article";
  form.category.value = paper.category || "other";
  form.review_pass.value = paper.review_pass || "pass1";
  form.review_pass1.value = paper.review_pass1 || "pending";
  form.review_pass2.value = paper.review_pass2 || "pending";
  form.review_pass3.value = paper.review_pass3 || "pending";
  form.ignore.value = String(paper.ignore);
  form.silicon.value = String(paper.silicon);
  form.platform.value = paper.platform || "";
  form.publisher.value = paper.publisher || "";
  form.pubname.value = paper.pubname || "";
  form.authors.value = Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors || "";
  form.model.value = Array.isArray(paper.model) ? paper.model.join(", ") : paper.model || "";
  form.doi.value = paper.doi || "";
  form.url.value = paper.url || "";
  form.pdf.value = paper.pdf === false || paper.pdf === "False" ? "False" : paper.pdf || "False";
  form.review_notes.value = paper.review_notes || "";
  $("paper-id").textContent = paper.id ?? "new";
  $("editor-title").textContent = paper.id ? "Edit paper" : "Add paper";
  renderReviewBox(paper);
}

function showEditor(show) {
  $("empty").classList.toggle("hidden", show);
  $("editor").classList.toggle("hidden", !show);
  $("analysis").classList.add("hidden");
  if (!show) $("review-box").classList.add("hidden");
}

function renderList() {
  const root = $("list");
  root.innerHTML = "";
  state.items.forEach((p) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "item" + (p.id === state.selectedId ? " active" : "");
    const rp = p.review_pass || p.review?.current || "?";
    btn.innerHTML = `<div class="t">${escapeHtml(p.title || "")}</div>
      <div class="m">#${p.id} · ${escapeHtml(String(p.year || ""))} · ${escapeHtml(rp)} · ${escapeHtml(p.category_label || p.category || "")} · ignore=${escapeHtml(String(p.ignore))}</div>`;
    btn.addEventListener("click", () => selectPaper(p.id));
    root.appendChild(btn);
  });
  $("count").textContent = String(state.items.length);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function refreshList() {
  const params = new URLSearchParams();
  const q = $("q").value.trim();
  const ign = $("f-ignore").value;
  const sil = $("f-silicon").value;
  const cat = $("f-category").value;
  const rp = $("f-review-pass").value;
  const rs = $("f-review-status").value;
  if (q) params.set("q", q);
  if (ign) params.set("ignore", ign);
  if (sil) params.set("silicon", sil);
  if (cat) params.set("category", cat);
  if (rp) params.set("review_pass", rp);
  if (rs) params.set("review_status", rs);
  const data = await api(`/api/papers?${params.toString()}`);
  state.items = data.items || [];
  renderList();
}

async function selectPaper(id) {
  state.mode = "edit";
  state.selectedId = id;
  const paper = await api(`/api/papers/${id}`);
  fillForm(paper);
  showEditor(true);
  renderList();
}

function newPaper() {
  state.mode = "new";
  state.selectedId = null;
  const d = state.meta?.review?.defaults || {};
  fillForm({
    id: null,
    title: "",
    year: new Date().getFullYear(),
    type: "article",
    category: "other",
    ignore: "check",
    silicon: "check",
    platform: "__no_data__",
    publisher: "__no_data__",
    pubname: "__no_data__",
    authors: [],
    model: ["Transformer"],
    doi: "__no_data__",
    url: "__no_data__",
    pdf: false,
    review_pass: d.review_pass || "pass1",
    review_pass1: d.review_pass1 || "pending",
    review_pass2: d.review_pass2 || "pending",
    review_pass3: d.review_pass3 || "pending",
    review_notes: "",
    review: {
      current: d.review_pass || "pass1",
      current_label: "Pass 1 — Relevance",
      checklist: (state.meta?.review?.passes || []).find((p) => p.id === "pass1")?.checklist || [],
      description: (state.meta?.review?.passes || []).find((p) => p.id === "pass1")?.description || "",
    },
  });
  showEditor(true);
  renderList();
}

async function savePaper(ev) {
  ev.preventDefault();
  const payload = formData();
  try {
    if (state.mode === "new") {
      const created = await api("/api/papers", { method: "POST", body: JSON.stringify(payload) });
      toast(`Added #${created.id}`);
      state.mode = "edit";
      state.selectedId = created.id;
      fillForm(created);
    } else {
      const updated = await api(`/api/papers/${state.selectedId}`, {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      toast(`Saved #${updated.id}`);
      fillForm(updated);
    }
    await refreshList();
  } catch (err) {
    toast(String(err.message || err));
  }
}

async function decidePass(decision) {
  if (!state.selectedId) {
    toast("Select a paper first");
    return;
  }
  const notes = $("editor").review_notes.value || "";
  try {
    const data = await api(`/api/papers/${state.selectedId}/review`, {
      method: "POST",
      body: JSON.stringify({ decision, notes }),
    });
    fillForm(data.paper);
    toast(`${decision} → ${data.paper.review_pass}`);
    await refreshList();
  } catch (err) {
    toast(String(err.message || err));
  }
}

async function analyze(apply) {
  if (!state.selectedId) {
    toast("Save/select a paper first");
    return;
  }
  try {
    const data = await api(`/api/papers/${state.selectedId}/analyze`, {
      method: "POST",
      body: JSON.stringify({ apply: !!apply }),
    });
    const a = data.analysis || {};
    $("analysis").classList.remove("hidden");
    $("analysis").innerHTML = `<strong>${a.provider || "analyzer"}</strong>
      · related=<code>${a.related}</code>
      · confidence=<code>${a.confidence}</code><br/>
      ${escapeHtml(a.reason || "")}<br/>
      suggest silicon=<code>${a.silicon}</code>
      platform=<code>${a.platform}</code>
      model=<code>${(a.model || []).join(", ")}</code>
      category=<code>${a.category}</code>`;
    if (apply && data.paper) {
      fillForm(data.paper);
      toast("Analysis applied");
      await refreshList();
    } else {
      toast("Analysis ready");
    }
  } catch (err) {
    toast(String(err.message || err));
  }
}

async function softDelete() {
  if (!state.selectedId) return;
  if (!confirm(`Set ignore=True for #${state.selectedId}?`)) return;
  try {
    const data = await api(`/api/papers/${state.selectedId}`, { method: "DELETE" });
    toast(`Ignored #${data.deleted}`);
    await refreshList();
    if (data.paper) fillForm(data.paper);
  } catch (err) {
    toast(String(err.message || err));
  }
}

async function hardDelete() {
  if (!state.selectedId) return;
  if (!confirm(`HARD DELETE #${state.selectedId} from papers.yaml? This cannot be undone easily.`)) return;
  try {
    await api(`/api/papers/${state.selectedId}?hard=1`, { method: "DELETE" });
    toast(`Deleted #${state.selectedId}`);
    state.selectedId = null;
    showEditor(false);
    await refreshList();
  } catch (err) {
    toast(String(err.message || err));
  }
}

async function rebuild() {
  try {
    toast("Rebuilding website…");
    const data = await api("/api/rebuild", { method: "POST", body: "{}" });
    if (!data.ok) throw new Error(data.stderr || "rebuild failed");
    toast("Website rebuilt");
  } catch (err) {
    toast(String(err.message || err));
  }
}

async function init() {
  state.meta = await api("/api/meta");
  const catSel = $("category");
  const fCat = $("f-category");
  (state.meta.categories || []).forEach((c) => {
    const o1 = document.createElement("option");
    o1.value = c.id;
    o1.textContent = c.label;
    catSel.appendChild(o1);
    const o2 = document.createElement("option");
    o2.value = c.id;
    o2.textContent = c.label;
    fCat.appendChild(o2);
  });

  $("q").addEventListener("input", () => refreshList());
  $("f-ignore").addEventListener("change", () => refreshList());
  $("f-silicon").addEventListener("change", () => refreshList());
  $("f-category").addEventListener("change", () => refreshList());
  $("f-review-pass").addEventListener("change", () => refreshList());
  $("f-review-status").addEventListener("change", () => refreshList());
  $("btn-new").addEventListener("click", newPaper);
  $("btn-rebuild").addEventListener("click", rebuild);
  $("btn-analyze").addEventListener("click", () => analyze(false));
  $("btn-analyze-apply").addEventListener("click", () => analyze(true));
  $("btn-soft-delete").addEventListener("click", softDelete);
  $("btn-hard-delete").addEventListener("click", hardDelete);
  $("btn-pass-accept").addEventListener("click", () => decidePass("accepted"));
  $("btn-pass-reject").addEventListener("click", () => decidePass("rejected"));
  $("btn-pass-skip").addEventListener("click", () => decidePass("skipped"));
  $("editor").addEventListener("submit", savePaper);

  await refreshList();
}

init().catch((err) => toast(String(err.message || err)));
