// Shared behaviour for every page: analytics loader, fixed footer, light/dark theme, hover explanations,
// and a custom dropdown that replaces the browser's <select> popup.

// ---------- analytics: only when the server configured it (window.SITE_CONFIG from /config.js) ----------
(function () {
  var c = window.SITE_CONFIG || {};
  if (c.clarityId) {
    (function (w, d, a, r, i, t, y) {
      w[a] = w[a] || function () { (w[a].q = w[a].q || []).push(arguments); };
      t = d.createElement(r); t.async = 1; t.src = "https://www.clarity.ms/tag/" + i;
      y = d.getElementsByTagName(r)[0]; y.parentNode.insertBefore(t, y);
    })(window, document, "clarity", "script", c.clarityId);
  }
  if (c.clarityId || c.formsubmitUrl) {
    var s = document.createElement("script");
    s.defer = true; s.src = "tracker.js";
    document.head.appendChild(s);
  }
  window.trackDemoAction = window.trackDemoAction || function () {};
})();

function onReady(fn) { if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn); else fn(); }

// ---------- fixed footer (one compact line). ----------
(function () {
  var REPO_URL = "https://github.com/nathyBekele/reproducing-gtta";
  var links = [
    ["SA Inspector (live)", "try"],
    ["DG Inspector (saved runs)", "dg"],
    ["Report (PDF)", "report.pdf"],
    ["Hugging Face Space", "https://huggingface.co/spaces/NathyB/gtta-sa-explorer"],
    ["Code", REPO_URL],
    ["GTTA paper", "https://arxiv.org/abs/2511.04847"],
    ["GTTA code", "https://github.com/r2llab/GTTA"],
  ];
  function size() {
    var f = document.querySelector(".sitefoot");
    if (f) document.documentElement.style.setProperty("--foot-h", f.offsetHeight + "px");
    var t = document.querySelector(".topbar");
    if (t) document.documentElement.style.setProperty("--top-h", t.offsetHeight + "px");
  }
  onReady(function () {
    if (document.querySelector(".sitefoot")) return;
    var f = document.createElement("footer"), inner = document.createElement("div");
    f.className = "sitefoot"; inner.className = "in";
    var note = document.createElement("span");
    note.className = "note"; note.textContent = "Independent reproduction of GTTA (ICLR 2026) · not affiliated with the authors";
    var nav = document.createElement("nav");
    nav.setAttribute("aria-label", "Project links");
    links.forEach(function (l) {
      if (!l[1]) return;
      var a = document.createElement("a");
      a.textContent = l[0]; a.href = l[1];
      if (/^https?:/.test(l[1]) || /\.pdf$/.test(l[1])) { a.target = "_blank"; a.rel = "noopener"; }
      nav.appendChild(a);
    });
    inner.appendChild(note); inner.appendChild(nav); f.appendChild(inner);
    document.body.appendChild(f);
    size();
    window.addEventListener("resize", size);
  });
})();

// ---------- light / dark theme. Default: follow the device; the switch stores a choice in this browser ----------
(function () {
  var KEY = "sa_theme", root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function saved() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function effective() { var s = saved(); return s === "light" || s === "dark" ? s : (media && media.matches ? "dark" : "light"); }
  function apply() {
    var s = saved();
    if (s === "light" || s === "dark") root.setAttribute("data-theme", s); else root.removeAttribute("data-theme");
  }
  apply();   // runs in <head>, before the page paints
  var ICONS = {
    dark: '<svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z"/></svg>',
    light: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
  };
  function label(btn) {
    var next = effective() === "dark" ? "light" : "dark";
    btn.innerHTML = ICONS[next] + "<span>" + (next === "dark" ? "Dark" : "Light") + "</span>";
    btn.setAttribute("aria-label", "Switch to " + next + " theme");
    btn.setAttribute("data-tip", "Switch to the " + next + " theme" + (saved() ? "." : ". The page currently follows your device setting."));
  }
  function changed() { window.dispatchEvent(new Event("themechange")); }
  onReady(function () {
    var bar = document.querySelector(".topbar .tb-actions") || document.querySelector(".topbar .in");
    if (!bar || bar.querySelector(".theme-btn")) return;
    var btn = document.createElement("button");
    btn.type = "button"; btn.className = "theme-btn";
    btn.onclick = function () {
      try { localStorage.setItem(KEY, effective() === "dark" ? "light" : "dark"); } catch (e) {}
      apply(); label(btn); changed();
    };
    label(btn); bar.appendChild(btn);
    if (media && media.addEventListener) media.addEventListener("change", function () { if (!saved()) { label(btn); changed(); } });
  });
})();

// ---------- hover explanations: any element with data-tip="…" (mouse, keyboard focus, or tap) ----------
(function () {
  var box, current, timer;
  function ensure() {
    if (box) return box;
    box = document.createElement("div");
    box.className = "tipbox"; box.setAttribute("role", "tooltip"); box.id = "tipbox";
    document.body.appendChild(box);
    return box;
  }
  function show(target) {
    var text = target.getAttribute("data-tip");
    if (!text || target.closest(".dd.open")) return;   // no tooltips over an open dropdown
    var b = ensure();
    b.textContent = text;
    b.style.display = "block";
    var r = target.getBoundingClientRect(), bw = b.offsetWidth, bh = b.offsetHeight;
    var left = Math.min(Math.max(8, r.left + r.width / 2 - bw / 2), window.innerWidth - bw - 8);
    var top = r.top - bh - 8;
    if (top < 56) top = r.bottom + 8;
    b.style.left = left + "px"; b.style.top = top + "px";
    current = target;
    target.setAttribute("aria-describedby", "tipbox");
  }
  function hide() {
    clearTimeout(timer);
    if (box) box.style.display = "none";
    if (current) current.removeAttribute("aria-describedby");
    current = null;
  }
  function tipTarget(e) { return e.target && e.target.closest ? e.target.closest("[data-tip]") : null; }
  document.addEventListener("mouseover", function (e) {
    var t = tipTarget(e);
    if (t === current) return;
    clearTimeout(timer);
    if (!t) { hide(); return; }
    timer = setTimeout(function () { show(t); }, 120);
  });
  document.addEventListener("focusin", function (e) { var t = tipTarget(e); if (t) show(t); });
  document.addEventListener("focusout", hide);
  document.addEventListener("scroll", hide, true);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") hide(); });
  document.addEventListener("pointerdown", function (e) {           // touch: tap a term to see its explanation
    if (e.pointerType !== "touch") return;
    var t = tipTarget(e);
    if (t && t !== current && !t.matches("button, a, select, input")) { e.preventDefault(); show(t); } else if (!t) hide();
  });
})();

// ---------- custom dropdown: enhanceSelect(selectEl, {search, wide}) ----------
// The native <select> stays (hidden) as the source of truth: choosing an item sets select.value and fires
// "change", so page code keeps working. Options can carry data-sub="second line" (e.g. the task's request).
window.enhanceSelect = function (select, opts) {
  opts = opts || {};
  if (select._dd) { select._dd.refresh(); return select._dd; }
  var wrap = document.createElement("div"); wrap.className = "dd" + (opts.wide ? " wide" : "");
  var trigger = document.createElement("button");
  trigger.type = "button"; trigger.className = "dd-trigger";
  trigger.setAttribute("aria-haspopup", "listbox"); trigger.setAttribute("aria-expanded", "false");
  var aria = select.getAttribute("aria-label"); if (aria) trigger.setAttribute("aria-label", aria);
  var tl = document.createElement("span"); tl.className = "dd-label";
  trigger.appendChild(tl);
  var pop = document.createElement("div"); pop.className = "dd-pop"; pop.hidden = true;
  var search = null;
  if (opts.search) {
    search = document.createElement("input");
    search.type = "search"; search.className = "dd-search"; search.placeholder = opts.placeholder || "Search…";
    search.setAttribute("aria-label", "Filter options");
    pop.appendChild(search);
  }
  var list = document.createElement("div"); list.className = "dd-list"; list.setAttribute("role", "listbox");
  pop.appendChild(list);
  var count = document.createElement("div"); count.className = "dd-count";
  if (opts.search) pop.appendChild(count);
  select.parentNode.insertBefore(wrap, select);
  wrap.appendChild(trigger); wrap.appendChild(pop); wrap.appendChild(select);
  select.classList.add("dd-native"); select.tabIndex = -1;

  var items = [], active = -1;
  function sync() {
    var o = select.options[select.selectedIndex];
    tl.textContent = o ? o.text : "";
    if (o && o.dataset.sub) trigger.setAttribute("data-tip", o.dataset.sub); else trigger.removeAttribute("data-tip");
  }
  function build() {
    list.innerHTML = ""; items = [];
    var q = search ? search.value.trim().toLowerCase() : "";
    Array.prototype.forEach.call(select.options, function (o) {
      var hay = (o.text + " " + (o.dataset.sub || "")).toLowerCase();
      if (q && hay.indexOf(q) === -1) return;
      var it = document.createElement("div");
      it.className = "dd-opt"; it.setAttribute("role", "option");
      it.setAttribute("aria-selected", String(o.value === select.value));
      var l = document.createElement("div"); l.className = "l"; l.textContent = o.text; it.appendChild(l);
      if (o.dataset.sub) { var s = document.createElement("div"); s.className = "s"; s.textContent = o.dataset.sub; it.appendChild(s); }
      it.onmousedown = function (e) { e.preventDefault(); choose(o.value); };
      it.onmousemove = function () { setActive(items.indexOf(it)); };
      list.appendChild(it); items.push(it); it._value = o.value;
    });
    if (opts.search) count.textContent = items.length + " of " + select.options.length;
    setActive(Math.max(0, items.findIndex(function (it) { return it._value === select.value; })));
  }
  function setActive(i) {
    if (active >= 0 && items[active]) items[active].classList.remove("active");
    active = i;
    if (items[i]) { items[i].classList.add("active"); items[i].scrollIntoView({ block: "nearest" }); }
  }
  function place() {     // float above everything (escapes scrolling panels), below the trigger or above if no room
    var r = trigger.getBoundingClientRect();
    var w = Math.min(Math.max(r.width, opts.wide ? 320 : 160), window.innerWidth - 24);   // list as wide as its button
    var left = Math.min(Math.max(12, r.left), window.innerWidth - w - 12);
    pop.style.width = w + "px"; pop.style.left = left + "px";
    var below = window.innerHeight - r.bottom, ph = pop.offsetHeight;
    pop.style.top = (below > ph + 12 || below > r.top ? r.bottom + 4 : r.top - ph - 4) + "px";
  }
  function onScroll(e) {             // follow the button while the page scrolls; close if it leaves the screen
    if (pop.contains(e.target)) return;
    var r = trigger.getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) close(false); else place();
  }
  function open() {
    if (!pop.hidden) return;
    pop.hidden = false; trigger.setAttribute("aria-expanded", "true"); wrap.classList.add("open");
    if (search) search.value = "";
    build(); place();
    var tb = document.getElementById("tipbox"); if (tb) tb.style.display = "none";
    setTimeout(function () { (search || list).focus({ preventScroll: true }); }, 0);
    document.addEventListener("mousedown", outside);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", place);
  }
  function close(focusTrigger) {
    if (pop.hidden) return;
    pop.hidden = true; trigger.setAttribute("aria-expanded", "false"); wrap.classList.remove("open");
    document.removeEventListener("mousedown", outside);
    window.removeEventListener("scroll", onScroll, true);
    window.removeEventListener("resize", place);
    if (focusTrigger) trigger.focus({ preventScroll: true });
  }
  function outside(e) { if (!wrap.contains(e.target)) close(false); }
  function choose(v) {
    close(true);
    if (select.value !== v) { select.value = v; select.dispatchEvent(new Event("change", { bubbles: true })); }
    sync();
  }
  function keys(e) {
    if (e.key === "ArrowDown") { e.preventDefault(); setActive(Math.min(items.length - 1, active + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive(Math.max(0, active - 1)); }
    else if (e.key === "Enter") { e.preventDefault(); if (items[active]) choose(items[active]._value); }
    else if (e.key === "Escape") { e.preventDefault(); close(true); }
    else if (e.key === "Tab") close(false);
  }
  trigger.onclick = function () { pop.hidden ? open() : close(true); };
  trigger.onkeydown = function (e) { if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); open(); } };
  pop.addEventListener("keydown", keys);
  list.tabIndex = -1;
  if (search) search.addEventListener("input", function () { build(); place(); });
  var api = { refresh: function () { sync(); if (!pop.hidden) build(); }, sync: sync, open: open, close: close };
  select._dd = api;
  sync();
  return api;
};

// ---------- glossary: one explanation per term, used by both pages (data-tip) ----------
window.TIPS = {
  sa: "Syntactic Alignment: a vector δ added to the model's final hidden states and trained with a few Adam steps on the current prompt, before the model answers.",
  dg: "Dynamics Grounding: the model first explores the tools, writes down rules about how they behave, and those rules are added to its prompt.",
  released: "The gradient exactly as in the released code (vllm/model_runner.py), with user-turn masking off, as in Eq. 3 of the paper. It averages over the prompt twice, so its step shrinks as the prompt gets longer.",
  mask: "The released code with user turns masked out of the loss (its MASK_TOKENS option). The repository doesn't record which setting the paper's runs used, so both are shown.",
  exact: "The true gradient of the mean next-token loss, averaged once. A reference point for comparison; it is not part of the paper.",
  delta: "‖δ‖ is the size (L2 norm) of the vector SA adds to the final hidden states. Bigger means a bigger change to the model's output.",
  eps: "Share of gradient coordinates smaller than Adam's ε = 10⁻⁶. When nearly all are below ε, Adam's step becomes about lr·g/ε instead of lr·sign(g), so it is tiny and shrinks with the gradient.",
  b: "Because logits = (H + δ)Wᵀ = HWᵀ + δWᵀ, SA adds the same score b = δWᵀ to each vocabulary token at every position. A large b means the token is pushed up.",
  lr: "Adam learning rate for δ. The paper uses 0.1 by default and reports 1.0 (with 3 steps) as its best setting for Qwen2.5-14B on BFCL.",
  steps: "How many Adam steps SA takes on the prompt before the model answers. The paper's best 14B setting uses 3.",
  model: "1.5B starts and runs fastest. 14B is the model used in the paper; it takes longer to start.",
  ratio: "How many times smaller the released code's first step is than the exact gradient's, at this prompt's full length.",
  anslp: "Average change in log-probability per token of the expected answer when δ is added. Positive means SA makes the answer more likely.",
  loss: "Mean next-token loss on the prompt before → after SA's update. SA is trained to lower this number.",
  curve: "The same prompt cut to its first L tokens. The model reads left to right, so one pass gives every L. The exact gradient's step stays about the same; the released code's falls roughly as 1/L.",
  gold: "The correct first function call from BFCL for this task. We score how much more or less likely SA makes it.",
  changed: "Tasks where SA altered at least one generated token. If SA changes nothing, the run is identical to the run without SA.",
  pytorch: "SA runs used a separate PyTorch loop, so compare them only with the no-SA run from the same loop.",
  vllm: "DG runs used vLLM (as in the paper). Compare DG only with the no-adaptation run from vLLM.",
  copy: "A simple baseline: add the same bonus (α = 0.25) to every token that appears in the prompt. It helps nothing either.",
  task: "A BFCLv3 multi-turn task from the paper's 200-task split. The prompt contains the tools' documentation and the user's first request.",
  dgruns: "Three runs with identical prompts. Scores move by about 1 point between identical runs.",
  spearman: "Rank correlation between ‖δ‖ and prompt length across the 200 tasks. −0.99 means longer prompts almost always give a smaller step.",
};
window.TIPS.gradient = "Which gradient SA uses: Released (as published), Released + mask (user turns masked), or Exact (the true gradient, for reference).";

// data-tipkey="lr" → data-tip from the glossary (static HTML); dynamic content sets data-tip directly.
window.tipify = function (root) {
  (root || document).querySelectorAll("[data-tipkey]").forEach(function (el) {
    var t = window.TIPS[el.getAttribute("data-tipkey")];
    if (t) el.setAttribute("data-tip", t);
  });
};
onReady(function () { window.tipify(); });

// ---------- in-page report viewer (PDF.js): every plain click on a .pdf link opens it in a panel ----------
// Cmd/Ctrl/Shift/middle-click still open the PDF in a new tab. PDF.js is loaded only on first use.
(function () {
  var PDFJS = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
  var WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  var lib = null, doc = null, url = null, zoom = 1, ui = null, lastFocus = null, renderToken = 0;

  function loadLib() {
    if (lib) return Promise.resolve(lib);
    return new Promise(function (ok, fail) {
      var s = document.createElement("script");
      s.src = PDFJS; s.onload = function () { lib = window.pdfjsLib; lib.GlobalWorkerOptions.workerSrc = WORKER; ok(lib); };
      s.onerror = function () { fail(new Error("PDF viewer failed to load")); };
      document.head.appendChild(s);
    });
  }
  function btn(label, tip, onClick, cls) {
    var b = document.createElement("button");
    b.type = "button"; b.className = "btn" + (cls ? " " + cls : ""); b.textContent = label;
    if (tip) { b.setAttribute("data-tip", tip); b.setAttribute("aria-label", tip); }
    b.onclick = onClick; return b;
  }
  function build() {
    if (ui) return ui;
    var root = document.createElement("div"); root.className = "pdfv"; root.hidden = true;
    root.setAttribute("role", "dialog"); root.setAttribute("aria-modal", "true"); root.setAttribute("aria-label", "Report");
    var panel = document.createElement("div"); panel.className = "pdfv-panel";
    var bar = document.createElement("div"); bar.className = "pdfv-bar";
    var title = document.createElement("div"); title.className = "pdfv-title"; title.textContent = "Report";
    var zoomLbl = document.createElement("span"); zoomLbl.className = "pdfv-zoom";
    var grp = document.createElement("div"); grp.className = "pdfv-tools";
    var out = btn("−", "Zoom out", function () { setZoom(zoom / 1.2); }, "icon");
    var inn = btn("+", "Zoom in", function () { setZoom(zoom * 1.2); }, "icon");
    var fit = btn("Fit width", "Fit the page to the panel width", function () { setZoom(1); });
    var dl = document.createElement("a"); dl.className = "btn"; dl.textContent = "Download"; dl.setAttribute("download", "GTTA-reproduction-notes.pdf");
    dl.setAttribute("data-tip", "Download the PDF");
    var tab = document.createElement("a"); tab.className = "btn"; tab.textContent = "Open in new tab"; tab.target = "_blank"; tab.rel = "noopener";
    var max = btn("Maximize", "Use the whole window", function () { toggleMax(); });
    var close = btn("✕", "Close (Esc)", function () { hide(); }, "icon");
    grp.append(out, zoomLbl, inn, fit, dl, tab, max, close);
    bar.append(title, grp);
    var body = document.createElement("div"); body.className = "pdfv-body"; body.tabIndex = 0;
    panel.append(bar, body); root.append(panel);
    root.addEventListener("mousedown", function (e) { if (e.target === root) hide(); });   // click the backdrop to close
    document.body.appendChild(root);
    ui = { root: root, panel: panel, body: body, title: title, zoomLbl: zoomLbl, dl: dl, tab: tab, max: max, close: close };
    var t; window.addEventListener("resize", function () { if (!root.hidden) { clearTimeout(t); t = setTimeout(render, 150); } });
    return ui;
  }
  function setZoom(z) { zoom = Math.min(3, Math.max(0.5, z)); render(); }
  function toggleMax() {
    ui.root.classList.toggle("max");
    ui.max.textContent = ui.root.classList.contains("max") ? "Restore" : "Maximize";
    render();
  }
  function keys(e) {
    if (e.key === "Escape") { e.preventDefault(); hide(); }
    else if ((e.key === "+" || e.key === "=") && (e.metaKey || e.ctrlKey)) { e.preventDefault(); setZoom(zoom * 1.2); }
    else if (e.key === "-" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); setZoom(zoom / 1.2); }
  }
  async function render() {
    if (!doc) return;
    var token = ++renderToken, body = ui.body;
    var avail = body.clientWidth - 32, dpr = window.devicePixelRatio || 1;
    ui.zoomLbl.textContent = Math.round(zoom * 100) + "%";
    var scrollRatio = body.scrollHeight ? body.scrollTop / body.scrollHeight : 0;
    var frag = document.createDocumentFragment();
    for (var n = 1; n <= doc.numPages; n++) {
      var page = await doc.getPage(n);
      if (token !== renderToken) return;
      var base = page.getViewport({ scale: 1 });
      var vp = page.getViewport({ scale: (avail / base.width) * zoom });
      var wrap = document.createElement("div"); wrap.className = "pdfv-page";
      wrap.style.width = vp.width + "px"; wrap.style.height = vp.height + "px";
      var canvas = document.createElement("canvas");
      canvas.width = Math.floor(vp.width * dpr); canvas.height = Math.floor(vp.height * dpr);
      canvas.style.width = vp.width + "px"; canvas.style.height = vp.height + "px";
      wrap.appendChild(canvas);
      await page.render({ canvasContext: canvas.getContext("2d"), viewport: page.getViewport({ scale: vp.scale * dpr }) }).promise;
      var annots = await page.getAnnotations();                     // keep the PDF's links clickable
      annots.forEach(function (a) {
        if (a.subtype !== "Link" || !a.url) return;
        var r = vp.convertToViewportRectangle(a.rect), link = document.createElement("a");
        link.href = a.url; link.target = "_blank"; link.rel = "noopener"; link.className = "pdfv-link";
        link.style.left = Math.min(r[0], r[2]) + "px"; link.style.top = Math.min(r[1], r[3]) + "px";
        link.style.width = Math.abs(r[2] - r[0]) + "px"; link.style.height = Math.abs(r[3] - r[1]) + "px";
        link.setAttribute("aria-label", a.url); link.setAttribute("data-tip", a.url.replace(/^mailto:/, ""));
        wrap.appendChild(link);
      });
      if (token !== renderToken) return;
      frag.appendChild(wrap);
    }
    body.innerHTML = ""; body.appendChild(frag);
    body.scrollTop = scrollRatio * body.scrollHeight;
  }
  async function show(href) {
    build();
    lastFocus = document.activeElement;
    ui.root.hidden = false; document.documentElement.classList.add("pdfv-open");
    ui.dl.href = href; ui.tab.href = href;
    document.addEventListener("keydown", keys);
    ui.close.focus({ preventScroll: true });
    if (url !== href) {
      url = href; doc = null; zoom = 1;
      ui.body.innerHTML = '<div class="pdfv-msg">Loading the report…</div>';
      try {
        await loadLib();
        doc = await lib.getDocument(href).promise;
        ui.title.textContent = "Report · " + doc.numPages + " page" + (doc.numPages > 1 ? "s" : "");
      } catch (e) {
        url = null;
        ui.body.innerHTML = '<div class="pdfv-msg">Couldn\'t display the report here. <a href="' + href + '" target="_blank" rel="noopener">Open it in a new tab</a>.</div>';
        return;
      }
    }
    render();
    try { window.trackDemoAction("📄 Opened the report viewer"); } catch (e) {}
  }
  function hide() {
    if (!ui || ui.root.hidden) return;
    ui.root.hidden = true; document.documentElement.classList.remove("pdfv-open");
    document.removeEventListener("keydown", keys);
    if (lastFocus && lastFocus.focus) lastFocus.focus({ preventScroll: true });
  }
  window.openReport = show;
  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest('a[href$=".pdf"]');
    if (!a || a.closest(".pdfv") || a.hasAttribute("download")) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;    // let the browser handle new-tab clicks
    e.preventDefault();
    show(a.getAttribute("href"));
  });
})();

// ---------- links open in a new tab, except: the top-bar tabs (page switching), report links (in-page viewer),
// email links, and in-page anchors. Works for links added later too (one delegated handler). ----------
(function () {
  function keepHere(a) {
    var href = a.getAttribute("href") || "";
    return !href || href.charAt(0) === "#" || /^mailto:|^javascript:/i.test(href) || /\.pdf$/i.test(href)
      || a.closest(".topbar") || a.closest(".pdfv") || a.hasAttribute("download");
  }
  function mark(root) {
    (root || document).querySelectorAll("a[href]").forEach(function (a) {
      if (!keepHere(a)) { a.target = "_blank"; a.rel = "noopener"; }
    });
  }
  onReady(function () {
    mark();
    new MutationObserver(function (muts) {
      muts.forEach(function (m) { m.addedNodes.forEach(function (n) { if (n.nodeType === 1) mark(n.parentNode || n); }); });
    }).observe(document.body, { childList: true, subtree: true });
  });
})();
Object.assign(window.TIPS, {
  rule: "One fact DG wrote down about how a tool behaves: what the agent did, what happened, and what the situation was before. The kept rules are added to the agent's prompt.",
  action: "The tool call the agent made while exploring.",
  effect: "What happened after the call. This is the part that teaches the agent how the tool behaves.",
  before: "What the environment looked like before the call, as described by the model that wrote the rule.",
  kept: "Rules written from all exploration logs, then filtered by the model to remove duplicates and vague rules. Only the kept ones are added to the prompt.",
  episode: "One exploration run: the agent is given a goal (from GTTA's 80 exploration goals) and calls the real tool simulators step by step. Rules are written from these logs.",
  stable: "Pass/fail in each of the 3 runs with identical prompts. Runs differ slightly because batched GPU decoding isn't perfectly deterministic.",
  dgwin: "Tasks where DG passes in more runs than no adaptation does.",
  dgloss: "Tasks where no adaptation passes in more runs than DG does.",
  selfexplore: "As in the paper's self-exploration variant, Qwen2.5-14B explored, wrote the rules and filtered them itself (the paper's main setup used GPT-4.1, GPT-4o and o3).",
});
