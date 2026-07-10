/* Minimal, robust single-page Markdown course reader.
   Depends on: marked, highlight.js, mermaid (loaded via CDN in index.html),
   and window.COURSE_MANIFEST (assets/manifest.js). */
(function () {
  "use strict";
  const M = window.COURSE_MANIFEST || [];
  const flat = M.flatMap(g => g.items);

  const $ = id => document.getElementById(id);
  const content = () => $("content");

  /* ---------- Theme ---------- */
  const getTheme = () => localStorage.getItem("lc-theme") || "dark";
  function setTheme(t) {
    localStorage.setItem("lc-theme", t);
    document.documentElement.setAttribute("data-theme", t);
    const b = $("themeBtn");
    if (b) b.textContent = t === "dark" ? "☀️" : "🌙";
  }

  /* ---------- Sidebar ---------- */
  function buildSidebar() {
    const nav = $("nav");
    nav.innerHTML = "";
    M.forEach(group => {
      const h = document.createElement("div");
      h.className = "nav-group";
      h.textContent = group.group;
      nav.appendChild(h);
      group.items.forEach(it => {
        const a = document.createElement("a");
        a.className = "nav-link";
        a.href = "#/" + it.path;
        a.textContent = it.title;
        a.dataset.path = it.path;
        nav.appendChild(a);
      });
    });
  }

  function setActive(path) {
    document.querySelectorAll(".nav-link").forEach(a =>
      a.classList.toggle("active", a.dataset.path === path));
  }

  /* ---------- Routing ---------- */
  function currentPath() {
    const h = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
    return h || (flat[0] && flat[0].path);
  }

  // Resolve a relative .md link against the current document path (handles ../).
  function resolveRel(href, basePath) {
    try {
      const u = new URL(href, "http://_/" + basePath);
      return u.pathname.replace(/^\//, "");
    } catch (e) { return href; }
  }

  /* ---------- Render ---------- */
  async function render(path) {
    const el = content();
    setActive(path);
    el.innerHTML = '<div class="loading">Loading…</div>';
    let md;
    try {
      const res = await fetch(path, { cache: "no-cache" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      md = await res.text();
    } catch (e) {
      el.innerHTML = location.protocol === "file:" ? fileHelp() :
        '<div class="error"><h2>Could not load this page</h2><p><code>' +
        path + '</code></p><pre>' + String(e) + "</pre></div>";
      return;
    }
    el.innerHTML = marked.parse(md);
    enhance(el, path);
    const item = flat.find(f => f.path === path);
    document.title = (item ? item.title + " · " : "") + "LangChain Course";
    const wrap = document.querySelector(".content-wrap");
    if (wrap) wrap.scrollTop = 0;
    window.scrollTo(0, 0);
  }

  function enhance(el, path) {
    // Build right-rail "On this page"
    const toc = $("toc");
    toc.innerHTML = '<div class="toc-title">On this page</div>';
    el.querySelectorAll("h2, h3").forEach((h, i) => {
      const id = (h.textContent || "").toLowerCase()
        .replace(/[^\w]+/g, "-").replace(/^-+|-+$/g, "") || ("sec-" + i);
      h.id = id;
      const link = document.createElement("a");
      link.className = "toc-link toc-" + h.tagName.toLowerCase();
      link.textContent = h.textContent;
      link.href = "javascript:void(0)";
      link.addEventListener("click", () => h.scrollIntoView({ behavior: "smooth", block: "start" }));
      toc.appendChild(link);
    });

    // Convert mermaid code fences into <div class="mermaid">
    el.querySelectorAll("pre code.language-mermaid").forEach(code => {
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent;
      const pre = code.closest("pre");
      if (pre) pre.replaceWith(div);
    });

    // Syntax highlight remaining code
    el.querySelectorAll("pre code").forEach(code => {
      try { hljs.highlightElement(code); } catch (e) { /* noop */ }
    });

    // Copy buttons
    el.querySelectorAll("pre").forEach(pre => {
      if (pre.querySelector(".copy-btn")) return;
      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.textContent = "Copy";
      btn.addEventListener("click", () => {
        const code = pre.querySelector("code");
        navigator.clipboard.writeText(code ? code.innerText : pre.innerText);
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = "Copy"), 1200);
      });
      pre.appendChild(btn);
    });

    // Render mermaid diagrams
    if (window.mermaid) {
      try { mermaid.run({ querySelector: "#content .mermaid" }); } catch (e) { /* noop */ }
    }

    // Intercept intra-course .md links so they route inside the SPA
    el.querySelectorAll("a[href]").forEach(a => {
      const href = a.getAttribute("href");
      if (!href) return;
      if (/^https?:/i.test(href) || href.startsWith("mailto:")) {
        a.target = "_blank"; a.rel = "noopener"; return;
      }
      if (href.includes(".md")) {
        a.addEventListener("click", e => {
          e.preventDefault();
          const resolved = resolveRel(href.split("#")[0], path);
          location.hash = "#/" + resolved;
        });
      }
    });

    // Prev / next
    const idx = flat.findIndex(f => f.path === path);
    const wrap = document.createElement("div");
    wrap.className = "prevnext";
    if (idx > 0) {
      const p = flat[idx - 1];
      const a = document.createElement("a");
      a.href = "#/" + p.path; a.className = "pn pn-prev";
      a.innerHTML = "<span>← Previous</span><strong>" + p.title + "</strong>";
      wrap.appendChild(a);
    } else { wrap.appendChild(document.createElement("span")); }
    if (idx > -1 && idx < flat.length - 1) {
      const n = flat[idx + 1];
      const a = document.createElement("a");
      a.href = "#/" + n.path; a.className = "pn pn-next";
      a.innerHTML = "<span>Next →</span><strong>" + n.title + "</strong>";
      wrap.appendChild(a);
    }
    el.appendChild(wrap);
  }

  function fileHelp() {
    return '<div class="error"><h2>Open this course via a local server</h2>' +
      "<p>Browsers block reading local files over <code>file://</code>. " +
      "From the <code>langchain-course</code> folder, run:</p>" +
      "<pre><code>python -m http.server 8000</code></pre>" +
      '<p>then open <a href="http://localhost:8000/">http://localhost:8000/</a>.</p>' +
      "<p>Prefer not to? The Markdown files render great on their own in VS Code " +
      "or on GitHub — see <code>modules/</code> and <code>appendix/</code>.</p></div>";
  }

  /* ---------- Search (filter sidebar titles) ---------- */
  function wireSearch() {
    const s = $("search");
    if (!s) return;
    s.addEventListener("input", () => {
      const q = s.value.trim().toLowerCase();
      document.querySelectorAll(".nav-link").forEach(a => {
        a.style.display = a.textContent.toLowerCase().includes(q) ? "" : "none";
      });
    });
  }

  function onRoute() {
    const p = currentPath();
    if (p) render(p);
  }

  function init() {
    setTheme(getTheme());
    buildSidebar();
    wireSearch();
    $("themeBtn").addEventListener("click", () =>
      setTheme(getTheme() === "dark" ? "light" : "dark"));
    $("menuBtn").addEventListener("click", () =>
      document.body.classList.toggle("nav-open"));
    if (window.mermaid) {
      mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });
    }
    window.addEventListener("hashchange", () => {
      document.body.classList.remove("nav-open");
      onRoute();
    });
    onRoute();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
