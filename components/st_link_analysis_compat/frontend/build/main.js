/* Minimal Streamlit component protocol + Cytoscape renderer.
 *
 * Goal: be API-compatible with the Python signature of `st_link_analysis`.
 *
 * Expected args from Python:
 * - elements: {nodes: [{data:{...}}], edges: [{data:{...}}]}
 * - style: Cytoscape stylesheet rules (list)
 * - layout: Cytoscape layout config (dict)
 * - height: string like "500px"
 * - nodeActions: ["remove","expand"]
 * - events: [{name,event_type,selector,debounce}]
 */

(function () {
  const API_VERSION = 1;

  function postToStreamlit(type, data) {
    window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
  }

  function setComponentReady() {
    postToStreamlit("streamlit:componentReady", { apiVersion: API_VERSION });
  }

  function setFrameHeight(height) {
    postToStreamlit("streamlit:setFrameHeight", { height });
  }

  function setComponentValue(value) {
    postToStreamlit("streamlit:setComponentValue", { value });
  }

  function debounce(fn, waitMs) {
    let t;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), waitMs);
    };
  }

  function nowMs() {
    return Date.now();
  }

  // Cytoscape instance
  let cy = null;
  let lastArgsSignature = null;
  let installedEventHandlers = false;
  let lastContainerHeight = null;
  let lastLayoutArgs = null;
  let lastSelectionInfoConfig = null;

  function normalizeStyles(customStyle) {
    const base = [
      {
        selector: "node",
        style: {
          width: 20,
          height: 20,
          "background-color": "#f8fafc",
          "border-width": 1,
          "border-color": "#94a3b8",
          "font-size": 10,
          color: "#0f172a",
          label: "data(label)",
          "text-valign": "bottom",
          "text-margin-y": 6,
          "background-repeat": "no-repeat",
          "background-width": "60%",
          "background-height": "60%",
        },
      },
      {
        selector: ":parent",
        style: {
          shape: "round-rectangle",
          "corner-radius": 24,
          "background-opacity": 0.08,
          "background-color": "#0f172a",
          "border-color": "#0f172a",
          "border-width": 1,
          "padding": 12,
          "text-valign": "top",
          "text-halign": "center",
          "font-size": 11,
          "font-weight": 700,
          "text-margin-y": -6,
        },
      },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": "#94a3b8",
          "target-arrow-color": "#94a3b8",
          "target-arrow-shape": "none",
          "curve-style": "bezier",
          "font-size": 9,
          color: "#0f172a",
          "text-background-color": "#e2e8f0",
          "text-background-opacity": 0.9,
          "text-background-shape": "round-rectangle",
          "text-background-padding": 2,
        },
      },
      {
        selector: ":selected",
        style: {
          "border-color": "#ef4444",
          "border-width": 3,
          "line-color": "#ef4444",
          "target-arrow-color": "#ef4444",
          "text-background-color": "#ef4444",
        },
      },
    ];

    if (Array.isArray(customStyle) && customStyle.length) {
      return base.concat(customStyle);
    }
    return base;
  }

  function safeStringify(obj) {
    try {
      return JSON.stringify(obj, null, 2);
    } catch {
      return String(obj);
    }
  }

  function escapeHtml(v) {
    const s = v == null ? "" : String(v);
    return s
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function renderKeyValueTable(obj, keys) {
    const rows = [];
    for (const k of keys) {
      if (!obj || !(k in obj)) continue;
      const v = obj[k];
      if (v === undefined) continue;
      rows.push(
        `<tr><td class="info__k">${escapeHtml(k)}</td><td class="info__v">${escapeHtml(
          typeof v === "object" ? safeStringify(v) : v
        )}</td></tr>`
      );
    }
    if (rows.length === 0) return "";
    return `<table class="info__kv">${rows.join("")}</table>`;
  }

  function renderKeyValueTableEntries(obj, entries) {
    const rows = [];
    for (const ent of entries || []) {
      if (!ent) continue;
      const k = ent.key;
      if (!k) continue;

      const label = ent.label != null ? String(ent.label) : String(k);
      const type = ent.type != null ? String(ent.type) : "text";
      const linkText = ent.link_text != null ? String(ent.link_text) : null;
      const noneText = ent.none_text != null ? String(ent.none_text) : "None";

      if (!obj || !(k in obj)) continue;
      const v = obj[k];
      if (v === undefined) continue;

      const raw = typeof v === "object" ? safeStringify(v) : v;
      const s = raw == null ? "" : String(raw);

      let valueHtml = escapeHtml(s);

      if (type === "link") {
        const href = s.trim();
        if (href && isSafeUrl(href)) {
          const text = linkText != null ? linkText : href;
          valueHtml = `<a class="info__link" href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(text)}</a>`;
        } else if (!href) {
          valueHtml = `<span class="info__none">${escapeHtml(noneText)}</span>`;
        }
      } else if (type === "image") {
        const src = s.trim();
        if (src && isSafeUrl(src)) {
          const alt = label || k;
          valueHtml = `<a class="info__imglink" href="${escapeHtml(src)}" target="_blank" rel="noopener noreferrer"><img class="info__img" src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy" /></a>`;
        } else {
          valueHtml = `<span class="info__none">${escapeHtml(noneText)}</span>`;
        }
      }

      rows.push(
        `<tr><td class="info__k">${escapeHtml(label)}</td><td class="info__v">${valueHtml}</td></tr>`
      );
    }
    if (rows.length === 0) return "";
    return `<table class="info__kv">${rows.join("")}</table>`;
  }

  function normalizeSelectionInfoConfig(cfg) {
    if (!cfg || typeof cfg !== "object") return null;
    const out = {};
    out.column_order = Array.isArray(cfg.column_order) ? cfg.column_order.map(String) : null;
    out.column_config = cfg.column_config && typeof cfg.column_config === "object" ? cfg.column_config : {};
    out.hide = Array.isArray(cfg.hide) ? cfg.hide.map(String) : [];
    // Default behavior: keep compatibility with current component by showing unknown fields.
    out.show_unknown_fields = cfg.show_unknown_fields === undefined ? true : Boolean(cfg.show_unknown_fields);
    out.unknown_fields_section = cfg.unknown_fields_section === "details" ? "details" : "raw";
    out.sections = cfg.sections && typeof cfg.sections === "object" ? cfg.sections : {};
    return out;
  }

  function isSafeUrl(href) {
    if (!href) return false;
    const s = String(href).trim();
    // allow only http(s) and mailto for safety
    return s.startsWith("http://") || s.startsWith("https://") || s.startsWith("mailto:");
  }

  function renderLegend(legendItems) {
    const legend = document.getElementById("legend");
    const body = document.getElementById("legendBody");
    if (!legend || !body) return;

    if (!Array.isArray(legendItems) || legendItems.length === 0) {
      legend.style.display = "none";
      body.innerHTML = "";
      return;
    }

    const parts = [];
    for (const it of legendItems) {
      if (!it) continue;
      const text = it.text != null ? String(it.text) : (it.label != null ? String(it.label) : "");
      const href = it.href != null ? String(it.href) : (it.url != null ? String(it.url) : "");
      const color = it.color != null ? String(it.color) : null;

      const dotStyle = color ? ` style=\"background:${escapeHtml(color)}\"` : "";
      let content = `<span class=\"legend__text\">${escapeHtml(text)}</span>`;
      if (href && isSafeUrl(href)) {
        content = `<a class=\"legend__link legend__text\" href=\"${escapeHtml(href)}\" target=\"_blank\" rel=\"noopener noreferrer\">${escapeHtml(text)}</a>`;
      }

      parts.push(`<div class=\"legend__item\"><span class=\"legend__dot\"${dotStyle}></span>${content}</div>`);
    }

    body.innerHTML = parts.join("");
    legend.style.display = "block";
  }

  function renderSelectionInfoHtml(data, selectionInfoConfig) {
    const d = data || {};

    const cfg = normalizeSelectionInfoConfig(selectionInfoConfig);
    if (cfg) {
      const hideSet = new Set(cfg.hide || []);
      const used = new Set();
      const detailsEntries = [];
      const rawEntries = [];

      const order = cfg.column_order && cfg.column_order.length ? cfg.column_order : Object.keys(cfg.column_config || {});
      for (const key of order || []) {
        const k = String(key);
        if (!k || hideSet.has(k)) continue;
        const col = (cfg.column_config || {})[k] || {};
        const visible = col.visible === undefined ? true : Boolean(col.visible);
        if (!visible) continue;
        const section = col.section === "raw" ? "raw" : "details";
        const label = col.label != null ? String(col.label) : k;
        const type = col.type != null ? String(col.type) : "text";
        const link_text = col.link_text != null ? String(col.link_text) : null;
        const none_text = col.none_text != null ? String(col.none_text) : null;
        used.add(k);
        const entry = { key: k, label, type, link_text, none_text };
        if (section === "raw") rawEntries.push(entry);
        else detailsEntries.push(entry);
      }

      if (cfg.show_unknown_fields) {
        const unknownKeys = Object.keys(d)
          .filter((k) => !used.has(k) && !hideSet.has(k))
          .sort((a, b) => a.localeCompare(b));
        const unknownEntries = unknownKeys.map((k) => ({ key: k, label: k, type: "text" }));
        if (cfg.unknown_fields_section === "details") detailsEntries.push(...unknownEntries);
        else rawEntries.push(...unknownEntries);
      }

      const detailsTitle = cfg.sections.details != null ? String(cfg.sections.details) : "Details";
      const rawTitle = cfg.sections.raw != null ? String(cfg.sections.raw) : "Raw";

      const details = renderKeyValueTableEntries(d, detailsEntries);
      const raw = renderKeyValueTableEntries(d, rawEntries);

      const parts = [];
      if (details) {
        parts.push(`<div class="info__section"><div class="info__section-title">${escapeHtml(detailsTitle)}</div>${details}</div>`);
      }
      if (raw) {
        parts.push(`<div class="info__section"><div class="info__section-title">${escapeHtml(rawTitle)}</div>${raw}</div>`);
      }
      if (parts.length === 0) {
        parts.push(
          `<div class="info__section"><div class="info__section-title">${escapeHtml(rawTitle)}</div><div class="info__v">${escapeHtml(
            safeStringify(d)
          )}</div></div>`
        );
      }
      return parts.join("");
    }

    // Preferential order for common graph fields
    const primaryKeys = [
      "name",
      "label",
      "id",
      "cluster_id",
      "friend_user_id",
      "soc_user_id",
      "soc_user_name",
      "weight",
      "raw_weight",
      "n_items",
      "description",
      "keyword",
      "concept",
      "user_id",
      "resource_type",
      "resource_id",
    ];

    const seen = new Set(primaryKeys);
    const extraKeys = Object.keys(d)
      .filter((k) => !seen.has(k))
      .sort((a, b) => a.localeCompare(b));

    const primary = renderKeyValueTable(d, primaryKeys);
    const extra = renderKeyValueTable(d, extraKeys);

    const parts = [];
    if (primary) {
      parts.push(`<div class="info__section"><div class="info__section-title">Details</div>${primary}</div>`);
    }
    if (extra) {
      parts.push(`<div class="info__section"><div class="info__section-title">Raw</div>${extra}</div>`);
    }

    // Final fallback
    if (parts.length === 0) {
      parts.push(
        `<div class="info__section"><div class="info__section-title">Raw</div><div class="info__v">${escapeHtml(
          safeStringify(d)
        )}</div></div>`
      );
    }

    return parts.join("");
  }

  function showSelectionInfo(ele) {
    const info = document.getElementById("info");
    const title = document.getElementById("infoTitle");
    const body = document.getElementById("infoBody");

    if (!ele) {
      info.style.display = "none";
      title.textContent = "";
      body.textContent = "";
      return;
    }

    const data = ele.data ? ele.data() : {};
    const label = (data && data.label) || (ele.group && ele.group()) || "selected";
    const isNode = (ele.group && ele.group()) === "nodes";

    let titleText = String(label);
    if (isNode) {
      const name = (data && (data.name || data.soc_user_name || data.friend_name)) || "";
      if (name) {
        titleText = `${label} (${name})`;
      } else {
        titleText = String(label);
      }
    } else {
      // Edge titles: do not include ids (they often contain user_id).
      titleText = String(label);
    }

    info.style.display = "block";
    title.textContent = titleText;
    body.innerHTML = renderSelectionInfoHtml(data, lastSelectionInfoConfig);
  }

  function ensureCy() {
    if (!window.cytoscape) {
      throw new Error("Cytoscape not loaded (CDN blocked?).");
    }

    // register extensions if available
    if (window.cytoscapeFcose) {
      try { window.cytoscape.use(window.cytoscapeFcose); } catch {}
    }
    if (window.cytoscapeCola) {
      try { window.cytoscape.use(window.cytoscapeCola); } catch {}
    }

    if (window.cytoscapeCoseBilkent) {
      try { window.cytoscape.use(window.cytoscapeCoseBilkent); } catch {}
    }

    if (!cy) {
      cy = window.cytoscape({
        container: document.getElementById("cy"),
        elements: [],
        style: normalizeStyles([]),
        layout: { name: "cose" },
      });

      cy.on(
        "select unselect",
        debounce(() => {
          const sel = cy.$(":selected");
          if (sel && sel.length === 1) {
            showSelectionInfo(sel[0]);
          } else {
            showSelectionInfo(null);
          }
          setFrameHeight();
        }, 50)
      );
    }

    return cy;
  }

  function setFrameHeight() {
    const h = document.body.scrollHeight;
    postToStreamlit("streamlit:setFrameHeight", { height: h });
  }

  function emitAction(action, data) {
    setComponentValue({ action, data, timestamp: nowMs() });
  }

  function installToolbarHandlers() {
    if (installedEventHandlers) return;
    installedEventHandlers = true;

    document.getElementById("btnFit").addEventListener("click", () => {
      if (!cy) return;
      cy.fit(undefined, 20);
      setFrameHeight();
    });

    document.getElementById("btnCenter").addEventListener("click", () => {
      if (!cy) return;
      cy.center();
      setFrameHeight();
    });

     document.getElementById("btnRelayout").addEventListener("click", () => {
       if (!cy) return;
       const layout = lastLayoutArgs || { name: "cose" };
       try {
         cy.once("layoutstop", () => {
           try {
             cy.fit(undefined, 20);
           } catch {
             // ignore
           }
           setTimeout(setFrameHeight, 50);
         });
         cy.layout(layout).run();
       } catch {
         try {
           cy.layout({ name: "cose" }).run();
         } catch {
           // ignore
         }
       }
       setFrameHeight();
     });

    document.getElementById("btnExport").addEventListener("click", () => {
      if (!cy) return;
      const json = cy.elements().not(":hidden").jsons();
      const blob = new Blob([JSON.stringify(json, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "graph.json";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

    async function toggleFullscreen() {
      const container = document.getElementById("container");
      if (!container) return;

      try {
        if (!document.fullscreenElement) {
          // Remember the last non-fullscreen height so we can restore it.
          lastContainerHeight = container.style.height;
          if (container.requestFullscreen) {
            await container.requestFullscreen();
          }
          // Fallbacks (older browsers)
          else if (container.webkitRequestFullscreen) {
            await container.webkitRequestFullscreen();
          } else if (container.msRequestFullscreen) {
            await container.msRequestFullscreen();
          }
        } else {
          if (document.exitFullscreen) {
            await document.exitFullscreen();
          } else if (document.webkitExitFullscreen) {
            await document.webkitExitFullscreen();
          } else if (document.msExitFullscreen) {
            await document.msExitFullscreen();
          }
        }
      } catch {
        // ignore
      }
    }

    document.getElementById("btnFullscreen").addEventListener("click", () => {
      toggleFullscreen();
    });

    document.addEventListener("fullscreenchange", () => {
      const container = document.getElementById("container");
      if (!container) return;

      // In fullscreen, fill the viewport; otherwise restore.
      if (document.fullscreenElement) {
        container.style.height = "100vh";
      } else if (lastContainerHeight != null) {
        container.style.height = lastContainerHeight;
      }

      if (cy) {
        try {
          cy.resize();
          cy.fit(undefined, 20);
        } catch {
          // ignore
        }
      }

      setTimeout(setFrameHeight, 50);
    });

    document.getElementById("btnRemove").addEventListener(
      "click",
      debounce(() => {
        if (!cy) return;
        const nodes = cy.$(":selected").filter("node");
        if (!nodes || nodes.length === 0) return;
        const ids = nodes.map((n) => n.id());
        nodes.remove();
        emitAction("remove", { node_ids: ids });
        showSelectionInfo(null);
        setFrameHeight();
      }, 80)
    );

    document.getElementById("btnExpand").addEventListener(
      "click",
      debounce(() => {
        if (!cy) return;
        const nodes = cy.$(":selected").filter("node");
        if (!nodes || nodes.length === 0) return;
        const ids = [nodes[0].id()];
        emitAction("expand", { node_ids: ids });
      }, 80)
    );

    // keyboard remove (Delete / Backspace)
    document.addEventListener("keydown", (ev) => {
      if (ev.key !== "Delete" && ev.key !== "Backspace") return;
      const btn = document.getElementById("btnRemove");
      if (btn && btn.style.display !== "none") btn.click();
    });

    // dblclick expand on node
    document.getElementById("cy").addEventListener("dblclick", (ev) => {
      if (!cy) return;
      const target = ev.target;
      if (!target) return;
    });

    // Cytoscape emits dblclick as "dblclick" on container isn't mapped;
    // use Cytoscape's own events.
    // We'll attach it per-render based on nodeActions.
  }

  function applyNodeActions(nodeActions) {
    const removeBtn = document.getElementById("btnRemove");
    const expandBtn = document.getElementById("btnExpand");

    const enabled = Array.isArray(nodeActions) ? nodeActions : [];

    removeBtn.style.display = enabled.includes("remove") ? "inline-block" : "none";
    expandBtn.style.display = enabled.includes("expand") ? "inline-block" : "none";

    if (cy) {
      cy.removeListener("dblclick", "node");
      cy.removeListener("dbltap", "node");

      if (enabled.includes("expand")) {
        const handler = debounce((evt) => {
          if (!evt || !evt.target) return;
          emitAction("expand", { node_ids: [evt.target.id()] });
        }, 80);

        cy.on("dblclick", "node", handler);
        cy.on("dbltap", "node", handler);
      }
    }
  }

  function installCustomEvents(events) {
    if (!cy) return;

    // remove previous custom listeners by namespace pattern
    cy.removeListener("click tap dblclick dbltap mouseover mouseout", "node");
    cy.removeListener("click tap dblclick dbltap mouseover mouseout", "edge");

    if (!Array.isArray(events)) return;

    events.forEach((e) => {
      if (!e || !e.name || !e.event_type || !e.selector) return;
      const wait = Math.max(Number(e.debounce || 100), 50);
      const handler = debounce((evt) => {
        emitAction(e.name, {
          type: evt.type,
          target_id: evt.target && evt.target.id ? evt.target.id() : null,
          target_group: evt.target && evt.target.group ? evt.target.group() : null,
        });
      }, wait);
      cy.on(e.event_type, e.selector, handler);
    });
  }

  function render(args, theme) {
      // Keep the selection panel config in sync with the latest args.
      lastSelectionInfoConfig = args && args.selectionInfoConfig ? args.selectionInfoConfig : null;
    const container = document.getElementById("container");
    if (args && args.height) {
      container.style.height = args.height;
    }

    // Only update lastContainerHeight when not in fullscreen.
    if (!document.fullscreenElement) {
      lastContainerHeight = container.style.height;
    }

    const signature = JSON.stringify({
      elements: args ? args.elements : null,
      style: args ? args.style : null,
      layout: args ? args.layout : null,
      nodeActions: args ? args.nodeActions : null,
      events: args ? args.events : null,
      legendItems: args ? args.legendItems : null,
      selectionInfoConfig: args ? args.selectionInfoConfig : null,
    });

    installToolbarHandlers();

    const instance = ensureCy();

    // legend is light; update on every render
    try {
      renderLegend(args && args.legendItems ? args.legendItems : null);
    } catch {
      // ignore
    }

    // Only reapply heavy updates if changed
    if (signature !== lastArgsSignature) {
      lastArgsSignature = signature;

      // elements
      instance.json({ elements: args && args.elements ? args.elements : { nodes: [], edges: [] } });

      // style
      instance.style(normalizeStyles(args && args.style ? args.style : []));

      // layout
      const layout = (args && args.layout) || { name: "cose" };
       lastLayoutArgs = layout;
      try {
        instance.layout(layout).run();
      } catch (err) {
        // fallback
        instance.layout({ name: "cose" }).run();
      }

      // node actions
      applyNodeActions(args && args.nodeActions ? args.nodeActions : []);

      // custom events
      installCustomEvents(args && args.events ? args.events : []);
    }

    setTimeout(setFrameHeight, 50);
  }

  // Streamlit protocol: listen for render messages.
  window.addEventListener("message", (event) => {
    const msg = event && event.data;
    if (!msg || msg.type !== "streamlit:render") return;

    // Streamlit uses { args, theme } in message; older versions may wrap it.
    const args = msg.args || (msg.detail && msg.detail.args) || msg;
    const theme = msg.theme || (msg.detail && msg.detail.theme) || null;

    render(args, theme);
  });

  // init
  setComponentReady();
  setTimeout(setFrameHeight, 50);
})();
