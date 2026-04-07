(function () {
  "use strict";

  var origin = window.location.origin;
  var prefix = typeof window.AI_API_PREFIX !== "undefined" ? window.AI_API_PREFIX : "/api";
  var AI_BASE;
  if (typeof window.ML_API_DIRECT === "string" && window.ML_API_DIRECT.length) {
    AI_BASE = window.ML_API_DIRECT.replace(/\/$/, "");
  } else {
    AI_BASE = origin.replace(/\/$/, "") + (prefix.startsWith("/") ? prefix : "/" + prefix);
  }
  var FHIR_BASE =
    typeof window.FHIR_URL !== "undefined" ? window.FHIR_URL : origin.replace(/\/$/, "") + "/fhir";

  var linkFhir = document.getElementById("linkFhir");
  var linkApiDocs = document.getElementById("linkApiDocs");
  if (linkFhir) {
    linkFhir.href = FHIR_BASE.replace(/\/fhir\/?$/, "") || "http://localhost:8080";
  }
  if (linkApiDocs) {
    linkApiDocs.href = AI_BASE.replace(/\/$/, "") + "/docs";
  }

  function el(tag, attrs, children) {
    var n = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "text") n.textContent = attrs[k];
        else if (k === "html") n.innerHTML = attrs[k];
        else n.setAttribute(k, attrs[k]);
      });
    }
    (children || []).forEach(function (c) {
      if (c) n.appendChild(c);
    });
    return n;
  }

  function validPatientId(s) {
    return /^p-\d{1,8}$/.test(String(s || "").trim());
  }

  function loadMetrics() {
    var wrap = document.getElementById("metricsWrap");
    if (!wrap) return;
    wrap.className = "loading";
    wrap.textContent = "Cargando…";
    fetch(AI_BASE + "/metrics", { method: "GET", credentials: "same-origin" })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) {
            var d = body.detail;
            throw new Error(typeof d === "string" ? d : JSON.stringify(d || body));
          }
          return body;
        });
      })
      .then(function (data) {
        wrap.textContent = "";
        wrap.className = "";
        if (!Array.isArray(data) || data.length === 0) {
          wrap.appendChild(
            el("p", {
              class: "error-box",
              text: "No hay métricas. Ejecute training/run_all.py y vuelva a cargar.",
            })
          );
          return;
        }
        var table = el("table", { class: "data" });
        var thead = el("thead");
        var hr = el("tr");
        ["Modelo", "F1", "AUC-ROC", "Train (s)", "Infer/muestra (s)"].forEach(function (h) {
          hr.appendChild(el("th", { text: h }));
        });
        thead.appendChild(hr);
        table.appendChild(thead);
        var tb = el("tbody");
        data.forEach(function (r) {
          var tr = el("tr");
          [
            String(r.model || ""),
            fmt(r.f1),
            fmt(r.auc_roc),
            fmt(r.train_time_s),
            fmt(r.infer_time_per_sample_s),
          ].forEach(function (cell) {
            tr.appendChild(el("td", { text: cell }));
          });
          tb.appendChild(tr);
        });
        table.appendChild(tb);
        wrap.appendChild(table);
      })
      .catch(function (e) {
        wrap.className = "";
        wrap.textContent = "";
        wrap.appendChild(el("p", { class: "error-box", text: "Error: " + String(e.message || e) }));
      });
  }

  function fmt(v) {
    if (v === undefined || v === null) return "—";
    var s = String(v);
    return s.length > 12 ? s.slice(0, 12) : s;
  }

  function renderCompare(data) {
    var out = document.getElementById("compareOut");
    if (!out) return;
    out.textContent = "";
    if (data.error) {
      out.appendChild(el("p", { class: "error-box", text: String(data.error) }));
      return;
    }
    var preds = data.predictions || {};
    Object.keys(preds).forEach(function (key) {
      var block = preds[key];
      var card = el("article", { class: "model-card" });
      card.appendChild(el("h3", { text: key.replace(/_/g, " ") }));
      if (block.error) {
        card.appendChild(el("span", { class: "status-pill err", text: String(block.error) }));
        out.appendChild(card);
        return;
      }
      var p = Number(block.probability_positive);
      var pct = Math.round(Math.min(1, Math.max(0, p)) * 1000) / 10;
      var bar = el("div", { class: "bar" });
      bar.appendChild(el("i", { style: "width:" + pct + "%" }));
      card.appendChild(bar);
      card.appendChild(
        el("div", {
          class: "model-meta",
          text: "Prob. clase positiva: " + pct + "% · " + (block.label || ""),
        })
      );
      var pill = el("span", {
        class: "status-pill " + (block.predicted_class === 1 ? "pos" : "neg"),
        text: block.predicted_class === 1 ? "Predicción: enfermedad cardíaca" : "Predicción: clase negativa",
      });
      card.appendChild(pill);
      out.appendChild(card);
    });
    if (data.fhir_push) {
      var pre = el("pre", {
        style:
          "font-size:11px;background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;max-height:160px;margin-top:8px",
        text: JSON.stringify(data.fhir_push, null, 2),
      });
      out.appendChild(el("p", { class: "sub", text: "Respuesta FHIR (push):" }));
      out.appendChild(pre);
    }
  }

  function runCompare() {
    var raw = document.getElementById("patientId");
    var pid = raw ? raw.value.trim() : "p-0";
    if (!validPatientId(pid)) {
      renderCompare({ error: "ID inválido. Use el formato p-0, p-1, …" });
      return;
    }
    var push = document.getElementById("chkFhir") && document.getElementById("chkFhir").checked;
    var url = AI_BASE + "/compare/" + encodeURIComponent(pid);
    if (push) url += "?push_fhir=true";
    var out = document.getElementById("compareOut");
    if (out) {
      out.textContent = "";
      out.appendChild(el("p", { class: "loading", text: "Consultando modelos…" }));
    }
    fetch(url, { method: "GET", credentials: "same-origin" })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) {
            var d = body.detail;
            throw new Error(typeof d === "string" ? d : JSON.stringify(d || body));
          }
          return body;
        });
      })
      .then(renderCompare)
      .catch(function (e) {
        renderCompare({ error: String(e.message || e) });
      });
  }

  var btnM = document.getElementById("btnMetrics");
  var btnC = document.getElementById("btnCompare");
  if (btnM) btnM.addEventListener("click", loadMetrics);
  if (btnC) btnC.addEventListener("click", runCompare);

  /* Mapa Leaflet — demo */
  var cali = { lat: 3.4516, lng: -76.532 };
  var mapEl = document.getElementById("map");
  if (mapEl && typeof L !== "undefined") {
    var map = L.map("map").setView([cali.lat, cali.lng], 12);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "&copy; OpenStreetMap",
    }).addTo(map);
    var palette = ["#0d9488", "#2563eb", "#d97706", "#7c3aed", "#db2777", "#059669"];
    var layerGroup = L.layerGroup().addTo(map);

    function randAround(center, n, spread) {
      var pts = [];
      for (var i = 0; i < n; i++) {
        pts.push({
          lat: center.lat + (Math.random() - 0.5) * spread,
          lng: center.lng + (Math.random() - 0.5) * spread,
          id: "geo-" + i,
        });
      }
      return pts;
    }

    function kmeans(points, k) {
      if (!points.length) return [];
      k = Math.max(1, Math.min(k, points.length));
      var centroids = points.slice(0, k).map(function (p) {
        return { lat: p.lat, lng: p.lng };
      });
      var it, p, best, bestD, c, i, g;
      for (it = 0; it < 10; it++) {
        var groups = [];
        for (i = 0; i < k; i++) groups.push([]);
        for (p = 0; p < points.length; p++) {
          best = 0;
          bestD = Infinity;
          for (i = 0; i < k; i++) {
            c = centroids[i];
            var d = Math.hypot(points[p].lat - c.lat, points[p].lng - c.lng);
            if (d < bestD) {
              bestD = d;
              best = i;
            }
          }
          groups[best].push(points[p]);
        }
        centroids = groups.map(function (g) {
          if (!g.length) return centroids[0];
          return {
            lat: g.reduce(function (s, x) {
              return s + x.lat;
            }, 0) / g.length,
            lng: g.reduce(function (s, x) {
              return s + x.lng;
            }, 0) / g.length,
          };
        });
      }
      return points.map(function (p) {
        best = 0;
        bestD = Infinity;
        for (i = 0; i < k; i++) {
          c = centroids[i];
          var d2 = Math.hypot(p.lat - c.lat, p.lng - c.lng);
          if (d2 < bestD) {
            bestD = d2;
            best = i;
          }
        }
        return Object.assign({}, p, { cluster: best });
      });
    }

    function drawMap() {
      var ks = document.getElementById("kvalue");
      var k = ks ? parseInt(ks.value, 10) || 4 : 4;
      var clustered = kmeans(randAround(cali, 85, 0.085), k);
      layerGroup.clearLayers();
      clustered.forEach(function (p) {
        var col = palette[p.cluster % palette.length];
        L.circleMarker([p.lat, p.lng], {
          radius: 6,
          color: col,
          fillColor: col,
          fillOpacity: 0.85,
          weight: 1,
        })
          .bindPopup("ID " + p.id + "<br>Cluster " + p.cluster)
          .addTo(layerGroup);
      });
      var stats = document.getElementById("mapStats");
      if (stats) {
        stats.textContent = "Puntos: " + clustered.length + " · K = " + k;
      }
    }

    var rm = document.getElementById("refreshMap");
    if (rm) rm.addEventListener("click", drawMap);
    drawMap();
  }

  loadMetrics();
})();
