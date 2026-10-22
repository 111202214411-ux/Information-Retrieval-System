/* =========================================================
   script.js — Tampilan: rak hasil, detail buku, tahapan proses
   Logika temu kembali ada di ir.js, data buku di data.js.
   ========================================================= */

// Warna kain sampul untuk punggung & sampul pengganti
const BINDINGS = [
  { bg: "#7A2E2E", fg: "#F4E9D8" }, // merah marun
  { bg: "#2F4F6F", fg: "#EEF0E6" }, // biru tua
  { bg: "#3E6B48", fg: "#F1EEDC" }, // hijau
  { bg: "#C99A3B", fg: "#2A2014" }, // mustard
  { bg: "#5B3F6B", fg: "#F2EADF" }, // ungu
  { bg: "#2E6E6E", fg: "#EDF2E9" }, // teal
  { bg: "#A0522D", fg: "#F7EDDF" }, // tembaga
  { bg: "#3F444D", fg: "#E9E4D6" }, // abu batu
];
const COVER_DIR = "covers/";

// Indeks dibangun sekali saat halaman dibuka
const INDEX = IR.buildIndex(BOOKS, "judul");

// ---------- Elemen ----------
const $ = (id) => document.getElementById(id);
const form = $("searchForm");
const input = $("searchInput");
const clearBtn = $("clearBtn");
const statusEl = $("status");
const shelf = $("shelf");
const emptyEl = $("empty");
const detail = $("detail");
const placeholder = $("placeholder");
const bookEl = $("book");
const coverFig = $("cover");
const coverArt = $("coverArt");
const stepsEl = $("steps");
const processIntro = $("processIntro");

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let view = [];         // [{ doc, score }] yang sedang tampil di rak
let lastSearch = null; // hasil IR.search untuk query saat ini
let selected = null;

// ---------- Utilitas ----------
const fmt = (x, d = 4) => x.toLocaleString("id-ID", { minimumFractionDigits: d, maximumFractionDigits: d });

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function hash(text) {
  let h = 0;
  for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return h;
}

function bindingFor(book) {
  // warna berurutan agar buku yang bersebelahan tidak sewarna
  return BINDINGS[BOOKS.indexOf(book) % BINDINGS.length];
}

function lastName(name) {
  const parts = name.trim().split(/\s+/);
  return parts[parts.length - 1];
}

// Tandai kata di judul yang kata dasarnya sama dengan term query
function highlightTitle(title, terms, prefixLen = 0) {
  // cocok dari awalan judul → tandai awalannya
  if (prefixLen) {
    return `<mark>${escapeHtml(title.slice(0, prefixLen))}</mark>${escapeHtml(title.slice(prefixLen))}`;
  }
  if (!terms || !terms.size) return escapeHtml(title);
  return title.split(/([\p{L}\p{N}'’]+)/u).map((part, i) => {
    if (i % 2 === 0) return escapeHtml(part);
    const token = IR.fold(part).replace(/['’]/g, "");
    if (IR.STOPWORDS.has(token)) return escapeHtml(part);
    return terms.has(IR.stem(token)) ? `<mark>${escapeHtml(part)}</mark>` : escapeHtml(part);
  }).join("");
}

// ---------- Pencarian ----------
function runSearch() {
  const query = input.value.trim();

  if (!query) {
    lastSearch = null;
    view = BOOKS.map((doc) => ({ doc, score: null }));
  } else {
    lastSearch = IR.search(INDEX, query);
    view = lastSearch.results.map((r) => ({ doc: r.doc, score: r.score, dot: r.dot, docId: r.docId, match: r.match }));
  }

  renderStatus(query);
  renderShelf();
  renderSteps(query);

  if (selected) {
    const still = view.find((v) => v.doc === selected);
    still ? updateRank(still) : clearSelection();
  }
}

function renderStatus(query) {
  if (!query) {
    statusEl.innerHTML = `<strong>${BOOKS.length} buku</strong> di rak. Ketik kata dari judul untuk mencari.`;
    return;
  }
  const { steps, qWeights } = lastSearch;
  if (view.length) {
    statusEl.innerHTML = `<strong>${view.length} buku</strong> ditemukan untuk “${escapeHtml(query)}”.`;
  } else if (!steps.terms.length) {
    statusEl.textContent = "Semua kata pada kata kunci termasuk stopword. Coba kata lain dari judul.";
  } else if (!qWeights.size) {
    statusEl.innerHTML = `Kata “${escapeHtml(query)}” tidak ada di indeks judul.`;
  } else {
    statusEl.innerHTML = `Tidak ada judul yang relevan untuk “${escapeHtml(query)}”.`;
  }
}

// ---------- Rak ----------
function renderShelf() {
  const terms = lastSearch ? new Set(lastSearch.qWeights.keys()) : null;
  shelf.innerHTML = "";
  emptyEl.hidden = view.length > 0;

  view.forEach((item, rank) => {
    const book = item.doc;
    const { bg, fg } = bindingFor(book);
    const h = hash(book.judul + book.pengarang);
    // judul panjang / berkata panjang dibuat lebih tinggi agar tidak terpotong
    const longestWord = Math.max(...book.judul.split(/\s+/).map((w) => w.length));
    const drop = book.judul.length > 20 || longestWord >= 7 ? h % 6 : h % 34;

    const slot = document.createElement("div");
    slot.className = "slot";

    const spine = document.createElement("button");
    spine.type = "button";
    spine.className = "spine";
    spine.style.setProperty("--bg", bg);
    spine.style.setProperty("--fg", fg);
    spine.style.height = `calc(var(--row) - var(--plank) - 26px - ${drop}px)`;
    spine.setAttribute("aria-pressed", String(selected === book));
    spine.setAttribute("aria-label", item.score == null
      ? `${book.judul}, karya ${book.pengarang}`
      : `Peringkat ${rank + 1}: ${book.judul}, karya ${book.pengarang}, ` +
        (item.match === "awalan" ? "cocok dengan awalan judul" : `skor ${fmt(item.score)}`));
    spine.innerHTML = `
      <span class="spine__title">${highlightTitle(book.judul, terms, item.match === "awalan" ? lastSearch.prefix.length : 0)}</span>
      <span class="spine__author">${escapeHtml(lastName(book.pengarang))}</span>`;
    spine.addEventListener("click", () => selectBook(item, { scroll: true }));
    slot.appendChild(spine);

    // nomor peringkat ditempel di papan rak
    if (item.score != null) {
      const label = document.createElement("span");
      label.className = "slot__rank";
      label.textContent = rank + 1;
      label.setAttribute("aria-hidden", "true");
      slot.appendChild(label);
    }
    shelf.appendChild(slot);
  });
}

function markSelected() {
  shelf.querySelectorAll(".spine").forEach((el, i) => {
    el.setAttribute("aria-pressed", String(view[i]?.doc === selected));
  });
  stepsEl.querySelectorAll("tr[data-doc]").forEach((tr) => {
    tr.classList.toggle("is-selected", BOOKS[tr.dataset.doc] === selected);
  });
}

// ---------- Detail ----------
function updateRank(item) {
  const rankEl = $("bookRank");
  if (item.score == null) { rankEl.hidden = true; return; }
  const pos = view.indexOf(item) + 1;
  rankEl.hidden = false;
  rankEl.innerHTML = item.match === "awalan"
    ? `Peringkat <strong>${pos}</strong> dari ${view.length}, cocok dengan awalan judul`
    : `Peringkat <strong>${pos}</strong> dari ${view.length}, skor cosine <strong>${fmt(item.score)}</strong>`;
}

function selectBook(item, { scroll = false } = {}) {
  const book = item.doc;
  selected = book;
  markSelected();

  const { bg, fg } = bindingFor(book);
  coverArt.style.setProperty("--bg", bg);
  coverArt.style.setProperty("--fg", fg);
  $("coverTitle").textContent = book.judul;
  $("coverAuthor").textContent = book.pengarang;
  coverArt.hidden = false;

  // sampul asli dari folder covers/, jika ada
  coverFig.querySelector("img")?.remove();
  if (book.cover) {
    const img = new Image();
    img.alt = `Sampul buku ${book.judul}`;
    img.hidden = true;
    img.onload = () => { if (selected === book) { img.hidden = false; coverArt.hidden = true; } };
    img.onerror = () => img.remove();
    img.src = COVER_DIR + book.cover;
    coverFig.prepend(img);
  }

  updateRank(item);
  $("bookTitle").textContent = book.judul;
  $("fJudul").textContent = book.judul;
  $("fPengarang").textContent = book.pengarang;
  $("fGenre").innerHTML = book.genre.map((g) => `<span class="tag">${escapeHtml(g)}</span>`).join("");
  $("fTahun").textContent = book.tahun;
  $("fPenerbit").textContent = book.penerbit;

  placeholder.hidden = true;
  bookEl.hidden = false;

  // animasi "ditarik dari rak"
  bookEl.classList.remove("is-pulled");
  void bookEl.offsetWidth;
  bookEl.classList.add("is-pulled");

  if (scroll && window.matchMedia("(max-width: 899px)").matches) {
    detail.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
  }
}

function clearSelection() {
  selected = null;
  markSelected();
  bookEl.hidden = true;
  placeholder.hidden = false;
}

// ---------- Tahapan proses ----------
const chip = (text, cls = "") => `<span class="chip ${cls}">${escapeHtml(text)}</span>`;

function renderSteps(query) {
  if (!query) {
    processIntro.textContent = "Ketik kata kunci untuk melihat tahapan temu kembali yang dijalankan: preprocessing, pembobotan TF-IDF, perankingan dengan cosine similarity, lalu pencocokan awalan judul.";
    stepsEl.hidden = true;
    stepsEl.innerHTML = "";
    return;
  }

  const { steps, qtf, qWeights, qNorm, unknown, prefix, prefixMatches } = lastSearch;
  processIntro.innerHTML = `Tahapan untuk kata kunci “${escapeHtml(query)}”. Koleksi berisi N = ${INDEX.N} judul buku.`;
  stepsEl.hidden = false;

  const changed = steps.stems.filter((s) => s.token !== s.stem);

  const weightRows = [...qtf.keys()].map((t) => {
    const known = INDEX.idf.has(t);
    const df = known ? INDEX.index.get(t).size : 0;
    return `<tr${known ? "" : ' class="is-muted"'}>
      <td>${escapeHtml(t)}</td>
      <td class="num">${qtf.get(t)}</td>
      <td class="num">${known ? df : "–"}</td>
      <td class="num">${known ? fmt(INDEX.idf.get(t)) : "–"}</td>
      <td class="num">${known ? fmt(qWeights.get(t)) : "tidak di indeks"}</td>
    </tr>`;
  }).join("");

  const rankRows = view.map((v, i) => v.match !== "kata" ? "" : `
    <tr data-doc="${v.docId}"${v.doc === selected ? ' class="is-selected"' : ""}>
      <td class="num">${i + 1}</td>
      <td><button type="button" class="link-btn" data-open="${v.docId}">${escapeHtml(v.doc.judul)}</button><span class="muted"> ${escapeHtml(v.doc.pengarang)}</span></td>
      <td class="num">${v.match === "awalan" ? "–" : fmt(v.dot)}</td>
      <td class="num">${v.match === "awalan" ? "–" : fmt(INDEX.norms[v.docId])}</td>
      <td class="num">${v.match === "awalan" ? '<span class="muted">awalan</span>' : `<strong>${fmt(v.score)}</strong>`}</td>
    </tr>`).join("");

  stepsEl.innerHTML = `
    <li class="step">
      <h3>Case folding</h3>
      <p>Semua huruf diubah menjadi huruf kecil.</p>
      <div class="chips">${chip(steps.folded, "chip--wide")}</div>
    </li>
    <li class="step">
      <h3>Tokenisasi</h3>
      <p>Teks dipecah menjadi kata; tanda baca dibuang.</p>
      <div class="chips">${steps.tokens.map((t) => chip(t)).join("") || '<span class="muted">tidak ada token</span>'}</div>
    </li>
    <li class="step">
      <h3>Stopword removal</h3>
      <p>Kata umum yang tidak membedakan judul dihapus.</p>
      <div class="chips">${steps.tokens.map((t) => IR.STOPWORDS.has(t) ? chip(t, "chip--removed") : chip(t)).join("") || '<span class="muted">tidak ada token</span>'}</div>
    </li>
    <li class="step">
      <h3>Stemming</h3>
      <p>Imbuhan dilepas sehingga kata kembali ke bentuk dasarnya.</p>
      <div class="chips">${steps.stems.map((s) => s.token === s.stem ? chip(s.stem) : `<span class="chip chip--stem">${escapeHtml(s.token)} <span aria-hidden="true">›</span><span class="visually-hidden">menjadi</span> <strong>${escapeHtml(s.stem)}</strong></span>`).join("") || '<span class="muted">tidak ada term tersisa</span>'}</div>
      ${changed.length ? "" : '<p class="muted small">Tidak ada kata berimbuhan pada kata kunci ini.</p>'}
    </li>
    <li class="step">
      <h3>Bobot TF-IDF query</h3>
      <p class="formula">w = tf × idf,&nbsp; idf = log<sub>10</sub>(N / df)</p>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th scope="col">Term</th><th scope="col" class="num">tf</th><th scope="col" class="num">df</th><th scope="col" class="num">idf</th><th scope="col" class="num">w(q)</th></tr></thead>
          <tbody>${weightRows || '<tr><td colspan="5" class="muted">Tidak ada term.</td></tr>'}</tbody>
        </table>
      </div>
      ${qWeights.size ? `<p class="small muted">Panjang vektor query |q| = ${fmt(qNorm)}</p>` : ""}
      ${unknown.length ? `<p class="small muted">Term yang tidak ada di indeks diabaikan.</p>` : ""}
    </li>
    <li class="step">
      <h3>Cosine similarity &amp; peringkat</h3>
      <p class="formula">cos(q, d) = Σ w(q)·w(d) / (|q| × |d|)</p>
      ${view.some((v) => v.match === "kata") ? `<div class="table-wrap">
        <table class="data-table">
          <thead><tr><th scope="col" class="num">#</th><th scope="col">Judul</th><th scope="col" class="num">Σ w(q)·w(d)</th><th scope="col" class="num">|d|</th><th scope="col" class="num">Skor</th></tr></thead>
          <tbody>${rankRows}</tbody>
        </table>
      </div>` : '<p class="muted">Tidak ada dokumen dengan skor di atas 0.</p>'}
    </li>
    <li class="step">
      <h3>Pencocokan awalan judul</h3>
      <p>Judul yang diawali “${escapeHtml(prefix)}” (huruf besar/kecil tidak dibedakan) ikut ditampilkan, agar hasil sudah muncul sejak huruf pertama diketik. Yang belum ditemukan lewat cosine ditambahkan di akhir peringkat, urut abjad.</p>
      <div class="chips">${prefixMatches.length
        ? [...prefixMatches].sort((x, y) => view.findIndex((v) => v.docId === x) - view.findIndex((v) => v.docId === y)).map((d) => {
            const pos = view.findIndex((v) => v.docId === d) + 1;
            const already = view[pos - 1]?.match === "kata";
            return `<span class="chip"><mark>${escapeHtml(BOOKS[d].judul.slice(0, prefix.length))}</mark>${escapeHtml(BOOKS[d].judul.slice(prefix.length))} <span class="muted">#${pos}${already ? ", sudah dari cosine" : ""}</span></span>`;
          }).join("")
        : '<span class="muted">Tidak ada judul yang diawali teks ini.</span>'}</div>
    </li>`;
}

// ---------- Tabel indeks terbalik ----------
function renderIndexTable() {
  const rows = [...INDEX.index.entries()]
    .sort((a, b) => a[0].localeCompare(b[0], "id"))
    .map(([term, postings]) => {
      const docs = [...postings.entries()].map(([d, f]) =>
        `${escapeHtml(BOOKS[d].judul)}${f > 1 ? ` <span class="muted">(tf ${f})</span>` : ""}`).join("; ");
      return `<tr><td>${escapeHtml(term)}</td><td class="num">${postings.size}</td><td class="num">${fmt(INDEX.idf.get(term))}</td><td>${docs}</td></tr>`;
    }).join("");
  $("indexTable").querySelector("tbody").innerHTML = rows;
  $("indexMeta").textContent = `(${INDEX.index.size} term dari ${INDEX.N} judul)`;
}

// ---------- Event ----------
input.addEventListener("input", () => {
  clearBtn.hidden = input.value === "";
  runSearch();
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (view.length) {
    selectBook(view[0], { scroll: true });
    shelf.querySelector(".spine")?.focus({ preventScroll: true });
  }
});

clearBtn.addEventListener("click", () => {
  input.value = "";
  clearBtn.hidden = true;
  runSearch();
  input.focus();
});

$("returnBtn").addEventListener("click", () => {
  const index = view.findIndex((v) => v.doc === selected);
  clearSelection();
  const spine = shelf.querySelectorAll(".spine")[index];
  if (spine) {
    spine.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "center" });
    spine.focus({ preventScroll: true });
  }
});

// klik judul di tabel peringkat → buka bukunya
stepsEl.addEventListener("click", (e) => {
  const id = e.target.closest("[data-open]")?.dataset.open;
  if (id == null) return;
  const item = view.find((v) => v.docId === Number(id));
  if (!item) return;
  selectBook(item);
  detail.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
});

emptyEl.addEventListener("click", (e) => {
  const word = e.target.dataset?.suggest;
  if (!word) return;
  input.value = word;
  clearBtn.hidden = false;
  runSearch();
  input.focus();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && selected) clearSelection();
  if (e.key === "/" && document.activeElement !== input) {
    e.preventDefault();
    input.focus();
  }
});

renderIndexTable();
runSearch();
