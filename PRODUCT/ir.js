/* =========================================================
   ir.js — Mesin Temu Kembali Informasi (Vector Space Model)
   ---------------------------------------------------------
   Alur:
     1. Preprocessing  : case folding → tokenisasi → stopword removal → stemming
     2. Indexing       : membangun indeks terbalik (inverted index) + tf
     3. Pembobotan     : TF-IDF,  w(t,d) = tf(t,d) × idf(t),  idf(t) = log10(N / df(t))
     4. Perankingan    : cosine similarity antara vektor query dan vektor dokumen
                         cos(q,d) = Σ w(t,q)·w(t,d) / ( |q| × |d| )
     5. Awalan judul   : judul yang diawali teks yang diketik ikut ditampilkan
                         (setelah hasil cosine), sehingga hasil muncul per huruf
   Dokumen = judul buku (pencarian nama buku).
   ========================================================= */

(function (global) {
  "use strict";

  // ---------------------------------------------------------
  // 1a. Daftar stopword bahasa Indonesia (+ beberapa kata Inggris)
  // ---------------------------------------------------------
  const STOPWORDS = new Set(`
    ada adalah agar akan aku amat anda antara apa apakah apalagi atau bagi bahwa begitu belum
    bersama bisa boleh dahulu dalam dan dapat dari daripada demi demikian dengan di dia dimana
    dll dsb dst engkau guna hal hanya harus ia ingin ini itu itulah jika juga kah kalau kamu
    karena ke kecuali kemana kenapa kepada ketika kita lagi lain maka mari masih melainkan
    menurut mengapa mereka namun nanti nya oleh pada para pula pun saat saja sambil sampai
    saya sebab sebagai sebelum secara sedangkan sehingga sekitar selagi selain sementara serta
    sesudah setelah setiap seperti si sudah supaya tanpa tapi telah tentang terhadap tetapi
    toh untuk walau yaitu yakni yang
    a an and at by for from in is of on the to with
  `.trim().split(/\s+/));

  // ---------------------------------------------------------
  // 1b'. Kata dasar bahasa Inggris (untuk judul buku terjemahan).
  //      Bentuk jamak "-s" dikembalikan ke bentuk tunggal hanya jika
  //      bentuk tunggalnya ada di kamus ini: brothers → brother.
  // ---------------------------------------------------------
  const KATA_DASAR_EN = new Set(`
    attack brother crime dragon note miserable piece prince punishment saga shore titan wood
  `.trim().split(/\s+/));

  // ---------------------------------------------------------
  // 1b. Kamus kata dasar untuk stemmer (Nazief–Adriani sederhana).
  //     Stemmer hanya memotong imbuhan jika hasilnya ada di kamus,
  //     sehingga kata dasar seperti "menara" atau "pelangi" tidak rusak.
  // ---------------------------------------------------------
  const KATA_DASAR = new Set(`
    air ajar alam amba anak angin api asuh atheis ayah ayat baca bahagia baik bangsa bangun
    bapak baru bawah benar besar biasa bintang biru buku bulan bumi bunga cahaya cantik cari
    cerita cinta damai dasar datang dengar desa dilan doa dukuh dunia edensor gadis gelap
    gunung habibie ainun harap hari harimau hati hidup hijau hilang hitam hujan hutan ibu
    iman indah ingat istri jalan jambu jantan jatuh jejak juni kaca kabah kambing kapal karpov
    kasih kata kembang kenang kertas kecil kisah koala kota kretek ksatria kumal kuning laut
    langit langkah larung laskar layar lelaki lihat lindung lupa luka main malam manusia
    marmut maryamah matahari menara merah milea mimpi muara murid negara negeri nurbaya orang
    pagi panjang pantai paruk pelangi perahu perang perempuan pergi pohon pulang puteri putih
    rahasia ranah rantau rindu ronggeng rumah sabtu sahabat sakit salah saman sayang sehat
    sekolah semua seni senyum siti suami suara sungai supernova tahun tanah tangis tasbih tawa
    teman temu tenggelam terang terbang terbit tidur tinggal tulis tunggu waktu warna
  `.trim().split(/\s+/));

  const VOWEL = /^[aiueo]/;

  // Kemungkinan bentuk kata setelah awalan dilepas (dengan aturan peluluhan meN-/peN-)
  function prefixCandidates(word) {
    const out = [];
    const add = (s) => { if (s.length >= 2) out.push(s); };
    const rules = [
      ["meng", (r) => [r, "k" + r]], ["peng", (r) => [r, "k" + r]],
      ["meny", (r) => ["s" + r]],    ["peny", (r) => ["s" + r]],
      ["mem",  (r) => VOWEL.test(r) ? ["m" + r, "p" + r] : [r]],
      ["pem",  (r) => VOWEL.test(r) ? ["m" + r, "p" + r] : [r]],
      ["men",  (r) => VOWEL.test(r) ? ["n" + r, "t" + r] : [r]],
      ["pen",  (r) => VOWEL.test(r) ? ["n" + r, "t" + r] : [r]],
      ["ber", (r) => [r]], ["ter", (r) => [r]], ["per", (r) => [r]],
      ["me", (r) => [r]],  ["pe", (r) => [r]],  ["be", (r) => [r]], ["te", (r) => [r]],
      ["di", (r) => [r]],  ["ke", (r) => [r]],  ["se", (r) => [r]],
    ];
    for (const [p, fn] of rules) {
      if (word.startsWith(p) && word.length > p.length + 1) fn(word.slice(p.length)).forEach(add);
    }
    return out;
  }

  // Lepas awalan secara bertahap (maksimal 3 lapis), berhenti saat ketemu kata dasar
  function stripPrefix(word, depth = 0) {
    if (KATA_DASAR.has(word)) return word;
    if (depth >= 3) return null;
    for (const c of prefixCandidates(word)) {
      const r = stripPrefix(c, depth + 1);
      if (r) return r;
    }
    return null;
  }

  // Kemungkinan bentuk kata setelah akhiran dilepas
  function suffixCandidates(word) {
    const forms = [word];
    let w = word;
    for (const p of ["lah", "kah", "tah", "pun"]) {           // partikel
      if (w.endsWith(p) && w.length > p.length + 2) { w = w.slice(0, -p.length); forms.push(w); break; }
    }
    for (const p of ["nya", "ku", "mu"]) {                     // kata ganti milik
      if (w.endsWith(p) && w.length > p.length + 2) { w = w.slice(0, -p.length); forms.push(w); break; }
    }
    for (const p of ["kan", "an", "i"]) {                      // akhiran turunan
      if (w.endsWith(p) && w.length > p.length + 2) { forms.push(w.slice(0, -p.length)); break; }
    }
    return forms;
  }

  const stemCache = new Map();
  function stem(word) {
    if (stemCache.has(word)) return stemCache.get(word);
    let result = word;
    if (KATA_DASAR_EN.has(word)) {
      result = word;
    } else if (word.endsWith("s") && KATA_DASAR_EN.has(word.slice(0, -1))) {
      result = word.slice(0, -1);                                       // jamak Inggris
    } else if (!KATA_DASAR.has(word) && !/^\d+$/.test(word)) {
      for (const form of suffixCandidates(word)) {
        const r = stripPrefix(form);
        if (r) { result = r; break; }
      }
    }
    stemCache.set(word, result);
    return result;
  }

  // huruf kecil + buang tanda aksen (Misérables → miserables)
  function fold(text) {
    return text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  }

  // ---------------------------------------------------------
  // 1c. Preprocessing lengkap, mengembalikan setiap tahap (untuk ditampilkan)
  // ---------------------------------------------------------
  function preprocess(text) {
    const folded = fold(text);                                          // case folding
    const tokens = folded.replace(/['’`]/g, "")                         // ka'bah → kabah
      .split(/[^a-z0-9]+/).filter(Boolean);                             // tokenisasi
    const removed = tokens.filter((t) => STOPWORDS.has(t));
    const kept = tokens.filter((t) => !STOPWORDS.has(t));               // stopword removal
    const stems = kept.map((t) => ({ token: t, stem: stem(t) }));       // stemming
    return { folded, tokens, removed, stems, terms: stems.map((s) => s.stem) };
  }

  // ---------------------------------------------------------
  // 2–3. Membangun indeks terbalik dan bobot TF-IDF
  // ---------------------------------------------------------
  function buildIndex(docs, field = "judul") {
    const N = docs.length;
    const tf = docs.map(() => new Map());      // tf[d] : term → frekuensi
    const index = new Map();                   // term → Map(docId → tf)

    docs.forEach((doc, d) => {
      for (const term of preprocess(doc[field]).terms) {
        tf[d].set(term, (tf[d].get(term) || 0) + 1);
      }
      for (const [term, f] of tf[d]) {
        if (!index.has(term)) index.set(term, new Map());
        index.get(term).set(d, f);
      }
    });

    const idf = new Map();
    for (const [term, postings] of index) idf.set(term, Math.log10(N / postings.size));

    // bobot dokumen dan panjang vektornya |d|
    const weights = tf.map((m) => {
      const w = new Map();
      for (const [term, f] of m) w.set(term, f * idf.get(term));
      return w;
    });
    const norms = weights.map((w) => Math.sqrt([...w.values()].reduce((s, x) => s + x * x, 0)));

    return { N, docs, field, index, idf, tf, weights, norms };
  }

  // ---------------------------------------------------------
  // 4. Pencarian: bobot query → cosine similarity → urutkan
  // ---------------------------------------------------------
  function search(idx, query) {
    const steps = preprocess(query);

    const qtf = new Map();
    for (const t of steps.terms) qtf.set(t, (qtf.get(t) || 0) + 1);

    const qWeights = new Map();
    const unknown = [];
    for (const [t, f] of qtf) {
      if (idx.idf.has(t)) qWeights.set(t, f * idx.idf.get(t));
      else unknown.push(t);
    }
    const qNorm = Math.sqrt([...qWeights.values()].reduce((s, x) => s + x * x, 0));

    // hanya dokumen yang memuat minimal satu term query (lewat indeks terbalik)
    const candidates = new Set();
    for (const t of qWeights.keys()) for (const d of idx.index.get(t).keys()) candidates.add(d);

    const results = [];
    for (const d of candidates) {
      let dot = 0;
      for (const [t, wq] of qWeights) dot += wq * (idx.weights[d].get(t) || 0);
      const denom = qNorm * idx.norms[d];
      const score = denom ? dot / denom : 0;
      if (score > 0) results.push({ doc: idx.docs[d], docId: d, score, dot, match: "kata" });
    }
    results.sort((a, b) => b.score - a.score || a.doc[idx.field].localeCompare(b.doc[idx.field]));

    // 5. Pencocokan awalan judul: judul yang diawali teks yang diketik
    //    (tidak peka huruf besar/kecil), agar hasil muncul sejak huruf pertama.
    //    Ditambahkan setelah hasil cosine, urut abjad.
    const prefix = fold(query.trim()).replace(/\s+/g, " ");
    const prefixMatches = [];
    if (prefix) {
      const found = new Set(results.map((r) => r.docId));
      idx.docs.forEach((doc, d) => {
        if (fold(doc[idx.field]).startsWith(prefix)) prefixMatches.push(d);
      });
      prefixMatches
        .filter((d) => !found.has(d))
        .sort((a, b) => idx.docs[a][idx.field].localeCompare(idx.docs[b][idx.field], "id"))
        .forEach((d) => results.push({ doc: idx.docs[d], docId: d, score: 0, dot: 0, match: "awalan" }));
    }

    return { steps, qtf, qWeights, qNorm, unknown, results, prefix, prefixMatches };
  }

  global.IR = { STOPWORDS, KATA_DASAR, KATA_DASAR_EN, fold, stem, preprocess, buildIndex, search };
  if (typeof module !== "undefined") module.exports = global.IR;
})(typeof window !== "undefined" ? window : globalThis);
