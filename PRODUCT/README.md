# Rak Buku — Mesin Pencarian Judul Buku

Tugas Sistem Temu Kembali Informasi (STKI). Mesin pencarian judul buku dengan
Vector Space Model: preprocessing, indeks terbalik, pembobotan TF-IDF, dan
perankingan cosine similarity, ditambah pencocokan awalan judul.

## Isi folder

| File | Isi |
|---|---|
| `pencarian_buku.py` | Aplikasi Streamlit (tampilan) |
| `ir.py` | Mesin temu kembali: preprocessing, stemming, indeks terbalik, TF-IDF, cosine |
| `buku.json` | Korpus: 82 buku (yang diindeks: teks judul) |
| `covers/` | Gambar sampul (opsional), nama file sesuai field `cover` |
| `requirements.txt` | Paket Python yang dibutuhkan |
| `.streamlit/config.toml` | Tema warna aplikasi |

## Menjalankan di komputer sendiri

```bash
pip install -r requirements.txt
streamlit run pencarian_buku.py
```

Menguji mesin IR saja dari terminal:

```bash
python ir.py "hujan bulan yang menceritakan"
```

## Deploy ke Streamlit Community Cloud

1. Upload semua isi folder ini ke repository GitHub (termasuk folder `.streamlit`).
2. Di share.streamlit.io, buat app dari repository tersebut dengan
   **Main file path** = `pencarian_buku.py`.
3. Jika app sudah pernah dibuat, cukup ganti file di repository; app akan
   memperbarui dirinya sendiri (atau pilih **Reboot app**).
