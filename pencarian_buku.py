"""
Mesin Pencarian Nama Buku (GUI)
-------------------------------
- Ketik judul buku di kolom pencarian (hasil langsung tersaring saat mengetik).
- Klik salah satu judul pada daftar hasil untuk menampilkan cover + detail buku.

Kebutuhan:
    pip install pillow
    (Linux Mint/Ubuntu, jika tkinter belum ada: sudo apt install python3-tk python3-pil.imagetk)

Cover:
    Simpan gambar cover di folder "covers/" (sejajar dengan file ini) dengan nama
    sesuai field "cover" pada data buku, misalnya covers/laskar_pelangi.jpg.
    Jika file cover tidak ada, program otomatis membuat cover sederhana sebagai pengganti.
"""

import os
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COVER_DIR = os.path.join(BASE_DIR, "covers")
COVER_SIZE = (220, 320)

# ---------------------------------------------------------------------------
# Data buku (contoh). Silakan tambah/ubah sesuai kebutuhan.
# ---------------------------------------------------------------------------
BOOKS = [
    {"judul": "Laskar Pelangi", "pengarang": "Andrea Hirata", "genre": "Novel, Drama",
     "tahun": 2005, "penerbit": "Bentang Pustaka", "cover": "laskar_pelangi.jpg"},
    {"judul": "Sang Pemimpi", "pengarang": "Andrea Hirata", "genre": "Novel, Drama",
     "tahun": 2006, "penerbit": "Bentang Pustaka", "cover": "sang_pemimpi.jpg"},
    {"judul": "Bumi Manusia", "pengarang": "Pramoedya Ananta Toer", "genre": "Novel Sejarah",
     "tahun": 1980, "penerbit": "Hasta Mitra", "cover": "bumi_manusia.jpg"},
    {"judul": "Ronggeng Dukuh Paruk", "pengarang": "Ahmad Tohari", "genre": "Novel, Drama",
     "tahun": 1982, "penerbit": "Gramedia Pustaka Utama", "cover": "ronggeng_dukuh_paruk.jpg"},
    {"judul": "Negeri 5 Menara", "pengarang": "Ahmad Fuadi", "genre": "Novel, Inspiratif",
     "tahun": 2009, "penerbit": "Gramedia Pustaka Utama", "cover": "negeri_5_menara.jpg"},
    {"judul": "Perahu Kertas", "pengarang": "Dee Lestari", "genre": "Novel, Romansa",
     "tahun": 2009, "penerbit": "Bentang Pustaka", "cover": "perahu_kertas.jpg"},
    {"judul": "Ayat-Ayat Cinta", "pengarang": "Habiburrahman El Shirazy", "genre": "Novel, Religi",
     "tahun": 2004, "penerbit": "Republika", "cover": "ayat_ayat_cinta.jpg"},
    {"judul": "Pulang", "pengarang": "Tere Liye", "genre": "Novel, Aksi",
     "tahun": 2015, "penerbit": "Republika", "cover": "pulang.jpg"},
    {"judul": "Hujan", "pengarang": "Tere Liye", "genre": "Novel, Fiksi Ilmiah",
     "tahun": 2016, "penerbit": "Gramedia Pustaka Utama", "cover": "hujan.jpg"},
    {"judul": "Laut Bercerita", "pengarang": "Leila S. Chudori", "genre": "Novel Sejarah",
     "tahun": 2017, "penerbit": "Kepustakaan Populer Gramedia", "cover": "laut_bercerita.jpg"},
    {"judul": "Dilan: Dia adalah Dilanku Tahun 1990", "pengarang": "Pidi Baiq", "genre": "Novel, Romansa",
     "tahun": 2014, "penerbit": "Pastel Books", "cover": "dilan_1990.jpg"},
]

# Palet warna untuk cover pengganti
PALETTE = ["#2E5E8C", "#8C2E3F", "#2E8C5E", "#8C6A2E", "#5E2E8C", "#2E7F8C", "#8C4A2E"]


# ---------------------------------------------------------------------------
# Logika pencarian & cover
# ---------------------------------------------------------------------------
def cari_buku(kata_kunci):
    """Mengembalikan daftar buku yang judulnya mengandung kata kunci (tidak peka huruf besar/kecil)."""
    kunci = kata_kunci.strip().lower()
    if not kunci:
        return list(BOOKS)
    return [b for b in BOOKS if kunci in b["judul"].lower()]


def _font(size):
    for nama in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(nama, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bungkus_teks(draw, teks, font, lebar_maks):
    kata, baris, sekarang = teks.split(), [], ""
    for k in kata:
        coba = (sekarang + " " + k).strip()
        if draw.textlength(coba, font=font) <= lebar_maks:
            sekarang = coba
        else:
            if sekarang:
                baris.append(sekarang)
            sekarang = k
    if sekarang:
        baris.append(sekarang)
    return baris


def buat_cover_pengganti(buku):
    """Membuat gambar cover sederhana bila file cover asli tidak tersedia."""
    w, h = COVER_SIZE
    warna = PALETTE[sum(map(ord, buku["judul"])) % len(PALETTE)]
    img = Image.new("RGB", (w, h), warna)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, w - 11, h - 11], outline="white", width=2)

    f_judul, f_kecil = _font(22), _font(13)
    y = 60
    for baris in _bungkus_teks(d, buku["judul"].upper(), f_judul, w - 40):
        d.text((w / 2, y), baris, font=f_judul, fill="white", anchor="mm")
        y += 28
    d.line([(50, y + 6), (w - 50, y + 6)], fill="white", width=1)
    baris_pengarang = _bungkus_teks(d, buku["pengarang"], f_kecil, w - 40)
    y = h - 40 - 18 * (len(baris_pengarang) - 1)
    for baris in baris_pengarang:
        d.text((w / 2, y), baris, font=f_kecil, fill="white", anchor="mm")
        y += 18
    return img


def muat_cover(buku):
    path = os.path.join(COVER_DIR, buku.get("cover", ""))
    if buku.get("cover") and os.path.isfile(path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(COVER_SIZE)
            return img
        except OSError:
            pass
    return buat_cover_pengganti(buku)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class AplikasiPencarianBuku(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mesin Pencarian Buku")
        self.geometry("900x600")
        self.minsize(820, 560)
        self.configure(bg="#F4F5F7")

        self.hasil = []
        self._cover_tk = None  # simpan referensi agar gambar tidak hilang

        self._gaya()
        self._bangun_ui()
        self.perbarui_hasil()

    def _gaya(self):
        s = ttk.Style(self)
        if "clam" in s.theme_names():
            s.theme_use("clam")
        s.configure("TFrame", background="#F4F5F7")
        s.configure("Card.TFrame", background="white")
        s.configure("Judul.TLabel", background="#F4F5F7", font=("Helvetica", 16, "bold"))
        s.configure("Info.TLabel", background="#F4F5F7", foreground="#666")
        s.configure("Label.TLabel", background="white", foreground="#777", font=("Helvetica", 10))
        s.configure("Nilai.TLabel", background="white", font=("Helvetica", 11, "bold"))
        s.configure("Buku.TLabel", background="white", font=("Helvetica", 15, "bold"))

    def _bangun_ui(self):
        # --- Header & kolom pencarian ---
        atas = ttk.Frame(self, padding=(16, 14, 16, 6))
        atas.pack(fill="x")
        ttk.Label(atas, text="📚  Mesin Pencarian Buku", style="Judul.TLabel").pack(anchor="w")

        baris_cari = ttk.Frame(atas)
        baris_cari.pack(fill="x", pady=(10, 0))
        self.var_cari = tk.StringVar()
        self.var_cari.trace_add("write", lambda *_: self.perbarui_hasil())
        entri = ttk.Entry(baris_cari, textvariable=self.var_cari, font=("Helvetica", 12))
        entri.pack(side="left", fill="x", expand=True, ipady=4)
        entri.bind("<Return>", lambda e: self.pilih_pertama())
        entri.focus()
        ttk.Button(baris_cari, text="Cari", command=self.pilih_pertama).pack(side="left", padx=(8, 0))
        ttk.Button(baris_cari, text="Reset", command=lambda: self.var_cari.set("")).pack(side="left", padx=(6, 0))

        self.lbl_status = ttk.Label(atas, text="", style="Info.TLabel")
        self.lbl_status.pack(anchor="w", pady=(6, 0))

        # --- Isi: daftar hasil (kiri) & detail (kanan) ---
        isi = ttk.Frame(self, padding=(16, 4, 16, 16))
        isi.pack(fill="both", expand=True)

        kiri = ttk.Frame(isi)
        kiri.pack(side="left", fill="y")
        self.daftar = tk.Listbox(kiri, width=28, font=("Helvetica", 11), activestyle="none",
                                 selectbackground="#2E5E8C", selectforeground="white",
                                 relief="flat", highlightthickness=1, highlightcolor="#CCC")
        scroll = ttk.Scrollbar(kiri, orient="vertical", command=self.daftar.yview)
        self.daftar.configure(yscrollcommand=scroll.set)
        self.daftar.pack(side="left", fill="y")
        scroll.pack(side="left", fill="y")
        self.daftar.bind("<<ListboxSelect>>", self.tampilkan_detail)

        self.kartu = ttk.Frame(isi, style="Card.TFrame", padding=16)
        self.kartu.pack(side="left", fill="both", expand=True, padx=(14, 0))

        self.lbl_cover = tk.Label(self.kartu, bg="white")
        self.lbl_cover.pack(side="left", anchor="n")

        info = ttk.Frame(self.kartu, style="Card.TFrame", padding=(18, 0, 0, 0))
        info.pack(side="left", fill="both", expand=True, anchor="n")
        self.lbl_judul = ttk.Label(info, text="", style="Buku.TLabel", wraplength=260)
        self.lbl_judul.pack(anchor="w", pady=(0, 12))

        self.nilai = {}
        for kunci, label in [("judul", "Nama Buku"), ("pengarang", "Pengarang"),
                             ("genre", "Genre"), ("tahun", "Tahun Terbit"),
                             ("penerbit", "Penerbit")]:
            ttk.Label(info, text=label, style="Label.TLabel").pack(anchor="w")
            lbl = ttk.Label(info, text="-", style="Nilai.TLabel", wraplength=260)
            lbl.pack(anchor="w", pady=(0, 8))
            self.nilai[kunci] = lbl

        self.kosongkan_detail("Klik judul buku di daftar untuk melihat detailnya.")

    # --- Aksi ---
    def perbarui_hasil(self):
        kata = self.var_cari.get()
        self.hasil = cari_buku(kata)
        self.daftar.delete(0, tk.END)
        for b in self.hasil:
            self.daftar.insert(tk.END, f"  {b['judul']}")

        if not self.hasil:
            self.lbl_status.configure(text=f'Buku "{kata.strip()}" tidak ditemukan.', foreground="#B03A2E")
            self.kosongkan_detail("Tidak ada buku yang cocok.")
        elif kata.strip():
            self.lbl_status.configure(text=f"{len(self.hasil)} buku ditemukan.", foreground="#2E7D32")
        else:
            self.lbl_status.configure(text=f"Menampilkan semua buku ({len(self.hasil)}).", foreground="#666")

    def pilih_pertama(self):
        if self.hasil:
            self.daftar.selection_clear(0, tk.END)
            self.daftar.selection_set(0)
            self.daftar.see(0)
            self.tampilkan_detail()

    def tampilkan_detail(self, event=None):
        pilihan = self.daftar.curselection()
        if not pilihan:
            return
        buku = self.hasil[pilihan[0]]

        self._cover_tk = ImageTk.PhotoImage(muat_cover(buku))
        self.lbl_cover.configure(image=self._cover_tk, text="", width=0, height=0)

        self.lbl_judul.configure(text=buku["judul"])
        for kunci, lbl in self.nilai.items():
            lbl.configure(text=str(buku[kunci]))

    def kosongkan_detail(self, pesan):
        self._cover_tk = None
        self.lbl_cover.configure(image="", text="(cover)", fg="#BBB",
                                 width=28, height=18, font=("Helvetica", 10))
        self.lbl_judul.configure(text=pesan)
        for lbl in self.nilai.values():
            lbl.configure(text="-")


if __name__ == "__main__":
    AplikasiPencarianBuku().mainloop()
