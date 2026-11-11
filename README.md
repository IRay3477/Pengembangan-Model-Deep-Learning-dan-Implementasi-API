# Peringkasan Otomatis Transkrip Video Kuliah Berbahasa Indonesia

Peringkasan **abstraktif** transkrip video kuliah berbahasa Indonesia dengan **fine-tuning model Transformer seq2seq**. Model berjalan mandiri tanpa layanan AI pihak ketiga dan nantinya dilayani lewat REST API (FastAPI).

Tugas 2 Mata Kuliah **Deep Learning** — Program Studi Teknologi Informasi, Universitas Udayana.

**Kelompok 6**
- I Made Ray Adhya Khrisna
- Ida Bagus Komang Adi Prayista
- Nolan Mahotama Abipantara Urya Gotama

---

## Status Progres

| Tahap | Isi | Status |
|---|---|---|
| 1 | Profiling dataset + uji 3 model translasi EN → ID | ✅ Selesai |
| 2 | Cleaning, penggabungan Part I/II, split 80/10/10 per grup topik | ✅ Selesai |
| 3 | Translasi penuh EN → ID (NLLB-200) + kontrol kualitas | ✅ Selesai |
| 4 | Evaluasi ROUGE + 3 baseline + truncation vs chunk-and-summarize | ✅ Selesai |
| 5 | Seleksi model utama (mT5-base, mBART-50, IndoBART-v2) | ✅ Selesai → **IndoBART-v2** |
| 6 | Fine-tuning penuh IndoBART-v2 | ⏳ Berikutnya |
| 7 | Evaluasi akhir di set test + error analysis | ⏳ Rencana |
| 8 | REST API FastAPI (`POST /summarize`) + uji Postman | ⏳ Rencana |

---

## Dataset

- **Sumber:** `ak_lectures_summarized.csv`, subset AK Lectures dari dataset **SSLectures** (pasangan transkrip kuliah dan ringkasannya, berbahasa Inggris).
- **Isi:** 1.871 video dari 6 mata kuliah (Classical physics, Organic chemistry, Biochemistry, Biology, Modern physics, General chemistry).
- **Setelah cleaning:** 1.749 pasangan → **train 1.399 / val 175 / test 175**.
- Data **tidak disertakan** di repo (lihat `.gitignore`). Letakkan file CSV di Google Drive sesuai bagian [Cara Menjalankan](#cara-menjalankan).

### Alur pengolahan data

```
1.871 data awal
  → 1.842  buang transkrip / ringkasan kosong
  → 1.802  gabung Part I + Part II yang ringkasannya sama (39 baris gabungan)
  → 1.759  buang anomali (ringkasan < 20 kata / lebih panjang dari transkrip)
  → 1.749  buang ringkasan kembar beda topik
  → split per grup topik (GroupShuffleSplit, seed 42), irisan antar split = 0
  → translasi EN → ID per kalimat dengan NLLB-200 (distilled 600M)
```

Kontrol kualitas translasi: 0 baris gagal, sisa kata Inggris 0,02 per transkrip, 1 kasus halusinasi (repetition loop) diterjemahkan ulang, dan 3 istilah dikoreksi (*ceramah → kuliah*, *lapisan kuman → lapisan germinal*, *konstan gravitasi → konstanta gravitasi*).

Setelah translasi, **99%** transkrip lebih panjang dari 512 token (median 1.458 token mT5), sedangkan 97,7% ringkasan muat dalam 512 token.

---

## Hasil Sementara

### Baseline (set test, 175 video)

ROUGE-1/2/L F1, stemmer dimatikan (stemmer bawaan library hanya untuk bahasa Inggris).

| Metode | ROUGE-1 | ROUGE-2 | ROUGE-L | Panjang |
|---|---|---|---|---|
| **Lead-7** | **37,66** | **13,52** | **21,79** | 127 kata |
| TextRank-5 | 34,65 | 10,58 | 19,32 | 171 kata |
| Zero-shot T5 (truncation) | 27,60 | 9,94 | 18,07 | 45 kata |
| Zero-shot T5 (chunk-and-summarize) | 28,31 | 9,94 | 18,21 | 49 kata |

Zero-shot memakai `cahya/t5-base-indonesian-summarization-cased` (dilatih pada berita) tanpa fine-tuning. Nilai k untuk Lead-k dan TextRank dipilih di set val.

### Seleksi model utama (subset 300 data train × 2 epoch, evaluasi di set val)

| Model | Parameter | VRAM puncak | Menit/epoch | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|
| `google/mt5-base` | 582 jt | 7,8 GB | 2,83 | 22,37 | 7,24 | 14,87 |
| `facebook/mbart-large-50` | 611 jt | 8,5 GB | 1,28 | 32,35 | 12,94 | 21,48 |
| **`indobenchmark/indobart-v2`** | **132 jt** | **2,7 GB** | **0,41** | **33,87** | 12,31 | **21,81** |

**IndoBART-v2** dipilih karena ROUGE-nya setara mBART-50, tetapi 4,6× lebih kecil, memorinya 3× lebih hemat, dan 3× lebih cepat dilatih. Skor seleksi dihitung di set val, jadi tidak bisa dibandingkan langsung dengan tabel baseline (set test).

> Hasil akhir model fine-tuned di set test akan ditambahkan setelah Tahap 6–7 selesai.

---

## Struktur Repository

```
.
├── Deep_Learning_Peringkasan_Otomatis_Video_Kuliah_Bahasa_Indonesia.ipynb   # notebook utama (Tahap 1–5)
├── tahap1_profiling_dan_uji_terjemahan.py   # versi skrip Tahap 1
├── tahap2_cleaning_dan_split.py             # versi skrip Tahap 2
├── requirements.txt
└── README.md
```

Notebook adalah acuan utama dan versi paling mutakhir dari seluruh pipeline.

---

## Cara Menjalankan

Notebook dirancang untuk **Google Colab dengan GPU T4**.

1. Buka notebook di Google Colab, lalu pilih **Runtime → Change runtime type → T4 GPU**.
2. Siapkan folder di Google Drive:
   ```
   MyDrive/DEEP_LEARNING_test/
   └── Dataset/
       └── ak_lectures_summarized.csv
   ```
   Kalau memakai nama folder lain, ubah variabel `folder_utama` di cell pertama.
3. Jalankan **Runtime → Run all**.

Semua hasil antara disimpan ke `MyDrive/DEEP_LEARNING_test/tahap1` sampai `tahap5`.

### Melanjutkan setelah sesi Colab terputus

Cell **"Cek Hasil di Drive"** di awal notebook memeriksa file hasil tiap tahap. Proses berat (uji translasi, translasi NLLB, prediksi zero-shot) **tidak dijalankan ulang** bila hasilnya sudah ada di Drive, sedangkan translasi penuh dan seleksi model melanjutkan dari titik terakhir. Cukup **Run all** lagi.

Untuk sengaja menjalankan ulang suatu proses, hapus file hasilnya di Drive, lalu jalankan ulang cell "Cek Hasil di Drive" dan cell proses tersebut.

### Instalasi lokal (opsional)

```bash
pip install -r requirements.txt
```

---

## Catatan Teknis

- **Tokenizer IndoBART.** Tokenizer resmi (`IndoNLGTokenizer` dari paket `indobenchmark-toolkit`) tidak kompatibel dengan transformers 5.x. Notebook memakai kelas `TokenizerIndoBART` dengan format yang sama (teks huruf kecil, `<s> teks </s> [indonesian]`). Kelas ini diverifikasi menghasilkan `input_ids` dan `labels` identik dengan tokenizer resmi pada 525 teks.
- Keluaran IndoBART selalu **huruf kecil**, karena model dilatih pada teks huruf kecil.
- Seed **42** dipakai untuk split dan sampling. Saat dijalankan ulang dari nol, seluruh angka Tahap 1–4 keluar identik.

## Keterbatasan

- Data latih adalah hasil **translasi mesin** dari bahasa Inggris, bukan transkrip asli berbahasa Indonesia, sehingga masih mengandung sebagian kesalahan istilah dan noise pengenalan suara dari transkrip sumber.
- Dataset hanya menyediakan ringkasan per video, bukan per potongan transkrip.
- Seleksi model memakai subset kecil (300 data × 2 epoch), sehingga selisih kecil antarmodel belum tentu signifikan.

---

## Rencana Berikutnya

- [ ] Fine-tuning penuh IndoBART-v2 (input 512 vs 1.024 token, 2 nilai learning rate, early stopping)
- [ ] Bobot model terbaik diunggah ke Google Drive (link akan ditambahkan di sini)
- [ ] Evaluasi di set test + error analysis (konsep hilang, pengulangan, kalimat ambigu, halusinasi)
- [ ] REST API FastAPI: `POST /summarize` dengan chunking otomatis untuk teks panjang, validasi HTTP 422, diuji via Postman
