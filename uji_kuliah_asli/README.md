# Uji API dengan Transkrip Kuliah Asli Berbahasa Indonesia

Seluruh data latih dan uji kami adalah hasil terjemahan mesin (NLLB-200). Supaya terlihat perilaku model pada bahasa Indonesia lisan yang **asli**, API diuji dengan 5 video kuliah berbahasa Indonesia dari YouTube dengan topik berbeda. Video-video ini tidak ada di dataset dan tidak punya ringkasan referensi, jadi pengujiannya **kualitatif**: ROUGE tidak bisa dihitung.

## Cara mengambil transkrip

```bash
pip install youtube-transcript-api httpx
python uji_kuliah_asli/ambil_transkrip.py
```

- Script memakai library `youtube-transcript-api` untuk mengambil **subtitle bahasa Indonesia** tiap video. Teksnya sama dengan menu "Tampilkan transkrip" di YouTube.
- Kelima video hanya punya subtitle **otomatis** (pengenalan suara YouTube). Akibatnya teks sering tanpa tanda baca dan ada kata yang salah dengar.
- Pembersihan hanya membuang tag non-ucapan seperti `[Musik]` dan merapikan spasi. Isi ucapan tidak diubah.
- Judul dan kanal diambil dari oEmbed YouTube, lalu dicatat di `daftar_video.csv`. Videonya tidak diunduh.

## Cara menguji

Nyalakan server (`uvicorn api.main:app --host 127.0.0.1 --port 8000`), lalu:

```bash
python uji_kuliah_asli/uji_api.py
```

Hasilnya tersimpan di `hasil_ringkasan.json`. Kelima transkrip juga tersedia di koleksi Postman, request 6–11.

## Hasil (strategi chunk, CPU laptop)

| Topik | Video (kanal) | Kata input | Chunk | Kata ringkasan | Waktu |
|---|---|---|---|---|---|
| Fisika | Dinamika Partikel dan Hukum Newton (VIU ENERGY) | 1.503 | 2 | 204 | 66 s |
| Anatomi | Anatomi Sistem Saraf, Pendahuluan (Anatomi FK UNS) | 2.587 | 4 | 162 | 114 s |
| Fisiologi | Fisiologi Sistem Saraf Part 1 (UMEDS) | 1.251 | 2 | 108 | 42 s |
| Kimia | Kimia Dasar: Stoikiometri (Farmasi Unimuda) | 2.242 | 3 | 92 | 33 s |
| Genetika | Hereditas: Hukum Mendel 1 dan 2 (Dani Herdiana) | 3.406 | 4 | 147 | 56 s |

## Temuan

- **Topik utama selalu tertangkap.** Contohnya: hukum Newton, diagram benda bebas, serebrum dan lobus otak, neuron sensorik, definisi stoikiometri dan konsep mol, hukum Mendel dan persilangan.
- **Hasil terbaik: Stoikiometri.** Definisi stoikiometri, hukum kekekalan massa, dan satuan massa atom (dalton) diringkas dengan benar.
- **Kesalahan fakta masih sering muncul**, terutama pada materi yang padat istilah:
  - *anatomi*: "sistem saraf otonom terdiri dari nephron, sacralis dan coccygeus";
  - *fisiologi*: "sebelas pasang saraf tulang belakang";
  - *genetika*: hasil persilangan yang tidak masuk akal;
  - *fisika*: "tegangan tali … nol karena permukaan yang kurang padat".
- **Noise subtitle otomatis ikut terbawa.** Ringkasan genetika diawali kalimat salah dengar dari pembuka video ("Empat musim kemarau untuk tidur …").
- **Gaya terjemahan tetap muncul**, misalnya "Ingatlah bahwa" dan "Dalam kuliah ini, kita akan …", karena model dilatih dengan data hasil terjemahan.

Kesimpulannya, model bisa dipakai untuk menangkap **garis besar** kuliah berbahasa Indonesia asli. Namun ringkasannya belum bisa dipercaya untuk detail fakta, terutama istilah anatomi dan biologi.

## Catatan teknis: subtitle tanpa tanda baca

Pemecah chunk (`pecah_menjadi_chunk` di `api/peringkas.py`) memotong teks per kalimat berdasarkan titik. Subtitle otomatis umumnya tanpa titik, sehingga seluruh transkrip terbaca sebagai satu kalimat. Akibatnya strategi chunk diam-diam berubah menjadi truncation.

Masalah ini ditemukan saat pengujian ini, lalu diperbaiki: kalimat yang melebihi batas token sekarang dipotong per kata. Untuk teks bertanda baca, seperti data test, hasil chunking identik dengan versi sebelumnya.
