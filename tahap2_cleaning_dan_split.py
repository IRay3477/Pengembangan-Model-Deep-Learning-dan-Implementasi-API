"""
Tahap 2: Cleaning Dataset & Data Splitting (80/10/10 per Grup Topik)
Proyek Tugas 2 Deep Learning (Kelompok 6) - Universitas Udayana
Topik: Peringkasan Abstraktif Transkrip Video Pembelajaran Berbahasa Indonesia

Yang baru dibuat pada Tahap 2 ini (dibandingkan Tahap 1):
1. Normalisasi whitespace (menghapus spasi ganda, tab, newline, dan spasi di awal/akhir teks).
2. Pembuangan baris yang memiliki transkrip atau ringkasan kosong (NaN maupun string kosong).
3. Pembuangan baris anomali panjang: ringkasan terlalu pendek (< 20 kata) ATAU rasio kompresi > 1 (ringkasan lebih panjang dari transkrip).
4. Pembuangan seluruh baris yang memiliki ringkasan duplikat (`keep=False`) maupun transkrip duplikat.
5. Pembentukan atribut `grup_topik` dengan menghapus sufiks nomor seri video seperti '(Part I)', '(Part II)', 'Part 2', dsb.
6. Pengecekan daftar grup yang memiliki >1 video berseri dan pengecekan kata filler lisan ('um', 'uh', 'erm').
7. Pembagian dataset menjadi 80% Train, 10% Dev, dan 10% Test menggunakan `GroupShuffleSplit` berdasarkan `grup_topik` (dengan seed tetap = 42) agar tidak terjadi kebocoran data (zero data leakage) antar video berseri.
8. Verifikasi irisan antar split (train-dev, train-test, dev-test = 0) dan pemeriksaan proporsi Course Name di setiap split, lalu menyimpan hasilnya ke `data_clean_en.csv`.
"""

import os
import re
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# =====================================================================
# KONFIGURASI DEFAULT & KONSTANTA KOLOM
# =====================================================================
KOLOM_TRANSKRIP = 'Transcript'
KOLOM_RINGKASAN = 'Summary'
KOLOM_VIDEO = 'Youtube Video Link'


# =====================================================================
# FUNGSI BARU DI TAHAP 2 (CLEANING & GROUPING TOPIK)
# =====================================================================
def rapikan_spasi(teks: str) -> str:
    """1) Normalisasi spasi: hapus spasi ganda, tab, newline, spasi di awal/akhir."""
    return re.sub(r'\s+', ' ', str(teks)).strip()


def buat_grup_topik(baris: pd.Series) -> str:
    """
    6) Grup topik: judul tanpa "(Part I)", "(Part II)", "Part 2", dst.
       Part I dan Part II dari topik yang sama harus masuk split yang sama.
    """
    nama_video = re.sub(r'[\s\-–:,]*\(?\bpart\s+([ivx]+|\d+)\b\)?\s*$', '', str(baris['Video Name']), flags=re.I)
    return f"{baris['Course Name']} | {baris['Chapter']} | {nama_video.strip().lower()}"


# =====================================================================
# FUNGSI UTAMA TAHAP 2: CLEANING & DATA SPLITTING
# =====================================================================
def jalankan_cleaning_dan_split(csv_path: str, output_tahap2: str, seed: int = 42) -> pd.DataFrame:
    """Menjalankan pipeline pembersihan data dan pembagian split 80/10/10 bebas kebocoran."""
    os.makedirs(output_tahap2, exist_ok=True)

    data_bersih = pd.read_csv(csv_path)
    if 'Unnamed: 0' in data_bersih.columns:
        data_bersih = data_bersih.drop(columns=['Unnamed: 0'])

    riwayat_cleaning = [('Data awal', len(data_bersih))]

    # 1) Normalisasi spasi: hapus spasi ganda, tab, newline, spasi di awal/akhir
    for kolom in [KOLOM_TRANSKRIP, KOLOM_RINGKASAN]:
        data_bersih[kolom] = data_bersih[kolom].where(
            data_bersih[kolom].isna(),
            data_bersih[kolom].map(rapikan_spasi)
        )

    # 2) Buang transkrip / ringkasan kosong
    data_bersih = data_bersih[
        data_bersih[KOLOM_TRANSKRIP].notna() &
        data_bersih[KOLOM_RINGKASAN].notna() &
        (data_bersih[KOLOM_TRANSKRIP] != '') &
        (data_bersih[KOLOM_RINGKASAN] != '')
    ]
    riwayat_cleaning.append(('Buang kosong', len(data_bersih)))

    # 3) Buang anomali: ringkasan < 20 kata ATAU ringkasan lebih panjang dari transkrip
    data_bersih['kata_transkrip'] = data_bersih[KOLOM_TRANSKRIP].str.split().str.len()
    data_bersih['kata_ringkasan'] = data_bersih[KOLOM_RINGKASAN].str.split().str.len()
    data_bersih['kompresi'] = data_bersih['kata_ringkasan'] / data_bersih['kata_transkrip']
    data_bersih = data_bersih[(data_bersih['kata_ringkasan'] >= 20) & (data_bersih['kompresi'] <= 1)]
    riwayat_cleaning.append(('Buang anomali (ringkasan <20 kata / kompresi >1)', len(data_bersih)))

    # 4) Buang SEMUA baris yang ringkasannya duplikat (tidak bisa dipastikan mana yang benar)
    data_bersih = data_bersih[~data_bersih[KOLOM_RINGKASAN].duplicated(keep=False)]
    riwayat_cleaning.append(('Buang ringkasan duplikat', len(data_bersih)))

    # 5) Buang transkrip duplikat (jaga-jaga, harusnya 0)
    data_bersih = data_bersih[~data_bersih[KOLOM_TRANSKRIP].duplicated(keep=False)]
    riwayat_cleaning.append(('Buang transkrip duplikat', len(data_bersih)))

    # 6) Grup topik: judul tanpa "(Part I)", "(Part II)", "Part 2", dst.
    data_bersih['grup_topik'] = data_bersih.apply(buat_grup_topik, axis=1)
    data_bersih = data_bersih.reset_index(drop=True)

    print(pd.DataFrame(riwayat_cleaning, columns=['Langkah', 'Sisa baris']).to_string(index=False))
    print('\nJumlah grup topik:', data_bersih['grup_topik'].nunique())

    # Cek: grup yang berisi lebih dari 1 video (Part I/II yang tergabung)
    grup_multi_video = data_bersih.groupby('grup_topik')['Video Name'].apply(list)
    grup_multi_video = grup_multi_video[grup_multi_video.str.len() > 1]
    print('\nGrup berisi >1 video:', len(grup_multi_video))
    print(grup_multi_video.head(10))

    # Cek filler lisan (cuma dihitung, BELUM dibuang)
    jumlah_filler = data_bersih[KOLOM_TRANSKRIP].str.count(r'\b(um+|uh+|erm)\b', flags=re.I)
    print('\nTranskrip yang mengandung filler:', (jumlah_filler > 0).sum(),
          '| rata-rata per transkrip:', round(jumlah_filler.mean(), 2))
    print(data_bersih[['kata_transkrip', 'kata_ringkasan', 'kompresi']].describe().round(2))

    # =================================================================
    # SPLIT 80/10/10 PER GRUP TOPIK
    # =================================================================
    # Tahap 1 Split: 80% train, 20% sisa
    pemisah_pertama = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    indeks_train, indeks_sisa = next(pemisah_pertama.split(data_bersih, groups=data_bersih['grup_topik']))
    data_sisa = data_bersih.iloc[indeks_sisa]

    # Tahap 2 Split: 20% sisa dibagi dua -> 10% dev, 10% test
    pemisah_kedua = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    indeks_dev, indeks_test = next(pemisah_kedua.split(data_sisa, groups=data_sisa['grup_topik']))

    data_bersih['split'] = 'train'
    data_bersih.loc[data_sisa.index[indeks_dev], 'split'] = 'dev'
    data_bersih.loc[data_sisa.index[indeks_test], 'split'] = 'test'

    print('\nJumlah baris per split:')
    print(data_bersih['split'].value_counts())
    print('\nProporsi baris per split (%):')
    print((data_bersih['split'].value_counts(normalize=True) * 100).round(1))

    # Bukti tidak ada kebocoran: tidak ada grup topik yang muncul di dua split
    grup_per_split = {
        nama: set(data_bersih.loc[data_bersih['split'] == nama, 'grup_topik'])
        for nama in ['train', 'dev', 'test']
    }
    print('\nIrisan train-dev :', len(grup_per_split['train'] & grup_per_split['dev']))
    print('Irisan train-test:', len(grup_per_split['train'] & grup_per_split['test']))
    print('Irisan dev-test  :', len(grup_per_split['dev'] & grup_per_split['test']))

    # Sebaran course per split (persen), supaya tiap split tidak didominasi satu course
    print('\nSebaran Course Name per split (%):')
    print((pd.crosstab(data_bersih['Course Name'], data_bersih['split'], normalize='columns') * 100).round(1))

    output_file = os.path.join(output_tahap2, 'data_clean_en.csv')
    data_bersih.to_csv(output_file, index=False)
    print('\nTersimpan:', output_file)

    return data_bersih


def main():
    parser = argparse.ArgumentParser(description='Tahap 2: Cleaning & Split 80/10/10 per Grup Topik')
    parser.add_argument('--folder_utama', type=str, default='/content/drive/MyDrive/DEEP_LEARNING',
                        help='Path folder utama proyek')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path file CSV dataset awal (opsional)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed untuk GroupShuffleSplit (default: 42)')
    args = parser.parse_args()

    csv_path = args.csv_path or os.path.join(args.folder_utama, 'Dataset', 'ak_lectures_summarized.csv')
    output_tahap2 = os.path.join(args.folder_utama, 'tahap2')

    jalankan_cleaning_dan_split(csv_path, output_tahap2, seed=args.seed)


if __name__ == '__main__':
    main()
