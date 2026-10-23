"""
Tahap 1: Profiling Dataset + Uji Terjemahan EN->ID (3 Kandidat Model)
Proyek Tugas 2 Deep Learning (Kelompok 6) - Universitas Udayana
Topik: Peringkasan Abstraktif Transkrip Video Pembelajaran Berbahasa Indonesia
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =====================================================================
# KONFIGURASI DEFAULT & KONSTANTA KOLOM
# =====================================================================
KOLOM_TRANSKRIP = 'Transcript'
KOLOM_RINGKASAN = 'Summary'
KOLOM_VIDEO = 'Youtube Video Link'

KANDIDAT_MODEL_TERJEMAHAN = {
    'opusmt': 'Helsinki-NLP/opus-mt-en-id',
    'm2m100': 'facebook/m2m100_418M',
    'nllb': 'facebook/nllb-200-distilled-600M',
}


# =====================================================================
# FUNGSI UTILITAS TERJEMAHAN
# =====================================================================
def pecah_kalimat(teks: str) -> list:
    """Memecah teks menjadi daftar kalimat berdasarkan tanda baca akhir kalimat."""
    return [kalimat for kalimat in re.split(r'(?<=[.!?])\s+', str(teks).strip()) if kalimat]


def muat_model_terjemahan(nama_model: str, device: str):
    """Memuat tokenizer dan model Seq2Seq untuk translasi EN -> ID."""
    tokenizer = AutoTokenizer.from_pretrained(nama_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(nama_model).to(device).eval()
    pengaturan_bahasa = {}
    if 'm2m100' in nama_model:
        tokenizer.src_lang = 'en'
        pengaturan_bahasa['forced_bos_token_id'] = tokenizer.get_lang_id('id')
    elif 'nllb' in nama_model:
        tokenizer.src_lang = 'eng_Latn'
        pengaturan_bahasa['forced_bos_token_id'] = tokenizer.convert_tokens_to_ids('ind_Latn')
    return tokenizer, model, pengaturan_bahasa


@torch.no_grad()
def terjemahkan(daftar_kalimat: list, tokenizer, model, pengaturan_bahasa: dict, device: str, ukuran_batch: int = 32) -> list:
    """Menerjemahkan daftar kalimat secara batch menggunakan beam search."""
    hasil = []
    for i in range(0, len(daftar_kalimat), ukuran_batch):
        batch = tokenizer(
            daftar_kalimat[i:i + ukuran_batch],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        # Pakai max_length saja (bukan max_new_tokens) supaya tidak muncul warning berulang
        keluaran = model.generate(**batch, num_beams=4, max_length=256, **pengaturan_bahasa)
        hasil += tokenizer.batch_decode(keluaran, skip_special_tokens=True)
    return hasil


# =====================================================================
# FUNGSI UTAMA TAHAP 1: PROFILING & UJI TERJEMAHAN
# =====================================================================
def jalankan_profiling(csv_path: str, output_tahap1: str) -> pd.DataFrame:
    """
    Mencakup Bagian 1 s.d. Bagian 4 pada Notebook:
    1. Load data
    2. Pilih kolom
    3. Cek kualitas data (NaN, hanya spasi, duplikat transkrip/ringkasan/pasangan)
    4. Statistik panjang (kata & token mT5), rasio kompresi, grafik distribusi, dan anomali panjang
    """
    os.makedirs(output_tahap1, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(csv_path)
    print('Ukuran data:', df.shape)
    print('=' * 30)
    print('Tipe data tiap kolom')
    print('=' * 30)
    print(df.dtypes)
    print('\n3 Baris pertama:')
    print(df.head(3))

    # 2. Pilih kolom & hapus index lama
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    print('\nVideo unik:', df[KOLOM_VIDEO].nunique(), 'dari', len(df), 'baris')
    print('\nJumlah video per course:')
    print(df['Course Name'].value_counts())

    # 3. Kualitas data
    # 3.1 Jumlah nilai kosong
    print('\nNaN per kolom:')
    print(df.isna().sum())

    print('\nDetail kolom teks:')
    for kolom in [KOLOM_TRANSKRIP, KOLOM_RINGKASAN]:
        jumlah_nan = df[kolom].isna().sum()
        jumlah_spasi = (df[kolom].notna() & (df[kolom].str.strip() == '')).sum()
        print(f'{kolom}: NaN = {jumlah_nan}, hanya spasi = {jumlah_spasi}')

    # 3.2 Contoh baris dengan transkrip kosong
    transkrip_kosong = df[df[KOLOM_TRANSKRIP].isna()]
    print('\nJumlah baris transkrip kosong:', len(transkrip_kosong))
    print(transkrip_kosong[['Course Name', 'Chapter', 'Video Name', KOLOM_RINGKASAN]].head(10))

    # 3.3 Contoh baris dengan ringkasan kosong
    ringkasan_kosong = df[df[KOLOM_RINGKASAN].isna()]
    print('\nJumlah baris ringkasan kosong:', len(ringkasan_kosong))
    print(ringkasan_kosong[['Course Name', 'Chapter', 'Video Name', KOLOM_TRANSKRIP]].head(10))

    # 3.4 Apakah ada baris yang kosong di dua kolom sekaligus?
    print('\nCrosstab transkrip kosong vs ringkasan kosong:')
    print(pd.crosstab(
        df[KOLOM_TRANSKRIP].isna(),
        df[KOLOM_RINGKASAN].isna(),
        rownames=['transkrip kosong'],
        colnames=['ringkasan kosong']
    ))

    # 3.5 Duplikat transkrip
    transkrip_terisi = df[df[KOLOM_TRANSKRIP].notna()]
    duplikat_transkrip = transkrip_terisi[transkrip_terisi[KOLOM_TRANSKRIP].str.strip().duplicated(keep=False)]
    print('\nBaris dengan transkrip duplikat:', len(duplikat_transkrip))

    # 3.6 Duplikat ringkasan
    ringkasan_terisi = df[df[KOLOM_RINGKASAN].notna()].copy()
    ringkasan_terisi['ringkasan_rapi'] = ringkasan_terisi[KOLOM_RINGKASAN].str.strip()
    duplikat_ringkasan = ringkasan_terisi[ringkasan_terisi['ringkasan_rapi'].duplicated(keep=False)].copy()
    duplikat_ringkasan['nomor_grup'] = duplikat_ringkasan.groupby('ringkasan_rapi').ngroup()
    duplikat_ringkasan['awal_ringkasan'] = duplikat_ringkasan[KOLOM_RINGKASAN].str[:80] + '...'
    print('\nBaris dengan ringkasan duplikat:', len(duplikat_ringkasan))
    print('Jumlah grup ringkasan yang sama:', duplikat_ringkasan['nomor_grup'].nunique())
    print(duplikat_ringkasan.sort_values('nomor_grup')[['nomor_grup', 'Course Name', 'Chapter', 'Video Name', 'awal_ringkasan']].head(20))

    # 3.7 Duplikat pasangan
    data_terisi = df.dropna(subset=[KOLOM_TRANSKRIP, KOLOM_RINGKASAN])
    jumlah_duplikat_pasangan = data_terisi.duplicated([KOLOM_TRANSKRIP, KOLOM_RINGKASAN]).sum()
    print('\nDuplikat pasangan:', jumlah_duplikat_pasangan)

    # 4. Statistik panjang (kata & token) + rasio kompresi
    print('\nMenghitung panjang kata dan token menggunakan google/mt5-small...')
    tokenizer_mt5 = AutoTokenizer.from_pretrained('google/mt5-small')

    data_panjang = df.dropna(subset=[KOLOM_TRANSKRIP, KOLOM_RINGKASAN]).copy()
    data_panjang['kata_transkrip'] = data_panjang[KOLOM_TRANSKRIP].str.split().str.len()
    data_panjang['kata_ringkasan'] = data_panjang[KOLOM_RINGKASAN].str.split().str.len()

    token_transkrip = tokenizer_mt5(data_panjang[KOLOM_TRANSKRIP].tolist(), add_special_tokens=False)['input_ids']
    token_ringkasan = tokenizer_mt5(data_panjang[KOLOM_RINGKASAN].tolist(), add_special_tokens=False)['input_ids']
    data_panjang['token_transkrip'] = [len(ids) for ids in token_transkrip]
    data_panjang['token_ringkasan'] = [len(ids) for ids in token_ringkasan]
    data_panjang['kompresi_kata'] = data_panjang['kata_ringkasan'] / data_panjang['kata_transkrip']

    kolom_statistik = ['kata_transkrip', 'token_transkrip', 'kata_ringkasan', 'token_ringkasan', 'kompresi_kata']
    statistik = data_panjang[kolom_statistik].describe(percentiles=[.25, .5, .75, .95]).T.round(2)
    print('\nStatistik Panjang:')
    print(statistik)
    print(f"\nTranskrip > 512 token: {(data_panjang['token_transkrip'] > 512).mean() * 100:.1f}%")

    stats_csv_path = os.path.join(output_tahap1, 'profiling_stats.csv')
    statistik.to_csv(stats_csv_path)
    print(f'Tersimpan: {stats_csv_path}')

    # 4.1 Grafik distribusi panjang
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data_panjang['token_transkrip'], bins=40)
    axes[0].axvline(512, color='red', linestyle='--', label='batas 512 token')
    axes[0].set_title('Panjang transkrip (token mT5)')
    axes[0].set_xlabel('jumlah token')
    axes[0].set_ylabel('jumlah video')
    axes[0].legend()

    axes[1].hist(data_panjang['token_ringkasan'], bins=40, color='tab:orange')
    axes[1].set_title('Panjang ringkasan (token mT5)')
    axes[1].set_xlabel('jumlah token')

    plt.tight_layout()
    plot_path = os.path.join(output_tahap1, 'distribusi_panjang.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Tersimpan grafik: {plot_path}')

    # 4.2 Anomali panjang
    ringkasan_terlalu_pendek = data_panjang[data_panjang['kata_ringkasan'] < 20]
    print('\nRingkasan < 20 kata:', len(ringkasan_terlalu_pendek))
    print(ringkasan_terlalu_pendek.sort_values('kata_ringkasan')[['Video Name', 'kata_transkrip', 'kata_ringkasan', KOLOM_RINGKASAN]].head(10))

    ringkasan_lebih_panjang = data_panjang[data_panjang['kompresi_kata'] > 1]
    print('\nRingkasan lebih panjang dari transkrip:', len(ringkasan_lebih_panjang))
    print(ringkasan_lebih_panjang.sort_values('kompresi_kata', ascending=False)[['Video Name', 'kata_transkrip', 'kata_ringkasan', KOLOM_TRANSKRIP]].head(10))

    return data_panjang


def jalankan_uji_terjemahan(data_panjang: pd.DataFrame, output_tahap1: str, jumlah_sampel: int = 60, jumlah_kalimat_transkrip: int = 15):
    """
    Mencakup Bagian 5 pada Notebook:
    5.1 Cek panjang kalimat dengan tokenizer Opus-MT
    5.2 & 5.3 Jalankan terjemahan pada 60 sampel yang sama untuk ketiga model kandidat
    """
    os.makedirs(output_tahap1, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 5.1 Cek panjang kalimat
    tokenizer_opus = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-id')
    semua_kalimat = [kalimat for teks in data_panjang[KOLOM_TRANSKRIP] for kalimat in pecah_kalimat(teks)]
    panjang_kalimat = pd.Series([len(ids) for ids in tokenizer_opus(semua_kalimat)['input_ids']])

    print('\nJumlah kalimat:', len(semua_kalimat))
    print(panjang_kalimat.describe(percentiles=[.5, .95, .99]).round(1))
    print(f'Kalimat > 256 token: {(panjang_kalimat > 256).sum()} ({(panjang_kalimat > 256).mean() * 100:.2f}%)')
    print('\nContoh kalimat terpanjang (300 karakter awal):')
    print(semua_kalimat[panjang_kalimat.idxmax()][:300])

    # 5.3 Jalankan pada sampel yang sama untuk ketiga model
    sampel = data_panjang.sample(jumlah_sampel, random_state=42)
    semua_hasil = []

    for nama_singkat, nama_model in KANDIDAT_MODEL_TERJEMAHAN.items():
        print(f'\nMemproses model terjemahan: {nama_singkat} ({nama_model}) pada {device}...')
        tokenizer, model, pengaturan_bahasa = muat_model_terjemahan(nama_model, device)
        for indeks, baris in sampel.iterrows():
            kalimat_transkrip = pecah_kalimat(baris[KOLOM_TRANSKRIP])[:jumlah_kalimat_transkrip]
            kalimat_ringkasan = pecah_kalimat(baris[KOLOM_RINGKASAN])
            semua_hasil.append({
                'idx': indeks,
                'model': nama_singkat,
                'transkrip_en': ' '.join(kalimat_transkrip),
                'transkrip_id': ' '.join(terjemahkan(kalimat_transkrip, tokenizer, model, pengaturan_bahasa, device)),
                'ringkasan_en': baris[KOLOM_RINGKASAN],
                'ringkasan_id': ' '.join(terjemahkan(kalimat_ringkasan, tokenizer, model, pengaturan_bahasa, device)),
            })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('Selesai:', nama_singkat)

    hasil_terjemahan = pd.DataFrame(semua_hasil).sort_values(['idx', 'model']).reset_index(drop=True)
    for kolom_isian in ['skor_makna', 'skor_istilah', 'skor_kealamian', 'catatan']:
        hasil_terjemahan[kolom_isian] = ''  # diisi manual, skala 1-3

    out_csv = os.path.join(output_tahap1, 'uji_terjemahan_3model.csv')
    hasil_terjemahan.to_csv(out_csv, index=False)
    print(f'\nTersimpan: {out_csv}')

    # Lihat satu contoh ringkasan dari ketiga model berdampingan
    contoh_idx = hasil_terjemahan['idx'].iloc[0]
    print('\n[EN]', hasil_terjemahan.loc[hasil_terjemahan['idx'] == contoh_idx, 'ringkasan_en'].iloc[0][:400], '\n')
    for _, baris in hasil_terjemahan[hasil_terjemahan['idx'] == contoh_idx].iterrows():
        print(f"[{baris['model']}] {baris['ringkasan_id'][:400]}\n")


def rekap_penilaian_manual(output_tahap1: str):
    """Mencakup Bagian 6 pada Notebook: Rekapitulasi skor penilaian manual 3 model translasi."""
    out_csv = os.path.join(output_tahap1, 'uji_terjemahan_3model.csv')
    if not os.path.exists(out_csv):
        print(f'\nFile {out_csv} belum tersedia untuk penilaian manual.')
        return

    nilai = pd.read_csv(out_csv)
    kolom_skor = ['skor_makna', 'skor_istilah', 'skor_kealamian']
    nilai[kolom_skor] = nilai[kolom_skor].apply(pd.to_numeric, errors='coerce')

    print('\nBaris yang sudah dinilai:', nilai[kolom_skor].notna().all(axis=1).sum(), 'dari', len(nilai))
    print(nilai.groupby('model')[kolom_skor].mean().round(2))


def main():
    parser = argparse.ArgumentParser(description='Tahap 1: Profiling Dataset + Uji Terjemahan EN->ID')
    parser.add_argument('--folder_utama', type=str, default='/content/drive/MyDrive/DEEP_LEARNING',
                        help='Path folder utama proyek')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path file CSV dataset awal (opsional)')
    parser.add_argument('--skip_translasi', action='store_true',
                        help='Lewati proses inferensi translasi 3 model jika file uji_terjemahan_3model.csv sudah ada')
    args = parser.parse_args()

    csv_path = args.csv_path or os.path.join(args.folder_utama, 'Dataset', 'ak_lectures_summarized.csv')
    output_tahap1 = os.path.join(args.folder_utama, 'tahap1')

    data_panjang = jalankan_profiling(csv_path, output_tahap1)
    if not args.skip_translasi:
        jalankan_uji_terjemahan(data_panjang, output_tahap1)
    rekap_penilaian_manual(output_tahap1)


if __name__ == '__main__':
    main()
