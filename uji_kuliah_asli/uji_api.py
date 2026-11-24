"""Mengirim 5 transkrip kuliah asli ke POST /summarize dan menyimpan hasilnya.

Nyalakan server dulu (uvicorn api.main:app --host 127.0.0.1 --port 8000),
lalu jalankan dari folder repo:
    python uji_kuliah_asli/uji_api.py
"""
import csv
import json
import os

import httpx

base_url = 'http://127.0.0.1:8000'
folder = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(folder, 'daftar_video.csv'), encoding='utf-8') as berkas:
    daftar_video = list(csv.DictReader(berkas))

hasil = []
for video in daftar_video:
    with open(os.path.join(folder, 'transkrip', f"{video['id_video']}.txt"), encoding='utf-8') as berkas:
        teks = berkas.read().strip()
    respons = httpx.post(f'{base_url}/summarize', json={'text': teks, 'strategi': 'chunk'}, timeout=600)
    respons.raise_for_status()
    hasil.append({'judul': video['judul'], 'url': video['url'], **respons.json()})
    print(video['judul'], '|', respons.json()['n_chunks'], 'chunk |', respons.json()['waktu_detik'], 'detik')

with open(os.path.join(folder, 'hasil_ringkasan.json'), 'w', encoding='utf-8') as berkas:
    json.dump(hasil, berkas, ensure_ascii=False, indent=2)
print('Tersimpan di', os.path.join(folder, 'hasil_ringkasan.json'))
