"""Mengambil subtitle bahasa Indonesia dari video kuliah YouTube untuk uji API.

Subtitle diambil dengan library youtube-transcript-api, yaitu teks yang sama
dengan menu "Tampilkan transkrip" di YouTube. Judul dan kanal diambil dari
endpoint oEmbed YouTube. Video tidak diunduh.

Jalankan dari folder repo:
    pip install youtube-transcript-api httpx
    python uji_kuliah_asli/ambil_transkrip.py
"""
import csv
import os
import re
import time

import httpx
from youtube_transcript_api import YouTubeTranscriptApi

daftar_id_video = [
    '-j5MHhrP-zc',   # fisika: dinamika partikel dan hukum Newton
    '4y_MCMl9PRI',   # anatomi: sistem saraf (FK UNS)
    '_8TdLyOXA0I',   # fisiologi: sistem saraf
    'Yhhp9yyz2tk',   # kimia: stoikiometri
    'X7bZcnzDEpc',   # genetika: hukum Mendel
]

folder = os.path.dirname(os.path.abspath(__file__))
folder_transkrip = os.path.join(folder, 'transkrip')
os.makedirs(folder_transkrip, exist_ok=True)


def bersihkan(teks):
    # tag seperti [Musik] atau [Tepuk tangan] bukan ucapan dosen
    teks = re.sub(r'\[[^\]]*\]', ' ', teks)
    return re.sub(r'\s+', ' ', teks).strip()


api = YouTubeTranscriptApi()
baris_csv = []
for id_video in daftar_id_video:
    url = f'https://www.youtube.com/watch?v={id_video}'
    info = httpx.get('https://www.youtube.com/oembed', params={'url': url, 'format': 'json'}).json()

    subtitle = api.fetch(id_video, languages=['id'])
    teks = bersihkan(' '.join(potongan.text for potongan in subtitle))

    with open(os.path.join(folder_transkrip, f'{id_video}.txt'), 'w', encoding='utf-8') as berkas:
        berkas.write(teks + '\n')

    baris_csv.append({'id_video': id_video, 'judul': info['title'], 'kanal': info['author_name'], 'url': url,
                      'subtitle_otomatis': subtitle.is_generated, 'jumlah_kata': len(teks.split())})
    print(id_video, len(teks.split()), 'kata |', info['title'])
    time.sleep(3)   # jeda supaya tidak diblokir YouTube

with open(os.path.join(folder, 'daftar_video.csv'), 'w', encoding='utf-8', newline='') as berkas:
    penulis = csv.DictWriter(berkas, fieldnames=list(baris_csv[0]))
    penulis.writeheader()
    penulis.writerows(baris_csv)
print('Tersimpan di', folder)
