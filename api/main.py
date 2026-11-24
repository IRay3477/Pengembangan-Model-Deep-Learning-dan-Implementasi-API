"""API peringkasan transkrip video kuliah berbahasa Indonesia (Kelompok 6).

Jalankan dari folder utama repo:
    uvicorn api.main:app --host 127.0.0.1 --port 8000

Folder model diatur lewat variabel lingkungan MODEL_PATH (bawaan: folder model/ di repo).
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from api.peringkas import Peringkas

folder_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_model = os.environ.get('MODEL_PATH', os.path.join(folder_repo, 'model'))

minimal_kata = 30        # teks lebih pendek dari ini bukan transkrip kuliah, tidak perlu diringkas
maksimal_kata = 20000    # batas atas supaya satu request tidak memakan waktu terlalu lama

peringkas = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model dimuat sekali saat server dinyalakan, bukan di setiap request
    peringkas['model'] = Peringkas(folder_model)
    yield
    peringkas.clear()


app = FastAPI(
    title='API Peringkasan Transkrip Kuliah (Kelompok 6)',
    description='Ringkasan abstraktif transkrip video kuliah berbahasa Indonesia dengan IndoBART-v2 hasil fine-tuning.',
    version='1.0.0',
    lifespan=lifespan,
)


class PermintaanRingkasan(BaseModel):
    text: str = Field(description='Transkrip kuliah berbahasa Indonesia')
    strategi: Literal['chunk', 'truncation'] = Field(
        default='chunk',
        description="'chunk' membaca seluruh transkrip per potongan ±1.024 token; 'truncation' hanya membaca 1.024 token pertama",
    )

    @field_validator('text')
    @classmethod
    def cek_panjang_teks(cls, teks):
        jumlah_kata = len(teks.split())
        if jumlah_kata < minimal_kata:
            raise ValueError(f'teks terlalu pendek ({jumlah_kata} kata), minimal {minimal_kata} kata')
        if jumlah_kata > maksimal_kata:
            raise ValueError(f'teks terlalu panjang ({jumlah_kata} kata), maksimal {maksimal_kata} kata')
        return teks.strip()


class HasilRingkasan(BaseModel):
    summary: str
    strategi: str
    n_chunks: int
    jumlah_kata_input: int
    jumlah_kata_ringkasan: int
    waktu_detik: float


@app.get('/')
def cek_status():
    model = peringkas['model']
    return {
        'status': 'siap',
        'model': model.folder_model,
        'device': model.device,
        'pengaturan': model.pengaturan,
    }


@app.post('/summarize', response_model=HasilRingkasan)
def ringkas_transkrip(permintaan: PermintaanRingkasan):
    # fungsi biasa (bukan async): FastAPI menjalankannya di thread terpisah, jadi server tetap responsif saat model bekerja
    waktu_mulai = time.time()
    ringkasan, jumlah_chunk = peringkas['model'].ringkas(permintaan.text, permintaan.strategi)
    return HasilRingkasan(
        summary=ringkasan,
        strategi=permintaan.strategi,
        n_chunks=jumlah_chunk,
        jumlah_kata_input=len(permintaan.text.split()),
        jumlah_kata_ringkasan=len(ringkasan.split()),
        waktu_detik=round(time.time() - waktu_mulai, 2),
    )
