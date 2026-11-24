"""Inferensi model final (IndoBART-v2 hasil fine-tuning Tahap 6).

Isi file ini disalin dari notebook supaya hasil API sama dengan evaluasi:
TokenizerIndoBART (cell Tahap 5), pecah_kalimat, pecah_menjadi_chunk, dan
ringkas_batch / chunk-and-summarize (cell Tahap 4), serta pengaturan generate
dari tahap6/model_final.json.
"""
import json
import os
import re

import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, BatchEncoding

nama_model_dasar = 'indobenchmark/indobart-v2'

# nilai bawaan sama dengan tahap6/model_final.json; dipakai bila file itu tidak ikut diunduh
pengaturan_bawaan = {
    'batas_input': 1024,
    'panjang_maks_generate': 384,
    'pengaturan_generate': {'length_penalty': 2.0},
}


class TokenizerIndoBART:
    # meniru IndoNLGTokenizer: teks dikecilkan, lalu format <s> teks </s> [indonesian]
    def __init__(self, folder_model=None):
        path_lokal = os.path.join(folder_model, 'sentencepiece.bpe.model') if folder_model else ''
        path_sp = path_lokal if os.path.exists(path_lokal) else hf_hub_download(nama_model_dasar, 'sentencepiece.bpe.model')
        self.sp_model = spm.SentencePieceProcessor(model_file=path_sp)
        self.bos_token_id, self.pad_token_id, self.eos_token_id = 0, 1, 2
        self.id_bahasa = 40002   # token [indonesian]

    def __call__(self, teks, max_length=None, truncation=False, padding=False,
                 return_tensors=None, add_special_tokens=True):
        daftar_teks = [teks] if isinstance(teks, str) else list(teks)

        daftar_ids = []
        for satu_teks in daftar_teks:
            ids = self.sp_model.encode(str(satu_teks).lower())
            if add_special_tokens:
                if truncation and max_length:
                    ids = ids[:max_length - 3]
                ids = [self.bos_token_id] + ids + [self.eos_token_id, self.id_bahasa]
            daftar_ids.append(ids)

        daftar_mask = [[1] * len(ids) for ids in daftar_ids]
        if padding:
            panjang_maks = max(len(ids) for ids in daftar_ids)
            daftar_mask = [mask + [0] * (panjang_maks - len(mask)) for mask in daftar_mask]
            daftar_ids = [ids + [self.pad_token_id] * (panjang_maks - len(ids)) for ids in daftar_ids]

        if return_tensors == 'pt':
            return BatchEncoding({'input_ids': torch.tensor(daftar_ids), 'attention_mask': torch.tensor(daftar_mask)})
        if isinstance(teks, str):
            return {'input_ids': daftar_ids[0], 'attention_mask': daftar_mask[0]}
        return {'input_ids': daftar_ids, 'attention_mask': daftar_mask}

    def batch_decode(self, daftar_ids, skip_special_tokens=True):
        # id 0-3 (<s>, <pad>, </s>, <unk>) dan id >= 40000 (token bahasa, <mask>) dibuang
        return [self.sp_model.decode([int(i) for i in ids if 3 < int(i) < 40000]) for ids in daftar_ids]


def pecah_kalimat(teks):
    return [kalimat for kalimat in re.split(r'(?<=[.!?])\s+', str(teks).strip()) if kalimat]


def pecah_menjadi_chunk(teks, tokenizer, batas_token):
    # kalimat dikumpulkan sampai hampir batas token, jadi potongan tidak memotong di tengah kalimat
    daftar_chunk, kalimat_chunk, jumlah_token_chunk = [], [], 0
    for kalimat in pecah_kalimat(teks):
        jumlah_token = len(tokenizer(kalimat, add_special_tokens=False)['input_ids'])
        if kalimat_chunk and jumlah_token_chunk + jumlah_token > batas_token:
            daftar_chunk.append(' '.join(kalimat_chunk))
            kalimat_chunk, jumlah_token_chunk = [], 0
        kalimat_chunk.append(kalimat)
        jumlah_token_chunk += jumlah_token
    if kalimat_chunk:
        daftar_chunk.append(' '.join(kalimat_chunk))
    return daftar_chunk


def rapikan_kapital(teks):
    # IndoBART selalu menghasilkan huruf kecil; huruf pertama tiap kalimat dibesarkan supaya enak dibaca
    return ' '.join(kalimat[:1].upper() + kalimat[1:] for kalimat in pecah_kalimat(teks))


class Peringkas:
    """Memuat model sekali, lalu dipakai berulang oleh endpoint API."""

    def __init__(self, folder_model):
        self.folder_model = folder_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        path_pengaturan = os.path.join(folder_model, 'model_final.json')
        self.pengaturan = dict(pengaturan_bawaan)
        if os.path.exists(path_pengaturan):
            with open(path_pengaturan) as berkas:
                model_final = json.load(berkas)
            self.pengaturan.update({kunci: model_final[kunci] for kunci in pengaturan_bawaan if kunci in model_final})

        self.tokenizer = TokenizerIndoBART(folder_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(folder_model).to(self.device).eval()
        # token awal ringkasan saat generate harus sama dengan format label saat training
        self.model.generation_config.decoder_start_token_id = self.tokenizer.id_bahasa
        self.model.generation_config.forced_bos_token_id = self.tokenizer.bos_token_id

    @torch.no_grad()
    def ringkas_batch(self, daftar_teks, ukuran_batch=4):
        hasil = []
        for awal in range(0, len(daftar_teks), ukuran_batch):
            batch = self.tokenizer(daftar_teks[awal:awal + ukuran_batch], return_tensors='pt', padding=True,
                                   truncation=True, max_length=self.pengaturan['batas_input']).to(self.device)
            keluaran = self.model.generate(**batch, num_beams=4, max_new_tokens=self.pengaturan['panjang_maks_generate'],
                                           no_repeat_ngram_size=3, early_stopping=True,
                                           **self.pengaturan['pengaturan_generate'])
            hasil += [ringkasan.strip() for ringkasan in self.tokenizer.batch_decode(keluaran)]
        return hasil

    def ringkas(self, teks, strategi='chunk'):
        """Mengembalikan (ringkasan, jumlah_chunk). strategi: 'chunk' atau 'truncation'."""
        if strategi == 'truncation':
            return rapikan_kapital(self.ringkas_batch([teks])[0]), 1

        # chunk-and-summarize: tiap chunk diringkas, gabungannya diringkas sekali lagi bila lebih dari 1 chunk
        daftar_chunk = pecah_menjadi_chunk(teks, self.tokenizer, self.pengaturan['batas_input'] - 2)
        ringkasan_chunk = self.ringkas_batch(daftar_chunk)
        ringkasan = ringkasan_chunk[0] if len(ringkasan_chunk) == 1 else self.ringkas_batch([' '.join(ringkasan_chunk)])[0]
        return rapikan_kapital(ringkasan), len(daftar_chunk)
