# Dashboard Monitoring Sentimen Investor dengan Streamlit

Proyek ini menampilkan hasil klasifikasi sentimen komunitas investor pada Platform X
dalam tiga kelas: **Positif, Netral, dan Negatif**. Dashboard dapat:

- membaca data CSV;
- menjalankan inferensi menggunakan model mBERT hasil fine-tuning;
- menampilkan total dan distribusi sentimen;
- menampilkan tren sentimen berdasarkan waktu;
- memfilter data berdasarkan sentimen, ticker, tanggal, dan kata kunci;
- memprediksi satu teks secara langsung;
- mengunduh hasil yang telah difilter.

## 1. Struktur folder

```text
streamlit_sentiment_dashboard/
├── app.py
├── requirements.txt
├── sample_data.csv
└── model_mbert/
    ├── config.json
    ├── model.safetensors atau pytorch_model.bin
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.txt
```

Folder `model_mbert` tidak disertakan karena harus diisi dengan model hasil fine-tuning Anda.

## 2. Menyimpan model hasil fine-tuning

Jika pelatihan menggunakan Hugging Face Trainer:

```python
trainer.save_model("model_mbert")
tokenizer.save_pretrained("model_mbert")
```

Agar label model langsung terbaca dengan benar, sebaiknya konfigurasi model menggunakan:

```python
id2label = {
    0: "Negatif",
    1: "Netral",
    2: "Positif",
}

label2id = {
    "Negatif": 0,
    "Netral": 1,
    "Positif": 2,
}
```

Apabila urutan label model Anda berbeda, ubah pemetaan `LABEL_0`, `LABEL_1`,
dan `LABEL_2` melalui sidebar dashboard.

## 3. Menjalankan di Windows

Buka Command Prompt atau terminal VS Code pada folder proyek.

### Membuat virtual environment

```bash
python -m venv .venv
```

### Mengaktifkan virtual environment

Command Prompt:

```bash
.venv\Scripts\activate
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Memasang library

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Menjalankan dashboard

```bash
streamlit run app.py
```

Alternatif:

```bash
python -m streamlit run app.py
```

Browser biasanya terbuka otomatis pada:

```text
http://localhost:8501
```

## 4. Format CSV

Format yang disarankan:

```csv
created_at,text,sentiment,confidence,ticker
2026-06-01 09:00:00,"IHSG menguat hari ini",Positif,0.94,IHSG
```

Kolom minimum hanya `text`. Jika kolom `sentiment` belum tersedia, dashboard akan
menampilkan tombol **Jalankan klasifikasi mBERT**.

Nama kolom teks yang didukung antara lain:

- `text`
- `tweet`
- `full_text`
- `content`
- `post`
- `unggahan`
- `komentar`
- `teks`

## 5. Kesalahan yang sering muncul

### `ModuleNotFoundError`

Pastikan virtual environment aktif, kemudian jalankan:

```bash
pip install -r requirements.txt
```

### Model tidak ditemukan

Pastikan folder `model_mbert` berada sejajar dengan `app.py`, atau isi kolom
**Folder atau model ID** pada sidebar menggunakan lokasi model yang benar.

### Label hasil prediksi tertukar

Periksa urutan label saat pelatihan. Sesuaikan pemetaan `LABEL_0`, `LABEL_1`,
dan `LABEL_2` pada sidebar.

### Komputer lambat ketika klasifikasi

Kurangi nilai **Batch size inferensi** pada sidebar, misalnya menjadi 4 atau 8.
