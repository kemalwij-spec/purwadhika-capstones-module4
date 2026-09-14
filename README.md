# 🦺 Construction Safety PPE Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-purple)

*Baca dalam [English](README.md)*

Capstone Project Modul 4 — AI Engineering, Purwadhika

## Ringkasan Project

Project ini mendeteksi kelengkapan Alat Pelindung Diri (APD) pekerja konstruksi dari sebuah foto. Diberikan satu gambar, aplikasi akan mendeteksi setiap pekerja, mengecek apakah masing-masing memakai **helm** dan **rompi keselamatan**, lalu melaporkan berapa pekerja yang sudah lengkap APD-nya, berapa yang terkonfirmasi APD tidak lengkap, dan berapa yang belum jelas statusnya sehingga perlu dicek manual.

Model deteksi yang dipakai adalah **YOLOv8n** yang di-fine-tune untuk 5 kelas: `person`, `helmet`, `no-helmet`, `vest`, `no-vest`. Tiga arsitektur (YOLOv8n, YOLOv8s, YOLOv12s) dilatih dan dibandingkan selama development — YOLOv8n akhirnya dipilih bukan karena mAP mentahnya paling tinggi, tapi karena tingkat **kesalahan false-safe**-nya paling rendah (yaitu kesalahan melabel pekerja yang sebenarnya APD tidak lengkap sebagai "aman"), yang merupakan jenis kesalahan paling mahal untuk alat safety-monitoring. Detail lengkap training, persiapan dataset, dan perbandingan model ada di notebook training.

## Fitur

- 📤 **Upload gambar** — menerima foto lokasi konstruksi berformat JPG/JPEG/PNG
- 🎯 **Status APD per pekerja** — mengklasifikasikan tiap pekerja terdeteksi sebagai lengkap, terkonfirmasi APD tidak lengkap, atau perlu dicek (bukan otomatis dianggap "aman" kalau tidak yakin)
- 🖼️ **Visualisasi berdampingan** — gambar asli di samping hasil anotasi berwarna (🟩 lengkap, 🟥 APD tidak lengkap, 🟧 belum jelas)
- 📊 **Ringkasan metrik** — total pekerja terdeteksi, jumlah APD lengkap, jumlah APD tidak lengkap, jumlah yang perlu dicek
- ⚙️ **Threshold yang bisa diatur** — atur confidence deteksi dan threshold pencocokan gear-ke-orang langsung dari sidebar
- 📥 **Download hasil anotasi** — export gambar hasil anotasi dalam format PNG
- 🔍 **Detail per pekerja** — breakdown status helmet/vest dan confidence level untuk tiap pekerja

## Alur Kerja (Flow)

```
Upload foto
     │
     ▼
YOLOv8n mendeteksi box mentah: person, helmet, no-helmet, vest, no-vest
     │
     ▼
Tiap box gear dicocokkan ke box person yang paling banyak overlap
     │
     ▼
Per orang: status kepala (helmet / no-helmet / unclear)
           status badan (vest / no-vest / unclear)
     │
     ▼
Ringkasan + gambar hasil anotasi berwarna + hasil bisa di-download
```

## Cara Menjalankan

### 1. Clone repository

```bash
git clone <your-repo-url>
cd Capstone-Module4
```

### 2. Buat virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # di Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan model weights

Pastikan `best.pt` (weights hasil training YOLOv8n) berada satu folder dengan `app.py`. File ini sudah termasuk di dalam repo.

### 5. Jalankan aplikasi secara lokal

```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

### 6. Cara pakai aplikasi

1. Upload foto lokasi konstruksi (JPG/PNG).
2. (Opsional) Atur threshold confidence dan overlap-matching di sidebar.
3. Lihat gambar asli dan hasil anotasi berdampingan.
4. Cek ringkasan metrik dan expand **"Detail per pekerja"** untuk hasil per individu.
5. Klik **"Download gambar hasil anotasi"** untuk menyimpan hasilnya.

## Struktur File

```
Capstone-Module4/
├── app.py               # UI Streamlit + pipeline inference
├── ppe_analysis.py       # Logic analisis deteksi -> status APD (di-import oleh app.py)
├── best.pt                # Weights hasil training YOLOv8n
├── requirements.txt       # Dependencies Python
├── .gitignore
├── README.md              
```

## Dependencies

Package utama yang dipakai langsung oleh aplikasi:

| Package | Dipakai untuk |
| --- | --- |
| `streamlit` | UI aplikasi web |
| `ultralytics` | Load model YOLOv8 & inference |
| `Pillow` | Load gambar, menggambar anotasi |
| `opencv-python-headless` | Pemrosesan gambar (dipakai saat training/persiapan dataset) |
| `numpy` | Operasi array/numerik |

> `requirements.txt` juga berisi beberapa package tambahan (`pandas`, `plotly`, `langchain`, `langchain-openai`, `reportlab`, `supervision`) yang saat ini tidak dipakai di `app.py` maupun `ppe_analysis.py`. Boleh dihapus sebelum deploy kalau mau install lebih ringan dan cepat di Streamlit Cloud — kecuali memang berencana mengembangkan fitur yang memakainya.

Install semuanya dengan:

```bash
pip install -r requirements.txt
```

## Deployment Streamlit

🔗 **Link aplikasi:** _[link Streamlit Community Cloud diisi di sini]_

<!-- Contoh setelah deploy:
🔗 **Link aplikasi:** https://nama-app-kamu.streamlit.app
-->

---

Dibuat sebagai bagian dari Capstone Project Modul 4 (AI Engineering — Purwadhika): Object Detection & Computer Vision.
