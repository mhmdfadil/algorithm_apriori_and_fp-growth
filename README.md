# 🛍️ Analisis Market Basket Transaction Kosmetik dengan Algoritma Apriori dan FP-Growth

![Python](https://img.shields.io/badge/python-3670A0?style=plastic&logo=python&logoColor=ffdd54)  ![OpenPYXL](https://img.shields.io/badge/Openpyxl-%23FFFFFF.svg?style=plastic&logo=openpyxl&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=plastic&logo=pandas&logoColor=white) ![XlsxWriter](https://img.shields.io/badge/XlsxWriter-990000?style=plastic&logo=xlsxwriter&logoColor=white) ![NetworkX](https://img.shields.io/badge/NetworkX-4479A1?style=plastic&logo=networkx&logoColor=white) ![PyGraphviz](https://img.shields.io/badge/PyGraphviz-%2307405e.svg?style=plastic&logo=pygraphviz&logoColor=white)  ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=plastic&logo=Matplotlib&logoColor=black) ![Pydot](https://img.shields.io/badge/Pydot-%23F7931E.svg?style=plastic&logo=pydot&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?style=plastic&logo=github&logoColor=white)

Proyek ini mengimplementasikan **Analisis Market Basket (MBA)** pada data transaksi kosmetik menggunakan dua algoritma populer: **Apriori** dan **FP-Growth**. Tujuannya adalah menemukan aturan asosiasi antar produk untuk memahami perilaku pembelian pelanggan, mengoptimalkan penempatan produk, dan meningkatkan strategi pemasaran.

---

## 📌 Tentang Proyek

Analisis Market Basket adalah teknik untuk menemukan hubungan antara produk yang sering dibeli bersama. Proyek ini:
- Mengolah data transaksi kosmetik
- Mengimplementasikan dua algoritma populer (Apriori & FP-Growth)
- Membandingkan performa kedua algoritma
- Menghasilkan rekomendasi asosiasi produk

---

## 🎯 Tujuan

1. Menemukan pola pembelian produk kosmetik
2. Memahami perbedaan Apriori vs FP-Growth
3. Memberikan rekomendasi penjualan produk
4. Optimasi tata letak produk (product placement)

---

## ✨ Fitur Utama


| Fitur | Apriori | FP-Growth |
|-------|---------|-----------|
| Preprocessing Data | ✅ | ✅ |
| Frequent Itemset Mining | ✅ | ✅ |
| Association Rule Generation | ✅ | ✅ |
| Visualisasi FP-Tree | ❌ | ✅ |
| Perbandingan Algoritma | ✅ | ✅ |
| Metrik Evaluasi (Support, Confidence, Lift) | ✅ | ✅ |

✅ Pra-pemrosesan Data – Mengubah data transaksi mentah menjadi format terstruktur.
✅ Algoritma Apriori – Mencari itemset yang sering muncul dan aturan asosiasi.
✅ Algoritma FP-Growth – Menemukan pola secara efisien menggunakan struktur pohon.
✅ Visualisasi – Menyertakan visualisasi FP-Tree dan grafik perbandingan.
✅ Laporan Excel – Menghasilkan laporan detail dengan metrik dan analisis.

---

## ⚙️ Konfigurasi Parameter

Anda dapat mengubah parameter analisis di file utama:

### Untuk Mengubah Parameter Min. Support dan Min. Confidance Pada Algoritma Apriori dan FP-Growth:
```python
min_support = 0.2    # Ubah nilai support (0-1)
min_confidence = 0.5 # Ubah nilai confidence (0-1) 
```

---

### Visualisasi
- Diagram FP-Tree
- Grafik perbandingan algoritma
- Laporan Excel interaktif

---

## 💻 Instalasi

### Persyaratan
- Python 3.9+
- Graphviz 2.50+

### Panduan Instalasi Langkah-demi-Langkah

1. **Instal Paket Python**:
   ```bash
   pip install pandas matplotlib openpyxl xlsxwriter networkx pydot pygraphviz scikit-learn
   ```
2. ** Instalasi Graphviz (Diperlukan untuk Visualisasi FP-Tree)**:
   <br>
   **Windows**
   a. Unduh Graphviz dari https://graphviz.org/download/
   b. Instal (misalnya di C:\Program Files\Graphviz)
   c. Tambahkan Graphviz ke PATH Sistem:
   d. Buka Environment Variables (Win + R → sysdm.cpl → Advanced → Environment Variables)
   e. Tambahkan C:\Program Files\Graphviz\bin ke PATH
   <br>
   **Mac/Linux**
   ```bash
   # Untuk Mac (menggunakan Homebrew)
   brew install graphviz
   

   # Untuk Linux (Debian/Ubuntu)
   sudo apt-get install graphviz
   ```

3. Verifikasi Instalasi
   ```bash
   dot -V  # Harus menampilkan versi Graphviz
   ```

4. Instal PyGraphviz
   ```bash
   pip install pygraphviz
   ```

---

## 🔧 Cara Kerja

### 📂 Struktur File
   ```text
   📦 project/
   ├── 📄 transaksi_kosmetik.xlsx (Data input)
   ├── 📄 kp1_transaksi.py (Pra-pemrosesan data)
   ├── 📄 kp2_apriori.py (Algoritma Apriori)
   ├── 📄 kp3_fp-growth.py (Algoritma FP-Growth)
   ├── 📄 kp4_perbandingan.py (Perbandingan & visualisasi)
   └── 📂 data_temp/ (File output yang dihasilkan)
   ```

### 🔢 Urutan Eksekusi
**Jalankan skrip berurutan:**

   1. kp1_transaksi.py – Memproses data transaksi mentah.
      - [x] Input: transaksi_kosmetik.xlsx
      - [x] Output:
         - tahap1_transaksi_processed.xlsx
         - tahap2_transaksi_basket.json
         - tahap3_one_hot_transaksi.xlsx

   2. kp2_apriori.py – Menjalankan algoritma Apriori.
      - [x] Input: tahap3_one_hot_transaksi.xlsx
      - [x] Output: tahap4_hasil_perhitungan.xlsx

   3. kp3_fp-growth.py – Menjalankan algoritma FP-Growth.
      - [x] Input: tahap3_one_hot_transaksi.xlsx
      - [x] Output:
         - tahap5_hasil_perhitungan.xlsx
         - fp_tree.jpg (Visualisasi)

   4. kp4_perbandingan.py – Membandingkan Apriori vs. FP-Growth.
      - [x] Input:
         - tahap4_hasil_perhitungan.xlsx (Apriori)
         - tahap5_hasil_perhitungan.xlsx (FP-Growth)
      - [x] Output:
         - tahap6_hasil_perbandingan.xlsx
         - hasil_perbandingan.jpg (Visualisasi)

---

## 🚀 Menjalankan Proyek
   ```bash
   python kp1_transaksi.py
   python kp2_apriori.py
   python kp3_fp-growth.py
   python kp4_perbandingan.py
   ```

---

## 📜 Lisensi

**Lisensi MIT** ![License](https://img.shields.io/badge/License-MIT-green)

Hak Cipta (c) 2025 [MUHAMMAD FADILAH]
<div align="justify">
Dengan ini diberikan izin, secara gratis, kepada siapa pun yang memperoleh salinan perangkat lunak ini dan file dokumentasi terkait ("Perangkat Lunak"), untuk menggunakan Perangkat Lunak tanpa batasan, termasuk tanpa batasan hak untuk menggunakan, menyalin, memodifikasi, menggabungkan, memublikasikan, mendistribusikan, memberikan sublisensi, dan/atau menjual salinan Perangkat Lunak, dan mengizinkan orang yang menerima Perangkat Lunak untuk melakukannya, dengan syarat-syarat berikut:
<br>
Pemberitahuan hak cipta di atas dan pemberitahuan izin ini harus disertakan dalam semua salinan atau bagian substansial dari Perangkat Lunak.
<br>
PERANGKAT LUNAK INI DISEDIAKAN "SEBAGAIMANA ADANYA", TANPA JAMINAN APA PUN, BAIK TERSURAT MAUPUN TERSIRAT, TERMASUK TAPI TIDAK TERBATAS PADA JAMINAN KELAYAKAN JUAL, KESESUAIAN UNTUK TUJUAN TERTENTU DAN TIDAK MELANGGAR HAK ORANG LAIN. DALAM KEADAAN APA PUN, PENULIS ATAU PEMEGANG HAK CIPTA TIDAK BERTANGGUNG JAWAB ATAS KLAIM, KERUSAKAN, ATAU TANGGUNG JAWAB LAINNYA, BAIK DALAM PERKARA KONTRAK, KELALAIAN, ATAU LAINNYA, YANG TIMBUL DARI, DI LUAR, ATAU TERKAIT DENGAN PERANGKAT LUNAK ATAU PENGGUNAANNYA.
</div>
Dikembangkan dengan ❤️ oleh Muhammad Fadilah | [Portofolio Saya](https://muhammad-fadilah-portofolio.netlify.app/)