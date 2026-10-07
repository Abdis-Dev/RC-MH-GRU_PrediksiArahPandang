# RC-MH-GRU untuk prediksi arah pandang 360°

Repositori ini memuat kode, konfigurasi, data turunan numerik, *checkpoint*, dan keluaran evaluasi yang mendukung artikel **“RC-MH-GRU untuk Prediksi Arah Pandang 360° pada Telepresensi Robot”**. Judul tersebut dipertahankan sesuai naskah yang ditelaah. Bukti eksperimen dibatasi pada evaluasi *offline* atas jejak arah kepala ketika pengguna menonton video 360° secara pasif; repositori ini tidak mengklaim pengujian robot, jaringan, *streaming*, atau telepresensi *end-to-end*.

## Hasil utama

| Horizon | RC-MH-GRU | Tiny Transformer | Selisih RC − Transformer |
|---:|---:|---:|---:|
| 100 ms | 0,958° | 1,026° | −0,068° |
| 200 ms | 2,033° | 2,090° | −0,057° |
| 300 ms | 3,324° | 3,383° | −0,059° |
| 500 ms | 6,041° | 6,118° | −0,077° |

RC-MH-GRU memiliki 15.336 parameter, sedangkan Tiny Transformer memiliki 14.412 parameter. Pada 11 video uji, uji Wilcoxon eksak setelah ambang Bonferroni 0,003125 mendukung perbedaan pada 100–300 ms, tetapi tidak pada 500 ms. Nilai lengkap, termasuk statistik tingkat video dan *benchmark* waktu inferensi, tersedia di folder `results/`.

## Struktur repositori

- `scripts/`: praproses, pembanding, pelatihan MH-GRU, fusi RC-MH-GRU, evaluasi, dan Tiny Transformer.
- `data/`: *window* numerik turunan yang dipakai pada eksperimen serta metadata pembagian video.
- `models/`: *checkpoint* lima MH-GRU, lima Tiny Transformer, model validasi silang, dan empat LSTM pembanding.
- `results/`: CSV/JSON/NPZ yang mendasari tabel, uji statistik, dan angka naskah.
- `figures/`: grafik hasil yang digunakan pada revisi.
- `tests/verify_reported_results.py`: pemeriksaan otomatis bahwa angka ringkasan cocok dengan naskah.

## Lingkungan

Eksperimen terakhir dijalankan dengan Python 3.12, NumPy 2.5.2, SciPy 1.18.0, pandas 2.2.3, PyTorch 2.13.0, dan Matplotlib 3.11.1 pada CPU Apple M3 dengan satu utas PyTorch. Versi minimal yang wajar dicantumkan dalam `requirements.txt`; perbedaan perangkat keras dapat mengubah waktu inferensi, tetapi tidak seharusnya mengubah definisi metrik.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tests/verify_reported_results.py
```

## Menjalankan ulang

Pemeriksaan angka publikasi dapat dijalankan langsung karena keluaran eksperimen disertakan:

```bash
python tests/verify_reported_results.py
```

Untuk melatih ulang Tiny Transformer menggunakan pembagian data yang sama:

```bash
python scripts/18_train_tiny_transformer_baseline.py --max-epochs 20
```

Untuk melatih keluarga model MH-GRU lima *seed*:

```bash
python scripts/09_train_mh_gru_family.py --only mh_gru
```

Untuk menghitung ulang fusi RC-MH-GRU pada *main split*, jalankan fungsi `main_split` dari `scripts/05_ridge_calibrated_mh_gru.py`. Menjalankan keseluruhan skrip tersebut juga mengulang *five-fold cross-validation* dan memerlukan salinan data mentah pada `dataset/AggregatedDataset`.

## Integritas evaluasi

- Semua pemisahan dilakukan menurut ID video, bukan secara acak pada tingkat *window*.
- Ridge-AR dilatih hanya pada data latih; regularisasi dan bobot fusi dipilih pada data validasi.
- *Test split* digunakan setelah pemilihan model selesai.
- Statistik inferensial dihitung pada tingkat video untuk menghindari anggapan bahwa *window* yang tumpang tindih bersifat independen.
- Tiny Transformer memakai masukan 10×6, empat keluaran horizon, objektif kosinus, *batch size* 1.024, *early stopping*, dan *seed* 42–46 yang sebanding dengan MH-GRU.

## Data, lisensi, dan pendanaan

Sumber dan batasan data dijelaskan dalam `DATA.md`. Data mentah tidak disertakan. Folder `data/` hanya berisi representasi numerik turunan yang diperlukan untuk memeriksa eksperimen. Sebelum repositori dipublikasikan, penulis perlu memastikan bahwa distribusi data turunan sesuai ketentuan sumber asli.

Lisensi kode dan pernyataan pendanaan belum boleh diasumsikan. Penulis harus melengkapi `LICENSE_SELECTION_REQUIRED.md` dan `FUNDING_STATEMENT_REQUIRED.md` sebelum membuat repositori publik.
