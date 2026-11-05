# Brief: Training Pipeline dan Alur Kode

Ringkasan cara benchmark CCTV-SR dijalankan: dari data mentah sampai tabel dan gambar untuk slide. Tujuannya membandingkan **YOLO26n, YOLO26s, RT-DETR-L** pada deteksi orang, gender, dan postur, dengan evaluasi yang tidak bocor.

## 1. Alur satu halaman

```
 data/indoor, data/outdoor            (ekspor Roboflow, split bawaan acak = bocor)
          │
          ▼
 [prep_sr.py]  gabung semua frame → urutkan → 5 blok waktu (+ buffer) → fold per task
          │      data/sr/<dataset>/<task>/fold<k>/{train,test} + data.yaml
          ▼
 [run_sr.sh]   loop: fold × task × model (× seed)  ── tiap job memanggil ──┐
                                                                            ▼
                                       [sr_eval.py]  latih (COCO-pretrained) → evaluasi bobot epoch terakhir
                                                      → results/cctv_sr/sr/<job>.json, checkpoint dihapus
          │
          ▼
 [aggregate_sr.py]  rata-rata ± CI 95% antar fold + selisih berpasangan → sr_summary.csv, sr_paired.csv
          │
          ▼
 [make_figures_sr.py]  gambar slide → results/cctv_sr/fig_sr/        [latency.py]  (terpisah) → results/cctv_sr/latency.json
```

`common.py` dipakai semua skrip (path, parsing label, IoU). Skrip `profile_dataset.py`, `profile_leakage.py`, `check_label_conflicts.py` dijalankan di awal untuk mengenali masalah data, tidak menjadi bagian loop training.

## 2. Alasan di balik desain (inti cerita)

| Masalah | Solusi di kode |
|---|---|
| Split bawaan acak atas frame berurutan dari satu kamera, sehingga frame uji punya "kembaran" di train | `prep_sr.py`: 5 blok waktu yang berurutan, frame di sekitar blok uji dibuang dari train (buffer 5 frame indoor, 10 outdoor) |
| Hasil bisa dibuat tampak bagus dengan memilih checkpoint/threshold di data uji | `sr_eval.py`: tanpa validation set (`val=False`), evaluasi bobot **epoch terakhir**, hanya AP (bebas threshold), threshold akurasi kelas tetap 0.25 |
| Satu dataset, beberapa pertanyaan | `task_mappings()`: kotak yang sama dipetakan ulang menjadi `person`, `gender`, `posture`, `allclass` |
| Run panjang dan bisa dihentikan | Satu job = satu file JSON, job yang sudah ada dilewati, sehingga `run_sr.sh` bisa dilanjutkan |
| Perbandingan tidak adil bila fold selesai berbeda-beda | `aggregate_sr.py`: hanya fold yang diselesaikan **ketiga** model, selisih dihitung berpasangan per fold |

## 3. Pipeline training

**Matriks percobaan**

| | Tugas | Fold | Model | Epoch | Seed (rencana) |
|---|---|---|---|---|---|
| Indoor (125 frame) | person, gender, allclass | 5 | 3 | 100 | 0, 1, 2 |
| Outdoor (934 frame) | person, gender, posture, allclass | 5 | 3 | 30 | 0 |

Epoch outdoor lebih sedikit karena datanya sekitar 8 kali lebih banyak per fold (jumlah langkah optimasi tetap lebih besar dari indoor).

**Pengaturan latihan satu job** (`sr_eval.py`)

| Parameter | Nilai |
|---|---|
| Bobot awal | COCO-pretrained (`yolo26n.pt`, `yolo26s.pt`, `rtdetr-l.pt`) |
| Ukuran gambar | 512 |
| Batch | 8 (RT-DETR-L: 4, karena GPU 6 GB) |
| Presisi | AMP |
| Seed | `--seed`, mode `deterministic=True` |
| Validasi saat latihan | tidak ada (`val=False`) |
| Early stopping | mati (`patience = epochs`) |
| Optimizer, learning rate, augmentasi | default Ultralytics, tidak diubah |

**Evaluasi satu job** (blok uji dari fold itu)
1. `model.val(split="test", conf=0.001)` pada `last.pt` → mAP50, mAP50-95, AP per kelas.
2. Untuk tugas gender/posture: `matched_class_accuracy()`. Prediksi (conf ≥ 0.25) dicocokkan tanpa memperhatikan kelas ke kotak asli dengan IoU ≥ 0.5, lalu dicek kelasnya benar atau tidak.
3. Hasil ditulis ke `results/cctv_sr/sr/<dataset>_<task>_<model>_f<fold>_s<seed>.json`, lalu checkpoint dihapus (hemat disk).

## 4. Peta kode

| File | Fungsi utama | Masuk | Keluar |
|---|---|---|---|
| `common.py` | `parse_label_line`, `read_label_file`, `iou`, `class_names` | – | – |
| `prep_sr.py` | `task_mappings` → `load_frames` → `write_fold` → `main` | `data/indoor`, `data/outdoor` | `data/sr/...`, `manifest.json` |
| `run_sr.sh` | fungsi `run`: loop fold × task × model | – | memanggil `sr_eval.py` |
| `sr_eval.py` | `main` (latih + evaluasi), `matched_class_accuracy`, `count_test_instances` | `data/sr/.../data.yaml` | `results/cctv_sr/sr/<job>.json` |
| `aggregate_sr.py` | `mean_ci`, `main` | `results/cctv_sr/sr/*.json` | `sr_summary.csv`, `sr_paired.csv` |
| `make_figures_sr.py` | satu fungsi `plot_*` per gambar | CSV + JSON hasil | `results/cctv_sr/fig_sr/*.png` |
| `latency.py` | `benchmark_setting`, `time_it` | model pretrained | `results/cctv_sr/latency.json` |
| `profile_dataset.py` | `main` | `data/<dataset>` | ringkasan di terminal |
| `profile_leakage.py` | `main` | `data/<dataset>` | ringkasan di terminal |
| `check_label_conflicts.py` | `main` | `data/<dataset>` | ringkasan di terminal |

## 5. Cara menjalankan

```bash
python benchmarks/cctv_sr/profile_dataset.py data/outdoor     # opsional: kenali data
python benchmarks/cctv_sr/profile_leakage.py data/outdoor     # opsional: bukti kebocoran split
python benchmarks/cctv_sr/check_label_conflicts.py            # opsional: label bertentangan
python benchmarks/cctv_sr/prep_sr.py                          # bangun fold
benchmarks/cctv_sr/run_sr.sh                                  # semua job (lama, bisa dihentikan)
python benchmarks/cctv_sr/aggregate_sr.py                     # tabel + CI + perbandingan berpasangan
python benchmarks/cctv_sr/make_figures_sr.py                  # gambar slide
python benchmarks/cctv_sr/latency.py                          # opsional, GPU harus bebas
```

## 6. Status dan batasan

- Benchmark dihentikan lebih awal: **96 dari rencana job selesai** (45 indoor, 51 outdoor), semuanya seed 0. Outdoor gender/posture/allclass baru sampai fold 0-3.
- Satu kamera per dataset, jadi hasil mengukur generalisasi antar waktu di ruangan yang sama, bukan ke ruangan lain.
- Label outdoor mengandung banyak kotak orang yang hampir sama persis dengan label gender bertentangan (belum dibersihkan).
- Setelah refactor, hanya `prep_sr.py` yang dijalankan ulang untuk verifikasi. Skrip lain baru dicek sintaks dan lint-nya.
