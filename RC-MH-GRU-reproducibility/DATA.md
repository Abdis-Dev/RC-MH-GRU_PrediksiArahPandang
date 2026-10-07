# Ketersediaan dan asal data

Dataset pelacakan kepala berasal dari **A Large Dataset of 360 Video User Behaviour**:

https://github.com/360VidStr/A-large-dataset-of-360-video-user-behaviour

Dokumentasi sumber menyebut 88 ID video. Pada salinan yang digunakan dalam penelitian, berkas jejak arah kepala untuk ID 17–32 tidak tersedia. Program tidak melakukan imputasi dan hanya memproses berkas yang dapat diparsing. Akibatnya, eksperimen menggunakan 72 video dan 2.986 sesi pengguna-video.

Jejak arah telah diseragamkan menjadi 10 Hz oleh penyedia dataset. Penelitian tidak memakai berkas video atau *frame* sebagai masukan, sehingga FPS video sumber bukan variabel eksperimen. Indeks pengguna dimulai kembali pada setiap video; evaluasi karena itu mengukur generalisasi ke video yang belum terlihat, bukan ke pengguna baru.

`data/split_meta.json` menyimpan daftar ID video untuk latih, validasi, dan uji. Tiga berkas `windows_*.npz` menyimpan representasi turunan berupa vektor satuan tiga dimensi, target empat horizon, ID video, dan indeks pengguna. Data mentah tidak disertakan.
