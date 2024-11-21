# Laporan Proyek Machine Learning - Abid Juliant Indraswara

![gambar-mesin](https://raw.githubusercontent.com/Abito21/Project_Predictive_Analytics_Machine_Maintenace/main/pictures/machine.jpg)
*Source : https://pixabay.com/photos/gears-cogs-machine-machinery-1236578/*

## Domain Proyek

### Latar Belakang

Mesin dalam artian umum dikaitkan dengan suatu objek yang dapat merubah suatu sumber energi menjadi energi dalam upaya mempermudah pekerjaan atau menghasilkan sesuatu. Mesin hampir selalu berkaitan dengan yang namanya komponen motor (kebalikan dari generator atau versi kecil dinamo) yang memiliki sifat sementara. Mesin motor ini pada pabrik usaha kecil, menengah maupun besar memberikan akses kemudahan manusia dalam mengelola dan mempermudah pekerjaan misalnya traktor untuk membajak sawah, conveyor untuk memindahkan barang, mesin press, lift, eskalator dan sejenisnya.

Penggunaannya secara waktu juga beraneka ragam namun di pabrik mesin motor sebisa mungkin 24 jam harus berjalan atau setidaknya tetap dalam kondisi on karena starting motor di pabrik membutuhkan warm up lama dan membutuhkan biaya yang besar kecuali pada saat maintenance yang umunya dilakukan setahun sekali tepatnya di hari raya besar [\[1\]](https://doi.org/10.1109/CSNT51715.2021.9509696). Kebutuhan akan mesin bisa dikatakan sudah melekat di jaman modern sekarang, namun ada beberapa hal yang menjadi perhatian salah satunya mengenai kualitas mesin. Kualitas mesin ditentukan oleh banyak faktor namun penggunaan yang berlebihan dapat menyebabkan kualitas mesin menurun berakibat pada gagalnya mesin bekerja [\[2\]](https://doi.org/10.1109/AI4I49448.2020.00023).

Dalam upaya pencegahan kualitas menurun perlu adanya perawatan (maintenace). Perawatan sendiri berguna untuk menjaga kualitas mesin dan deteksi dini agar tidak terjadi kegagalan mesin bekerja. Perawatan dilakukan dalam kurun waktu tertentu atau melihat dari beberapa faktor yang barangkali bisa menjadi deteksi dini kegagalan mesin sehingga bisa dilakukan perawatan lebih awal [\[3\]](http://dx.doi.org/10.3390/app10010213). Oleh karenanya perlu suatu sistem yang dapat melakukan prediksi kualitas mesin untuk dapat melakukan deteksi dini kualitas atau gangguan mesin dalam upaya maintenance awal.
  
### Referensi
[1] K. Purnachand. etc, "[Predictive Maintenance of Machines and Industrial Equipment](https://doi.org/10.1109/AI4I49448.2020.00023)", IEEE, 2021.

[2] Matzka Stephan, "[Explainable  Artificial Intelligence for Predictive Maintenance Applications](https://doi.org/10.1109/AI4I49448.2020.00023)", IEEE, 2020.

[3] Petr Stodola and Jiˇrí Stodola, "[Model of Predictive Maintenance of Machines and Equipment](http://dx.doi.org/10.3390/app10010213)", IEEE, 2019.

## Business Understanding

Prediksi kualitas mesin untuk dapat menentukan perawatan mesin yang tepat diperlukan beberapa kriteria yang mecakup pada mesin motor. Mesin memiliki banyak jenis namun pada projek ini dimaksudkan pada mesin motor secara umum. Mesin dapat mengalami kerusakan yang timbul diantaranya akibat penggunaan yang berlebihan baik segi waktu, bobot maksimal yang dilewati, mesin terlalu panas, gangguan eksternal dan lain lain. Kerusakan tersebut dapat diminimalisir dengan perawatan yang baik dan tepat. Melakukan prediksi untuk mesin mana yang memiliki ciri-ciri serta menklasifikasikannya ke kategori tertentu dapat memberikan jawaban untuk bisa meminimalisir kerusakan dan deteksi dini kegagalan mesin. Mesin dapat bekerja dengan baik dan maksimal dengan perawatan yang baik dan konsisten.

### Problem Statements
- Sistem seperti apa yang dibuat dalam upaya menentukan prediksi perawatan mesin yang baik?
- Faktor apa yag paling berpengaruh dalam menentukan prediksi perawatan mesin?
- Mesin seperti apa yang mempunyai kualitas baik dan kualitas buruk sehingga perlu dilakukan perawatan?

### Goals

- Sistem prediksi perawatan mesin dibuat melalui sistem klasifikasi dengan integrasi kecerdasan buatan berbasis machine learning sistem.
- Faktor yang paling berpengaruh terhadap prediksi perawatan mesin dapat ditentukan melalui korelasi antar faktor terhadap kelas kategori kualitas mesin.
- Mesin dengan kualitas baik dan buruk bisa dilihat dari faktor-faktor yang sudah tersedia.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution Statements
Prediksi perawatan mesin terdapat beberapa kategori dengan nilai tertentu. Kategori mesin termasuk ke dalam permasalahan klasifikasi. Sehingga metodologi yang cocok untuk pada projek ini adalah membangun model klasifikasi dengan kategori target dan tipe kegagalan sebagai kelas.
- Membuat sistem prediksi yang berbasis machine learning dengan 5 pilihan algoritma diantaranya KNN, SVM, Decision Tree, Random Forest dan XGBoost. Diambil salah satu algoritma dengan akurasi terbaik dalam melakukan prediksi data test.
- Metrik evaluasi pada model klasifikasi umumnya digunakan accuracy, precision, recall, F-1 score dan confusion matrix. Metrik evaluasi yang disebutkan mengukur ketepatan dalam mengklasifikasikan suatu ketagori bergantung pada jenis data, imbangan kelas dan tujuan spesifik model.

Membuat model prediktif pada machine learning sangat membutuhkan data. Pada mesin maintenance terdapat data yang dapat diakses secara open-source di kaggle. Dataset machine predictive maintenance classification pada link berikut [dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data) . Merupakan dataset sintetis yang merefleksikan prediksi perawatan mesin secara real-world yang ditemui di industri.

## Data Understanding
Dataset yang digunakan adalah Machine Predictive Maintenance Classification dari Kaggle pada [link](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data) berikut. Paper dipublikasikan oleh Stephan Matzka pada 2020 di platform IEEE yang berjudul "[Explainable Artificial Intelligence for Predictive Maintenance Applications](https://doi.org/10.1109/AI4I49448.2020.00023)". Dataset ini cukup besar sehingga cukup satu ini digunakan dalam membuat model klasifikasi. 

Dataset prediksi perawatan mesin secara umum sulit untuk didapatkan dan dipublikasikan khususnya penggunaan di industri pabrik hal ini terkait masalah privasi yang ada. Sehingga dataset yang akan digunakan projek nantinya merupakan kumpulan data sintetis yang mencerminkan data nyata pemeliharaan prediktif yang sering ditemui di industri. Dataset terdiri dari 10000 data poin berdasarkan UID (Unique ID) dengan 10 variabel kolom.

### Beberapa 10 variabel atau parameter tersebut diantaranya:
- UID merupakan identifikasi unik dari setiap data dari 1 hingga 10000 jumlahnya sesuai yang ada di dataset.
- ProductID yaitu variasi spesifik serial number yang direpresentasikan sebagai type dengan 3 jenis tipe L, M dan H.
- Type berisi data dengan keterangan L untuk low (50% dari keseluruhan produk), M untuk medium (30% dari keseluruhan produk) dan H untuk high (20% dari keseluruhan produk) yang merupakan jenis kualitas produk.
- Air Temperatue [K] merupakan data temperatur udara yang dihasilkan dari proses random kemudian dinormalisasi ke standar deviasi dengan rentangnilai 2 K hingga 300 K.
- Process Temperature [K] merupakan data temperatur proses yang dihasilkan dari proses random kemudian dinormalisasi ke standar deviasi dengan nilai 1 K ditambahkan dengan temperatur udara ditambah 10 K.
- Rotational Speed [rpm] yaitu kecepatan rotasi dihitung dari daya 2860 W, dilapisi dengan kebisingan yang didistribusikan secara normal.
- Torque [Nm] yaitu nilai torsi umumnya terdistribusi sekitar 40 Nm dengan Ïƒ = 10 Nm dan tidak ada memiliki nilai negatif.
- Tool Wear [min] merupakan varian kualitas H/M/L dengan menambahkan keausan alat selama 5/3/2 menit pada alat yang digunakan dalam proses tersebut serta label 'machine failure' yang menunjukkan apakah mesin telah gagal pada titik data khusus dengan salah satu mode kegagalan adalah True.
- Target merupakan nilai kategori target dengan nilai Failure atau Not.
- Failure Type merupakan nilai kategori target terdiri 5 mode kegagalan yaitu no failure, heat dissipation failure, power failure, overstrain failure dan tool wear failure.

Pada parameter target dan failure type merupakan kolom target dari model klasifikasi yang akan dibuat. Target memiliki dua nilai sehingga bisa dikatakan binary classification. Sedangkan Failure type memiliki 5 nilai sehingga bisa dikatakan multi classification. Sisa kolom mendefinisikan ID, ProductID, Type serta lainnya ciri-ciri dari kegagalan mesin dengan nilai tertentu. Untuk menguraikan data diatas maka diperlukan proses analisis yang terdiri dari tahapan 
1. Import Library dan Dataset
2. Cek Info Dataset 
3. Describe Dataset
4. Cek Nilai Null
5. Cek Nilai NaN
6. Cek Duplikat Data
7. Cek Korelasi Data
8. Cek Distribusi Data
9. Exploratory Data Analysis - Univariate Analysis
   
   Tahapan EDA Univariate Analysis dilakukan untuk dapat menganalisis masing-masing fitur yang ada. Secara kategorik terdapat fitur Type, Failure Type dan secara numerik terdapat fitur air temperature, process temperature, rotational speed, torque, tool wear dan target. Secara visual yang ingin dilihat adalah secara jumlah data dan persebaran datanya sehingga bisa menyimpulkan beberapa hal seperti kualitas mesin baik dan buruk lebih banyak terlihat pada fitur apa serta bagaimana persebarannya.

10. Exploratory Data Analysis - Multivariate Analysis

    Tagapan EDA Multivariate Analysis dilakukan untuk mengetahui korelasi antar fitur baik secara jumlah dan persebarannya melalui 2 atau lebih hubungan fitur. Contoh saja ingin mengetahui bagaimana hubungan antara Torque dengan fitur Type dan Failure Type. Kemudian heatmap korelasi antar fitur khususnya untuk mengetahui fitur Target memiliki korelasi baik dengan fitur apa yang nantinya dapat menentukan ciri-ciri mesin apa yang perlu pemeliharaan dari korelasi tersebut.

Informasi mengenai dataset beserta tipe datanya
| #  | Column                      | Non-Null Count | Dtype    |
|----|-----------------------------|----------------|----------|
| 0  | UID                         | 10000 non-null | int64    |
| 1  | Product ID                  | 10000 non-null | object   |
| 2  | Type                        | 10000 non-null | object   |
| 3  | Air temperature [K]         | 10000 non-null | float64  |
| 4  | Process temperature [K]     | 10000 non-null | float64  |
| 5  | Rotational speed [rpm]      | 10000 non-null | int64    |
| 6  | Torque [Nm]                 | 10000 non-null | float64  |
| 7  | Tool wear [min]             | 10000 non-null | int64    |
| 8  | Target                      | 10000 non-null | int64    |
| 9  | Failure Type                | 10000 non-null | object   |

## Data Preparation
Data preparation yaitu tahapan untuk melakukan transformasi data agar sesuai atau dapat dengan mudah digunakan ketika modeling machine learning. Bagian data preparation yang umum dilakukan ada beberapa tahapan diantaranya:
- Dataset Menghilangkan Outlier
- Encoding Fitur Kategori
- Seleksi Fitur
- Pembagian dataset dengan fungsi train_test_split dari library sklearn
- Normaliasi Data

Tahapan diatas secara berurutan dijalankan melalui proses penghilangan outlier karena outlier ini dapat menganggu jalannya train dan test model diitakutkn sensitifitas terhadap data. Kemudian encoding untuk mengubah bentuk kolom kategori. Encoding dilakukan pada tahapan awal agar nantinya mudah dalam melakukan seleksi fitur karena kolom yang tersedia akan jadi bertambang melalui proses encoding. Selanjutnya seleksi fitur dilakukan dengan membagi ke dalam kolom mana saja yang merupakan data dan label. Seleksi fitur akan mengacu pada kolom setelah encoding. Setelah itu baru bisa membagi data ke dalam data train dan data test untuk nantinya digunakan dalam modeling dan testing. Terakhir data pre tahapannya adalah melakukan normalisasi data setelah pembagian data train dan test.

Berikut penjelasan tahapan diatas:
1. Dataset Menghilangkan Outlier : 
Pada bagian EDA - Deskripsi Variabel terlihat bahwa dataset yang dimiliki mempunyai outlier yang cukup banyak. Sehingga perlu dilakukan penyesuaian agar data yang dimiliki mampu mempunyai model yang baik dalam melakukan prediksi. Outlier sangat mengganggu pelatihan model menjadikan prediksi bisa saja underfitting karena tidak mampu mendeteksi klasifikasi yang sesuai. Untuk menghindari outlier ada cara yang digunakan yaitu mengetahui batas bawah dan batas atas kemudian bisa melakukan remove pada data yang tidak sesuai dengan rumus tersebut.
2. Encoding Fitur Kategori : 
Hal ini terkait dengan beberapa fitur kategori yang ada pada kolom Type dan Failure Type, perlu dilakukan penyesuaian agar model yang nantinya dibuat bisa membaca dengan baik. Jadi kedua kolom tersebut perlu diubah bentuknya menjadi beberapa kolom lagi yang mendefinisikan nilai numerik dari kategori yang dimaksud. Teknik yang yang digunakan adalah one-hot encoding yang sudah tersedia di library scikit-learn.
3. Seleksi Fitur : 
Seleksi fitur digunakan untuk mengelompokkan data yang termasuk ke dalam variabel data dan variabel label. Karena berbentuk klasifikasi maka variabel label yang terdiri dari beberapa kelas kualitas mesin akan digunakan dalam acuan prediksi data perawatan mesin.
4. Pembagian dataset dengan fungsi train_test_split dari library sklearn : 
Variabel data dan variabel label perlu dibagi menjadi 2 secara terpisah yaitu yang termasuk ke dalam data train dan data test. Data train digunakan untuk membentuk model sedangkan data test digunakan untuk menguji seberapa baik model yang dibuat terhadap data test.
5. Standardisasi Data
Standardisasi data merupakan tahapan dimana data akan dilakukan proses scaling dan standardisasi sehingga data yang sebelumnya memiliki nilai rentang bilangan bulat ataupun desimal menjadi bernilai rentang antara -1 s.d 1 dengan menggunakan metode Z-score scaling. Hal ini perlu dilakukan karena sistem komputer cenderung dapat mengenali suatu nilai -1 hingga 1.

## Modeling
## Model Development

Bagian model development akan berfokus pada pembuatan model dengan beberapa algoritma machine learning sebagai solusi untuk problem statemen yang muncul di awal. Ada beberapa algoritma yang bisa digunakan pada projek ini untuk dicoba dan dipilih mana model yang terbaik. Algoritma yang akan digunakan diantaranya :   
1. K-Nearest Neighbor
2. SVM
3. Decision Tree
4. Random Forest
5. Boosting Algorithm

Masing-masing algoritma diatas memiliki kelebihan dan kekurangan masing-masing yang akan dijabarkan pada tiap bagiannya.

### Algoritma K-Nearest Neighbor

Algoritma KNN atau singkatan dari K-Nearest Neighbor merupakan algoritma machine learning tipe supervised learning yang umumnya digunakan pada permasalahan klasifikasi atau regresi. Bekerja dengan cara mencari K tertangga terdekat (disebut neighbors) dari data yang ingin diprediksi. Algoritma ini mengambil mayoritas kelas dari tetangga terdekat untuk klasifikasi atau rata-rata nilai untuk regresi. 

Kelebihan :

- Salah satu algoritma yang paling mudah dipahami dan diimplementasikan.
- Algoritma ini adalah instance-based learning, yang berarti tidak perlu fase pelatihan eksplisit, cukup menyimpan dataset untuk digunakan saat prediksi.
- KNN bekerja dengan baik pada dataset kecil hingga menengah.

Kekurangan :

- KNN memerlukan waktu komputasi yang besar saat melakukan prediksi karena perlu menghitung jarak ke semua data latih, sehingga tidak efisien untuk dataset besar.
- Karena berbasis pada jarak, KNN sangat sensitif terhadap skala fitur dan noise dalam data.
- KNN menyimpan seluruh dataset, yang bisa memakan banyak memori, terutama untuk dataset besar.

### Algoritma SVM (Support Vector Machine)

Algoritma SVM atau singkatan dari Support Vector Machine merupakan algoritma machine learning tipe supervised learning yang umumnya digunakan pada permasalahan klasifikasi atau regresi. SVM bekerja dengan mencari hyperplane terbaik yang memisahkan data menjadi dua kelas. Pada kasus non-linear, kernel digunakan untuk mengubah data ke dimensi yang lebih tinggi agar pemisahan lebih mudah dilakukan 

Kelebihan :

- SVM sangat baik untuk bekerja dengan data yang memiliki banyak fitur (berdimensi tinggi).
- Algoritma SVM bekerja dengan baik ketika kelas terpisah dengan jelas dalam data.
- Karena margin yang jelas, SVM cenderung menghindari overfitting.

Kekurangan :

-  SVM memerlukan waktu komputasi yang besar, terutama pada dataset besar, karena mencari hyperplane optimal memerlukan perhitungan yang intensif.
- Pilihan kernel yang buruk atau parameter yang tidak optimal dapat membuat SVM kurang efektif.
- SVM dapat bekerja buruk jika dataset memiliki banyak noise atau overlap antara kelas.

### Algoritma Decsion Tree

Algoritma Decision merupakan algoritma machine learning tipe supervised learning yang umumnya digunakan pada permasalahan klasifikasi atau regresi. Dari namanya bisa diketahui bahwa decision tree menggambarkan sebuah pohon dengan banyak cabang. Melalui penggambaran tersebut algoritma ini bekerja dengan membagi data berdasarkan fitur-fitur yang ada, membangun struktur pohon keputusan yang berupa cabang-cabang (decisions) dan daun (outcomes).

Kelebihan :

- Decision tree sangat mudah dipahami dan diinterpretasikan.
- Decision tree dapat menangani berbagai jenis data tanpa perlu normalisasi.
- Decision tree relatif cepat dalam pelatihan dan dapat digunakan untuk dataset besar.

Kekurangan :

-  Decision tree cenderung overfit pada data jika pohon terlalu dalam, mempelajari noise dari data.
- Algoritma ini bisa sangat sensitif terhadap fluktuasi kecil dalam data, yang bisa menghasilkan pohon yang sangat berbeda.
- Decision tree mungkin tidak bekerja baik untuk data yang memiliki hubungan yang sangat non-linear dan kompleks.

### Algoritma Random Forest

Algoritma Random Forest merupakan algoritma machine learning tipe supervised learning yang umumnya digunakan pada permasalahan klasifikasi atau regresi. Merupakan algoritma lanjutan decision tree dengan banyaknya pohon seperti layaknya hutan bisa dikatakan ensemble method decision tree untuk membuat keputusan. Setiap pohon dalam hutan (forest) dilatih dengan subset acak dari data dan fitur, kemudian hasilnya digabungkan untuk membuat prediksi akhir (dengan voting untuk klasifikasi atau rata-rata untuk regresi).

Kelebihan :

- Karena menggunakan banyak pohon keputusan, Random Forest dapat mengurangi overfitting yang sering terjadi pada decision tree tunggal.
- Random Forest sering memberikan hasil yang sangat akurat dibandingkan dengan model tunggal seperti decision tree.
- Random Forest dapat menangani data yang besar dan memiliki banyak fitur dengan baik.

Kekurangan :

-  Meskipun lebih akurat, model Random Forest sulit untuk dipahami dan dijelaskan dibandingkan dengan decision tree tunggal.
- Pelatihan Random Forest memerlukan waktu yang lebih lama dibandingkan dengan algoritma tunggal seperti decision tree karena banyaknya pohon yang dilibatkan.
- Random Forest membutuhkan lebih banyak memori untuk menyimpan banyak pohon.

### Algoritma Boosting (XGBoost)

Algoritma Boosting memiliki banyak jenisnya diantaranya AdaBoost, Gradient Boosting dan XGBoost. Boosting adalah ensemble method yang berfokus pada meningkatkan akurasi model dengan menggabungkan beberapa model yang lebih lemah (weak learners) menjadi model yang lebih kuat. Setiap model berikutnya dilatih untuk mengoreksi kesalahan dari model sebelumnya, sehingga dapat meningkatkan prediksi secara bertahap. Pada projek ini menggunakan XGBoost sebagai Boosting Algorithm.

Kelebihan :

- Boosting sering memberikan hasil yang sangat akurat, bahkan pada data yang kompleks, dengan meningkatkan performa model secara bertahap.
- Boosting dapat menangani data yang tidak seimbang (misalnya kelas yang jarang muncul) dengan lebih baik.
- Boosting dapat mengubah model yang lemah menjadi model yang sangat kuat.

Kekurangan :

-  Karena melibatkan beberapa model, pelatihan algoritma boosting bisa memakan waktu lebih lama dibandingkan dengan model lain.
- Meskipun algoritma boosting dapat menangani overfitting lebih baik, jika tidak hati-hati, boosting bisa terlalu berfokus pada data latih dan mengarah pada overfitting.
- Model boosting, seperti Random Forest, cenderung sulit untuk dipahami dan dijelaskan karena melibatkan banyak model.

Dari kelima model yang sudah dibuat dari akurasinya saja yang memiliki nilai tertinggi ada pada model algoritma decision tree, random forest dan algoritma XGBoost. Keduanya menghasilkan nilai yang mirip termasuk confusion matriknya. Namun berdasarkan kelebihan dan kekurangannya bisa dibilang XGBoost jauh lebih baik karena memiliki keunggulan dalam menangani data yang tidak seimbang (misalnya kelas yang jarang muncul) dengan lebih baik dan kebutuhan memory yang lebih sedikit dibandingkan dengan Random Forest. Random Forest dibentuk dari esemble method Decision Tree dengan banyak pohon sehingga kurang cocok apabila dataset akan berkembang lebih besar. XGBoost bisa dikatakan sebagai solusi atas dataset besar kedepannya apabila dikembangkan.

## Evaluation
Selain menggunakan metrik akurasi dan confusion matrix pada bagian ini dilihat juga metrik evaluasi ROC Curve dan AUC untuk menentukan mana yang lebih baik secara mendalam. Berikut beberapa penjelasan metrik evaluasi yang digunakan.
- Akurasi : merupakan pengukuran proporsi prediksi yang benar terhadap total data. Cocok untuk kasus klasifikasi jika data tidak memiliki ketidakseimbangan kelas (class imbalance).
- Confusion Matrix : Menggunakan True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN) untuk memberikan gambaran yang lebih mendalam mengenai kinerja model.
- ROC Curve & AUC :
  - ROC Curve menggambarkan trade-off antara True Positive Rate (Recall) dan False Positive Rate (FPR) pada berbagai threshold.
  - AUC adalah ukuran kinerja keseluruhan dari model klasifikasi, semakin tinggi AUC, semakin baik model dalam membedakan antara kelas positif dan negatif.

Metrik Evaluasi Akurasi

| Model         | KNN       | SVM       | Decision Tree | Random Forest | XGBoost  |
|---------------|-----------|-----------|---------------|---------------|----------|
| **train_acc** | 0.993707  | 0.999213  | 1.0           | 1.0           | 1.0      |
| **test_acc**  | 0.969061  | 0.998427  | 0.998427      | 0.998427      | 0.998427 |

Metrik Evaluasi ROC AUC
| Model         | KNN       | SVM       | Decision Tree | Random Forest | XGBoost  |
|---------------|-----------|-----------|---------------|---------------|----------|
| **train**     | 0.999755  | 0.996669  | 1.0           | 1.0           | 1.0      |
| **test**      | 0.5       | 0.5       | 0.974576      | 0.968069      | 0.975636 |


Dari hasil evaluasi ROC AUC diatas bisa dilihat jika model XGBoost memiliki nilai yang tinggi sehingga hasilnya bisa dijadikan acuan mendalam dari model akurasi.

### Evaluasi

Melalui hasil yang sudah dibuat dari mulai pembuatan model hingga membantuk metrik evaluasi menjadi jawaban atas problem statement yang ditanyakan di awal.
- Sistem seperti apa yang dibuat dalam upaya menentukan prediksi perawatan mesin yang baik?

  Goals : Sistem prediksi perawatan mesin dibuat melalui sistem klasifikasi dengan integrasi kecerdasan buatan berbasis machine learning sistem.

  Answer : Pada model machine learning yang sudah dibuat bisa dilihat bahwa model sangat beradaptasi dengan data-data mesin sehingga mampu memberikan akurasi yang besar. Hanya dengan data kurang lebih 10000 akurasi bisa mencapai leboh dari 90% untuk semua model baik training beserta maupun tahap pengujian dengan sampel 20%. Sehingga sistem prediksi pemeliharaan mesin sangat cocok dalam upaya menentukan prediksi mesin mana yang perlu mendapatkan pemeliharaan segera maupun tidak.

- Faktor apa yag paling berpengaruh dalam menentukan prediksi perawatan mesin?

  Goals : Faktor yang paling berpengaruh terhadap prediksi perawatan mesin dapat ditentukan melalui korelasi antar faktor terhadap kelas kategori kualitas mesin.

  ![heatmap](https://raw.githubusercontent.com/Abito21/Project_Predictive_Analytics_Machine_Maintenace/main/pictures/heatmap.png)
  *Figure 1 : Heatmap Korelasi Fitur Numerik*

  Answer : Melalui heatmap pada bagian EDA - Multivariate Anlysis Numerical Features terdapat gambaran korelasi target terhadap semua variabel numeric lainnya. Heatmap yang memiliki nilai kearah positif mempunyai korelasi yang kuat sebaliknya nilai negatif mempunyai korelasi yang lemah. Sehingga melalui heatmap tersebut bisa disimpulkan bahwa korelasi yang kuat terhadap fitur Target yaitu Torque (0.22), Tool wear (0.12), Air temperature (0.09) dan Process temperature (0.04). Rotational speed (-0.17) menandakan bahwa variabel ini memiliki korelasi lemah dengan target sedangkan untuk UID tidak memiliki korelasi apapun karena merupakan nilai unik per record mesin.

- Mesin seperti apa yang mempunyai kualitas baik dan kualitas buruk sehingga perlu dilakukan perawatan?

  Goals : Mesin dengan kualitas baik dan buruk bisa dilihat dari faktor-faktor yang sudah tersedia.

  ![describe-dataset](https://raw.githubusercontent.com/Abito21/Project_Predictive_Analytics_Machine_Maintenace/main/pictures/describe-dataset.png)

  *Figure 2 : Deskripsi dataset yang digunakan untuk analisis prediktif*

  ![distribution-of-target](https://raw.githubusercontent.com/Abito21/Project_Predictive_Analytics_Machine_Maintenace/main/pictures/distribution-of-target.png)

  *Figure 3: Distribusi variabel target dalam dataset.*

  ![Failure-Type](https://raw.githubusercontent.com/Abito21/Project_Predictive_Analytics_Machine_Maintenace/main/pictures/Failure-Type.png)

  *Figure 4: Tipe-tipe kegagalan yang ditemukan dalam dataset.*

  Answer : Jawabannya akan berkaitan dengan jawaban dari pertanyaan kedua, melalui hasil heatmap. Mesin dengan kualitas baik pasti memiliki nilai Target 0 sedangkan kualitas buruk perlu dilakukan pemeliharaan memiliki nilai Target 1. Korelasi dari heatmap menghubungkan antara variabel numerik dengan fitur Target yang mana kualitas baik diidentifikasi berdasarkan variabel Torque, Tool wear, Air temperature dan Process temperature. Melalui persebaran datanya Target dengan nilai 1 memenuhi 9263 data hampir 97% dari dataset memiliki nilai mesin kualitas baik atau No Failure. Mesin dengan kualitas baik melalui hasil describe memiliki nilai rata-rata Torque 39.98Nm, Tool wear 107.95min, Air temperature 300K dan Process temperature 310K. Mesin dengan kualitas buruk memiliki nilai yang tinggi melebihi rata-rata atau dengan nilai maksimum Torque 76Nm, Tool wear 253min, Air temperature 304K dan Process temperature 313K.

Solution Statement

- Membuat sistem prediksi yang berbasis machine learning dengan 5 pilihan algoritma diantaranya KNN, SVM, Decision Tree, Random Forest dan XGBoost. Diambil salah satu algoritma dengan akurasi terbaik dalam melakukan prediksi data test.

  Dampak : Dengan membuat 5 model machine learning dalam membuat model prediksi bisa diketahui manakah model yang cukup baik diterapkan dalam sistem prediksi. Karena masing-masing algoritma mempunyak keunggulan dan kekurangan. Dilihat dari akurasi training dan testing bisa cukup yakin menggunakan akurasi yang besar. Dari akurasi tersebut bisa memprediksi dengan tepat pada tahap pengujian sehingga bisa diterapkan pada sistem prediksi.

- Metrik evaluasi pada model klasifikasi umumnya digunakan accuracy, precision, recall, F-1 score dan confusion matrix. Metrik evaluasi yang disebutkan mengukur ketepatan dalam mengklasifikasikan suatu ketagori bergantung pada jenis data, imbangan kelas dan tujuan spesifik model.

  Dampak : Metrik evaluasi ini sebagai acuan daripada penentuan model algorima terbaik. Dengan 5 algoritma diatas untuk dapat menguku mana yang lebih baik dibutuhkan metrik evaluasi. Metrik evaluasi diatas masih terdapat bebebrapa kesamaa hasil pada algoritma decision tree, random forest dan XGBoost. Sehingga pada projek ini ada tambahan metrik untuk mengukur ROC AUC dan hasilnya bisa mengeliminasi mana model yang terbaik yaitu XGBoost. Efek dari pemilihan metrik ini cukup besar dalam penentuan model algoritma terbaik.

## Kesimpulan

Mesin memiliki batasan dalam penggunaannya yang bisa disebabkan oleh banyak faktor internal maupun eksternal. Sehingga perlu dilakukan pemeliharaan mesin agar bisa terawat dan bisa digunakan dalam waktu panjang. Pemeliharaan bisa bersifat kontinyu namun juga bisa dilakukan secara spontan apabila memenuhi ciri-ciri mesin dengan kualitas buruk. Kualitas mesin yang buruk memiliki ciri yang bisa di analisis dan di prediksi menggunakan model machine learning. Projek yang sudah dibuat menggunakan 5 model machine learning yaitu KNN, SVM, Decision Tree, Random Forest dan XGBoots. Kelima model menghasilkan nilai akurasi diatas 90% tetapi model dengan akurasi terbaik ada 3 yaitu Decision Tree, Random Forest dan XGBoost dengan memenuhi akurasi train sebesar 100% dan akurasi test sebesar 99.87%. Melalui metrik evaluasi ROC AUC bisa terlihat lebih mendalam akurasi terbaik dari ketiga model menghasilkan model dengan akurasi terbaik yaitu XGBoost ROC AUC train sebesar 100% dan ROC AUC test sebesar 97,56%. XGBoost sebagai pilihan model machine learning yang tepat selain akurasi dan ROC AUC yang tinggi terdapat beberapa kelebihan yaitu mampu menangani data yang tidak seimbang (misalnya kelas yang jarang muncul) dengan lebih baik dan kebutuhan memory yang lebih sedikit dibandingkan dengan keempat algoritma lainnya.

## Referensi
[1] K. Purnachand. etc, "Predictive Maintenance of Machines and Industrial Equipment", IEEE, 2021.

[2] Matzka Stephan, "Explainable  Artificial Intelligence for Predictive Maintenance Applications", IEEE, 2020.

[3] Petr Stodola and Jiˇrí Stodola, "Model of Predictive Maintenance of Machines and Equipment", IEEE, 2019.
