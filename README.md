# Riset-Artikel-Research-Gap-
Pemanfaatan Machine Learning menggunakan algoritma Neural Network
untuk Tuning PID dalam Kontrol Suhu iTCLab
BAB 1
Pendahuluan
Dalam beberapa dekade terakhir, kontrol otomatis telah menjadi aspek penting dalam berbagai industri, mulai dari manufaktur hingga pengelolaan energi. Pengendalian suhu, khususnya, memainkan peranan krusial dalam banyak aplikasi industri, seperti proses kimia, penyimpanan makanan, dan sistem HVAC (Heating, Ventilation, and Air Conditioning). Di antara berbagai teknik kontrol yang ada, PID (Proportional-Integral-Derivative) merupakan metode yang paling banyak digunakan. PID berfungsi untuk menjaga variabel yang diinginkan, seperti suhu, dalam rentang yang ditentukan dengan cara menyesuaikan input berdasarkan kesalahan antara nilai yang diinginkan dan nilai aktual.

Namun, tuning parameter PID—yang meliputi Kp (gain proporsional), Ki (gain integral), dan Kd (gain derivatif)—sering kali menjadi tantangan. Proses ini dapat memakan waktu dan memerlukan pemahaman mendalam tentang karakteristik sistem. Metode tuning tradisional, seperti Ziegler-Nichols dan metode trial-and-error, memiliki keterbatasan, terutama ketika berhadapan dengan sistem nonlinier dan dinamis (Zhang et al., 2020). Oleh karena itu, pencarian metode tuning yang lebih efektif dan efisien menjadi sangat penting.Dengan kemajuan teknologi dalam bidang machine learning, ada potensi besar untuk meningkatkan proses tuning PID. Algoritma Neural Network, yang dikenal karena kemampuannya untuk belajar dari data dan mengidentifikasi pola, telah menunjukkan hasil yang menjanjikan dalam berbagai aplikasi, termasuk pengendalian sistem. Penerapan deep learning dalam tuning PID di iTCLab, seperti yang dijelaskan oleh Rahmat et al. (2023), memberikan indikasi bahwa pendekatan ini dapat menghasilkan peningkatan yang signifikan dalam performa kontrol suhu, serta mengurangi waktu respons dan overshoot.

Dengan latar belakang ini, artikel ini bertujuan untuk menjelajahi pemanfaatan algoritma Neural Network dalam tuning PID untuk kontrol suhu di iTCLab. Dengan pendekatan yang sistematis, penelitian ini diharapkan dapat memberikan kontribusi yang berarti dalam pengembangan teknologi kontrol otomatis yang lebih adaptif dan responsif.

Latar Belakang Masalah
Pengendalian (Kontrol) suhu salah satu elemen kunci dalam banyak aplikasi industri, dan stabilitas serta responsivitas sistem kontrol yang penting untuk memastikan operasi yang efisien. Tuning PID, meskipun sudah menjadi metode yang bagus, menghadapi tantangan dalam aplikasi praktis. Berbagai faktor, seperti perubahan dinamis dalam karakteristik sistem dan gangguan eksternal, sering kali menyebabkan kesalahan dalam tuning yang dapat berakibat fatal pada performa sistem pada aplikasi. Menurut penelitian oleh Zhang et al. (2020), kesalahan dalam proses tuning dapat mengakibatkan ketidakstabilan dan performa yang buruk, seperti overshoot yang tinggi, waktu settling yang lama, dan ketidakmampuan sistem untuk mencapai setpoint yang diinginkan. Dengan meningkatnya kompleksitas sistem dan tuntutan untuk kinerja yang lebih baik, kebutuhan akan metode tuning yang lebih adaptif menjadi semakin mendesak. Machine learning, khususnya algoritma Neural Network, telah muncul sebagai solusi inovatif untuk masalah ini. Neural Network mampu menganalisis data dalam jumlah besar, mengenali pola yang kompleks, dan menyesuaikan model berdasarkan kondisi yang berubah-ubah. Penelitian sebelumnya menunjukkan bahwa penggunaan machine learning dalam pengendalian sistem dapat menghasilkan peningkatan signifikan dalam efisiensi dan akurasi kontrol (Almeida et al., 2019). Sebagai contoh, Rahmat et al. (2023) menunjukkan bahwa penggunaan deep learning dalam tuning PID di iTCLab tidak hanya meningkatkan kecepatan respons, tetapi juga mengurangi overshoot dan meminimalkan deviasi suhu.

Seiring dengan tren pengembangan teknologi Internet of Things (IoT), di mana sistem pengendalian dapat terhubung dan saling berkomunikasi, integrasi machine learning dalam sistem kontrol suhu akan memberikan kemampuan real-time yang lebih baik. Sistem IoT dapat memungkinkan pemantauan dan kontrol yang lebih efektif, serta pengumpulan data yang lebih akurat untuk analisis lebih lanjut.Dengan memahami tantangan dan peluang dalam tuning PID serta potensi machine learning, penelitian ini bertujuan untuk mengeksplorasi dan mengimplementasikan algoritma Neural Network untuk meningkatkan performa kontrol suhu di iTCLab. Dengan demikian, diharapkan penelitian ini dapat berkontribusi pada pengembangan sistem kontrol yang lebih adaptif dan efisien, serta membuka jalan bagi penelitian lebih lanjut dalam bidang ini.

BAB 2
TINJAUAN PUSTAKA

2.1 Teori dan Konsep PID
PID adalah metode kontrol yang menggunakan tiga parameter untuk mengontrol sistem.
1. Proportional (P) berfungsi untuk mengurangi kesalahan saat ini dengan memberikan 
output yang proporsional terhadap kesalahan.
2. Integral (I) berfungsi untuk mengeliminasi kesalahan steady-state dengan menghitung 
integral dari kesalahan dari waktu ke waktu.
3. Derivative (D) berfungsi untuk mengantisipasi kesalahan masa depan dengan menghitung 
laju perubahan kesalahan.

2.2 Machine Learning dalam Kontrol
    Machine learning, khususnya algoritma Neural Network, telah muncul sebagai solusi inovatif 
    untuk masalah tuning PID. Neural Network memiliki kemampuan untuk menganalisis data 
    dalam jumlah besar dan mengenali pola yang kompleks. Penelitian sebelumnya menunjukkan 
    bahwa penggunaan machine learning dalam pengendalian sistem dapat menghasilkan 
    peningkatan signifikan dalam efisiensi dan akurasi kontrol (Almeida et al., 2019).
2.3 Penelitian Terdahulu
    Beberapa penelitian telah menunjukkan bahwa penerapan deep learning dalam tuning PID, 
    seperti yang dilakukan oleh Rahmat et al. (2023), dapat meningkatkan kecepatan respons, 
    mengurangi overshoot, dan meminimalkan deviasi suhu.
2.4 Kerangka Pemikiran
    Kerangka pemikiran penelitian ini mencakup penggabungan teori PID dan algoritma Neural 
    Network untuk menciptakan sistem kontrol yang lebih adaptif. Dengan menggunakan 
    machine learning, sistem dapat belajar dari data historis dan menyesuaikan parameter PID 
    secara otomatis.
    
BAB 3
METODOLOGI PENELITIAN

3.1 Desain Penelitian
    Jenis penelitian yang dilakukan adalah penelitian eksperimental dengan pendekatan 
    kuantitatif. Penelitian ini bertujuan untuk menguji efektivitas algoritma Neural 
    Network dalam tuning PID.
3.2 Pengumpulan Data
    Data historis dari sistem kontrol suhu di iTCLab akan dikumpulkan, mencakup 
    variabel input (seperti suhu input dan gangguan) dan output (suhu yang terukur).
3.3 Preprocessing Data
    Data yang dikumpulkan akan diproses untuk membersihkan dan menyiapkannya 
    untuk pelatihan model. Langkah-langkah ini meliputi normalisasi data dan pembagian 
    menjadi data pelatihan dan data pengujian.
3.4 Pelatihan Model Neural Network
    Model Neural Network akan dilatih menggunakan data pelatihan yang telah 
    disiapkan. Arsitektur model akan disesuaikan berdasarkan kompleksitas sistem yang 
    sedang ditangani.
3.5 Evaluasi Model
    Setelah model dilatih, performanya akan diuji menggunakan data pengujian. Kinerja 
    model akan dievaluasi berdasarkan kemampuannya dalam memprediksi parameter 
    tuning PID yang optimal.
3.6 Implementasi dan Pengujian
    Parameter tuning yang dihasilkan oleh model Neural Network akan diterapkan dalam 
    sistem kontrol suhu. Pengujian dilakukan untuk mengevaluasi performa sistem dan 
    membandingkan hasilnya dengan metode tuning tradisional.
    
Solusi
Solusi yang diusulkan dalam penelitian ini adalah pengembangan sistem yang mengintegrasikan algoritma Neural Network untuk melakukan tuning PID secara otomatis. Dengan pendekatan ini, sistem dapat secara dinamis menyesuaikan parameter tuning sesuai dengan perubahan kondisi lingkungan dan karakteristik sistem. Diharapkan bahwa dengan memanfaatkan machine learning, pengendalian suhu di iTCLab akan menjadi lebih responsif dan akurat, mengurangi deviasi suhu yang tidak diinginkan. Sebagaimana diungkapkan oleh Rahmat et al. (2023), integrasi antara teknologi IoT dan deep learning dalam sistem kontrol dapat menciptakan lingkungan yang lebih adaptif dan responsif.

Rekomendasi
Berdasarkan hasil penelitian, berikut adalah beberapa rekomendasi yang dapat diambil:
1. Pengembangan Algoritma: Perlu dilakukan penelitian lebih lanjut untuk mengembangkan algoritma Neural Network yang lebih canggih dan adaptif, serta 
mempertimbangkan pendekatan hybrid dengan algoritma pengendalian lainnya.
2. Penerapan di Sistem Lain: Penerapan algoritma Neural Network untuk tuning PID tidak hanya terbatas pada kontrol suhu, tetapi juga dapat diterapkan pada berbagai sistem kontrol lainnya, seperti tekanan, aliran, dan level.
3. Pendidikan dan Pelatihan: Diperlukan pelatihan dan pendidikan bagi para insinyur dan teknisi dalam menggunakan metode ini untuk memastikan pemahaman yang baik tentang integrasi machine learning dalam sistem kontrol.
   
Referensi
Astrom, K.J., & Wittenmark, B. (1997). Adaptive Control. Pearson.
Zhang, Y., Wang, Y., & Liu, X. (2020). Tuning PID controllers using machine learning 
techniques: A review. Journal of Process Control, 88, 24-36.
Huang, J., Lin, X., & Zhou, Y. (2021). Application of neural networks in PID controller 
design: A survey. Control Engineering Practice, 105, 104607.
Almeida, F. S., Santos, F. S., & Oliveira, L. A. (2019). Machine learning approaches for 
control systems: A systematic review. Control Systems Magazine, 39(1), 33-45.
Rahmat, B., Aditama, A. S., & Nursari, M. (2023). ITCLab PID Control Tuning Using Deep 
Learning. In Proceeding - IEEE 9th Information Technology International Seminar, ITIS 
2023. Institute of Electrical and Electronics Engineers Inc. Available at: 
https://doi.org/10.1109/ITIS59651.2023.10420130
