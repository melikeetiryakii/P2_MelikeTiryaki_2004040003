Proje: Ses Duygu Tanıma
Bu proje, ses dosyalarındaki duyguları tanımak için geliştirilmiş bir makine öğrenmesi uygulamasıdır. Flask, TensorFlow ve Librosa gibi kütüphaneleri kullanarak duyguları tanımlayan bir web uygulaması sunar.

Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
Flask==2.0.2
tensorflow==2.6.0
librosa==0.8.1
numpy==1.21.2
sounddevice==0.4.2
scikit-learn==0.24.2

pip install -r requirements.txt

Kurulum
Bu projeyi klonlayın:
git clone https://github.com/melikeetiryakii/P2_MelikeTiryaki_2004040003
.git
cd P2_MelikeTiryaki_2004040003

Gerekli kütüphaneleri yükleyin:


pip install -r requirements.txt
Model dosyasını yükleyin veya eğitin:

Modeli eğitmek için main.py dosyasını çalıştırın:
python main.py

Web uygulamasını başlatın:
python app.py

Kullanım
Tarayıcınızda http://127.0.0.1:5000 adresine gidin.
Bir ses dosyası yükleyin veya mikrofon ile kayıt yapın.
Yüklenen veya kaydedilen ses dosyasının duygusal analiz sonucunu görün.

Dosya Yapısı
app.py: Flask web uygulaması.
predict_from_microphone.py: Mikrofon ve ses dosyasından duygu tahmini yapar.
main.py: Modeli eğitir ve değerlendirir.
test.py: Eğitilmiş modeli kullanarak test verileri üzerinde tahminler yapar.
templates/: HTML şablonları içerir.
