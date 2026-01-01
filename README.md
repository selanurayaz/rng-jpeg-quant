---

# RNG Tabanlı JPEG Kuantalama Tablosu Deneyi

Bu çalışmada, rastgeleliğin sıkıştırma sistemlerindeki rolünü incelemek amacıyla
basit bir sözde rastgele sayı üreteci (XORSHIFT32) gerçekleştirilmiştir.

Rastgele sayı üreteci doğrudan veri şifrelemek için kullanılmamış; bunun yerine
standart JPEG parlaklık (luminance) kuantalama tablosu üzerinde kontrollü
pertürbasyonlar uygulayarak alternatif bir 8x8 kuantalama tablosu üretmek için
kullanılmıştır.

## Yöntem
- Sözde rastgele sayı üreteci (XORSHIFT32) 구현 edildi
- RNG kullanılarak yeni bir kuantalama-benzeri tablo üretildi
- OpenCV kullanılarak JPEG sıkıştırma uygulandı
- OpenCV özel kuantalama tablolarını doğrudan desteklemediği için,
  üretilen tablonun kuantalama agresifliği JPEG kalite parametresine eşlendi
- Sıkıştırma performansı dosya boyutu ve PSNR metriği kullanılarak değerlendirildi

## Sonuçlar
RNG tabanlı yapılandırmanın, benzer PSNR değerleri korunurken daha küçük dosya
boyutu elde edilmesini sağladığı gözlemlenmiştir. Bu durum, sıkıştırma–kalite
dengesi açısından olumlu bir sonuç olarak değerlendirilmiştir.

## Çıktılar
- `results/summary.md`: sayısal karşılaştırma sonuçları
- `results/*_std_*.jpg`: standart JPEG çıktıları
- `results/*_rng_*.jpg`: RNG tabanlı JPEG çıktıları
- `results/quant_tables_*.png`: kuantalama tablolarının görselleştirilmesi

## Kullanılan Teknolojiler
Python, NumPy, OpenCV, scikit-image

## Güvenlik Perspektifi
Bu çalışmada kullanılan rastgele sayı üreteci kriptografik olarak güvenli değildir.
Ancak deney, rastgelelik kalitesinin olasılıksal davranışa dayanan sistemler
üzerindeki etkisini göstermektedir.

Zayıf veya tahmin edilebilir rastgelelik; hem güvenlik odaklı uygulamalarda
hem de multimedya işleme sistemlerinde olumsuz sonuçlara yol açabilir.

## Basit Şifreleme Gösterimi
Projeye, rastgele sayı üretecinin şifreleme bağlamında nasıl kullanılabileceğini
göstermek amacıyla XOR tabanlı basit bir şifreleme örneği eklenmiştir.

Bu örnek yalnızca **eğitsel amaçlıdır** ve **kriptografik olarak güvenli değildir**.
Amaç, güvenlik sistemlerinde rastgelelik kalitesinin önemini vurgulamaktır.


## RSÜ Statistical Test Output (Sample)

The following screenshot shows the console output of the RSÜ algorithm,
including monobit, chi-square and runs test results.

![RSÜ Test Output](results/rsu_output.png)


**Türkçe Açıklama:**  
Aşağıdaki ekran görüntüsünde RSÜ algoritmasının çalıştırılması
sonucunda elde edilen monobit, ki-kare ve runs testlerine ait
örnek çıktı gösterilmektedir.


🔧 Kurulum ve Çalıştırma
Gereksinimler

Python 3.9 veya üzeri

Git

Kurulum Adımları

Depoyu bilgisayarınıza klonlayın:

git clone https://github.com/selanurayaz/rng-jpeg-quant.git
cd rng-jpeg-quant

(İsteğe bağlı ancak önerilir) Sanal ortam oluşturun:

python -m venv .venv

Sanal ortamı aktif edin:

.venv\Scripts\activate

Gerekli Python paketlerini yükleyin:

pip install -r requirements.txt


Projenin Çalıştırılması

images/ klasörü içerisine en az bir adet .jpg veya .png formatında
görüntü dosyası ekleyiniz.

Ardından ana programı çalıştırınız:

python main.py


# RNG-Based JPEG Quantization Table Experiment

In this study, a simple pseudo-random number generator (XORSHIFT32) was implemented
to analyze the role of randomness in compression-related systems.

Rather than directly encrypting data, the RNG is used to generate an alternative
8x8 quantization table by applying controlled perturbations to the standard JPEG
luminance quantization table.

## Method
- Implemented a PRNG (XORSHIFT32)
- Generated a new quantization-like table using RNG
- Applied JPEG compression using OpenCV
- Since OpenCV does not allow custom quantization tables directly, the aggressiveness
  of the generated table was mapped to the JPEG quality parameter
- Compression performance was evaluated using file size and PSNR

## Results
The RNG-derived configuration achieved a smaller file size while maintaining
comparable PSNR values, indicating a favorable compression-quality trade-off.

## Outputs
- `results/summary.md`: quantitative comparison
- `results/*_std_*.jpg`: standard JPEG outputs
- `results/*_rng_*.jpg`: RNG-derived outputs
- `results/quant_tables_*.png`: visualization of quantization tables

## Technologies
Python, NumPy, OpenCV, scikit-image

## Security Perspective
Although the implemented RNG is not cryptographically secure,
the experiment demonstrates how randomness quality directly affects
systems that rely on probabilistic behavior.
Weak or predictable randomness can negatively impact both
security-related applications and multimedia processing pipelines.

## Simple Encryption Demonstration
A minimal XOR-based encryption example is included to demonstrate
how a pseudo-random number generator (PRNG) can be used to generate
a keystream for encryption.

This example is provided **for educational purposes only** and is
**not cryptographically secure**. It is intended to highlight the
importance of randomness quality in security-related systems.


