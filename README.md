# Car Brand Classification — Akbank Derin Öğrenme Bootcamp

Araç marka görsellerini **33 sınıfta** sınıflandıran derin öğrenme projesi.  
Çalışma; veri önişleme ve **augmentation**, **CNN** tabanlı model eğitimi, eğitim eğrileri (accuracy/loss), **Confusion Matrix** & **Classification Report** ile değerlendirme ve karar bölgelerini açıklamak için **Grad-CAM** görselleştirmelerini içerir.

---

## İçindekiler
- [1. Problem Tanımı](#1-problem-tanımı)
- [2. Veri Seti](#2-veri-seti)
- [3. Yöntem](#3-yöntem)
  - [3.1. Önişleme & Augmentation](#31-önişleme--augmentation)
  - [3.2. Mimari (CNN)](#32-mimari-cnn)
  - [3.3. Eğitim Ayarları](#33-eğitim-ayarları)
  - [3.4. Callback’ler](#34-callbackler)
- [4. Değerlendirme](#4-değerlendirme)
  - [4.1. Metrikler](#41-metrikler)
  - [4.2. Grad-CAM](#42-grad-cam)
- [5. Deney Tasarımı ve Tekrarlanabilirlik](#5-deney-tasarımı-ve-tekrarlanabilirlik)
- [6. Hiperparametre Optimizasyonu](#6-hiperparametre-optimizasyonu)
- [7. Sonuçlar](#7-sonuçlar)
- [8. Hata Analizi ve Gelecek Çalışmalar](#8-hata-analizi-ve-gelecek-çalışmalar)
- [9. Hızlı Başlangıç (Kaggle)](#10-hızlı-başlangıç-kaggle)
- [10. Bağlantılar](#11-bağlantılar)
- [11. Lisans](#12-lisans)

---

## 1. Problem Tanımı
Amaç; farklı araç markalarına ait görselleri **çok sınıflı** bir sınıflandırma modeliyle doğru şekilde etiketlemektir. Bu problem, logo, ızgara ve far gibi **ince ayrıntıların** ayrıştırılmasını gerektirir. Bu nedenle veri çoğaltma, düzenli eğitim ve açıklanabilirlik (Grad-CAM) önemlidir.

---

## 2. Veri Seti
- **Kaynak:** Kaggle — *Car Brand Classification* (ahmedelsany)
- **Sınıf sayısı:** 33  
- **Örnek sayısı:** ~16.000 görsel  
- **Kullanım:** Kaggle Notebook’ta “Add Data” ile `ahmedelsany/car-brand-classification-dataset`

---

## 3. Yöntem

### 3.1. Önişleme & Augmentation
- **Girdi boyutu:** `224×224` (RGB)  
- **Normalizasyon:** `tf.image` ile `[0, 1]` aralığı  
- **Bölme:** Stratified **train/val/test = 80/10/10**  
- **Augmentation:** `RandomFlip('horizontal')`, `RandomRotation(0.1)`, `RandomZoom(0.1)`, `RandomContrast(0.1)`

### 3.2. Mimari (CNN)
Aşağıdaki yapı kullanılır:

| Blok | Katmanlar | Açıklama |
|------|-----------|----------|
| 1 | Conv2D(32, 3, ReLU) → MaxPool | Temel kenar/doku özellikleri |
| 2 | Conv2D(64, 3, ReLU) → MaxPool | Orta seviye özellikler |
| 3 | Conv2D(128, 3, ReLU) → **Conv2D(128, 3, ReLU, name='last_conv')** → MaxPool | Grad-CAM için `last_conv` |
| Çıkış | GlobalAveragePooling2D → Dropout(0.4) → Dense(33, Softmax) | Sınıf olasılıkları |

### 3.3. Eğitim Ayarları
- **Optimizer:** Adam (`lr = 1e-3`)  
- **Batch size:** 32  
- **Epoch:** 20 (erken durma etkin)  
- **Seed:** 42 (numpy/TF için sabitlenmiş)  
- **Kayıtlar:**  
  - `best_cnn.keras` — en iyi doğrulukta kaydedilen model  
  - `label_map.json` — sınıf indeks–isim eşlemeleri  
  - `history.csv` — epoch bazlı metrikler  

### 3.4. Callback’ler
- **ModelCheckpoint:** `val_accuracy` en iyi olduğunda ağırlıkları kaydeder  
- **ReduceLROnPlateau:** Kayıp iyileşmediğinde `lr` düşürülür  
- **EarlyStopping:** `val_loss` durduğunda eğitim erken sonlanır (best weights geri yüklenir)

---

## 4. Değerlendirme

### 4.1. Metrikler
- **Accuracy/Loss eğrileri** (train/val)  
- **Confusion Matrix**  
- **Classification Report** (precision/recall/F1, sınıf bazlı)  
- **Top-k** doğruluk (örn. top-5) desteği

### 4.2. Grad-CAM
Modelin karar verirken görselin **hangi bölgelerine odaklandığı** gösterilir.  
Logo, ızgara ve far konturları gibi ayırt edici bölgeler ön plana çıkar.

---

## 5. Deney Tasarımı ve Tekrarlanabilirlik
- **Ortam:** Kaggle (GPU)  
- **Determinism:** `SEED = 42` (numpy/TF)  
- **Veri işleme:** `tf.data` pipeline + prefetch  
- **Çıktılar:** En iyi model, metrikler ve sınıf eşlemeleri dosyalanır

Tek görsel tahmini için notebook fonksiyonu:
```python
predict_image("<gorsel_yolu>.jpg", top_k=5)
```

---

## 6. Hiperparametre Optimizasyonu
`learning rate`, `dropout` ve `batch size` üzerinde tarama yapılır; küçük bir ızgara/rasgele arama ile en iyi `val_accuracy` belirlenir. Deney sonuçları ilgili bölümde özetlenir.

---

## 7. Sonuçlar
- **Test Accuracy:** …  
- **Macro F1:** …  
- **Karışan sınıflar:** …  
- **Grad-CAM gözlemleri:** Modelin odaklandığı bölgeler (logo, ızgara, far vb.) …

Eğitim eğrileri ve Confusion Matrix görselleri:
```
images/train_curves.png
images/confusion_matrix.png
```

---

## 8. Hata Analizi ve Gelecek Çalışmalar
- Benzer gövde/renk/çekim açısı nedeniyle karışan sınıflar  
- Sınıf dengesizliği için class weight/focal loss yaklaşımları  
- Daha güçlü backbone ile fine-tuning  
- Veri kalitesi iyileştirme (bulanık veya etiketi hatalı imgelerin temizliği)

---





## 9. Hızlı Başlangıç (Kaggle)
1. **Settings → Accelerator → GPU**  
2. **Add Data:** `ahmedelsany/car-brand-classification-dataset`  
3. **Restart & Run All**

Çalışma sonunda üretilen dosyalar:
- `best_cnn.keras`  
- `label_map.json`  
- `history.csv`

---

## 10. Bağlantılar
- **Veri seti:** https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification-dataset

---

## 11. Lisans
MIT License
