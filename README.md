# Car Brand Classification — CNN (Akbank Derin Öğrenme Bootcamp)

Araç marka görsellerini **33 sınıfta** sınıflandıran bir derin öğrenme projesi.
Proje; veri önişleme, veri çoğaltma (augmentation), **CNN** eğitimi, model değerlendirme
(accuracy/loss grafikleri, **Confusion Matrix** & **Classification Report**) ve **Grad-CAM**
görselleştirmelerini içerir. İsteğe bağlı olarak **Transfer Learning** ve küçük bir
**HPO (Hyperparameter Optimization)** denemesi de eklenebilir.



---

## İçindekiler
- [Projenin Amacı](#projenin-amacı)
- [Veri Seti](#veri-seti)
- [Yöntem ve Mimariler](#yöntem-ve-mimariler)
- [Deney Düzeneği](#deney-düzeneği)
- [Nasıl Çalıştırılır (Kaggle)](#nasıl-çalıştırılır-kaggle)
- [Eğitim ve Değerlendirme](#eğitim-ve-değerlendirme)
- [Grad-CAM](#grad-cam)
- [Hiperparametre Optimizasyonu (Opsiyonel)](#hiperparametre-optimizasyonu-opsiyonel)
- [Repodaki Dosyalar](#repodaki-dosyalar)
- [Sonuçlar](#sonuçlar)
- [Kaggle Notebook](#kaggle-notebook)
- [Lisans](#lisans)

---

## Projenin Amacı
Araç marka görsellerini (ör. Audi, BMW, Ford, Mercedes, Tesla vb.) **çok sınıflı**
bir CNN modeliyle sınıflandırmak; eğitim sürecinde **overfitting/underfitting**
durumlarını izlemek, modelin tahminlerinde hangi bölgelere odaklandığını
**Grad-CAM** ile görselleştirip yorumlamak ve mümkünse **Transfer Learning**
ile performansı yükseltmek.

## Veri Seti
- Kaynak: **Kaggle — Car Brand Classification** (yazar: *ahmedelsany*)
- Sınıf sayısı: **33**
- Örnek sayısı (yaklaşık): **~16.000** görsel
- Kullanım: Kaggle Notebook’ta **Add Data** ile
  `ahmedelsany/car-brand-classification-dataset` eklenir.

> Not: Final için veri setini repoya yüklemen gerekmez; Kaggle notebook linki yeterlidir.

## Yöntem ve Mimariler

**Önişleme**
- Görseller **224×224** boyutuna yeniden örneklenir, `tf.image` ile **[0,1]** aralığına normalize edilir.
- Veri **train/validation/test = 80/10/10** oranında, *stratified* olarak bölünür.

**Veri Çoğaltma (Augmentation)**
- `RandomFlip('horizontal')`, `RandomRotation(0.1)`,
  `RandomZoom(0.1)`, `RandomContrast(0.1)`

**Model (Baseline CNN)**
- `Conv2D` + `MaxPooling2D` blokları
- `GlobalAveragePooling2D` + `Dropout(0.4)`
- `Dense(num_classes, activation='softmax')`

**Callback’ler**
- `ModelCheckpoint` (en iyi `val_accuracy`)
- `ReduceLROnPlateau`
- `EarlyStopping` (best weights)

**Transfer Learning (Opsiyonel)**
- EfficientNetB0 / ResNet50 vb. bir temel ağ ile karşılaştırma ve (isteğe bağlı) fine-tune.

## Deney Düzeneği
- Ortam: **Kaggle Notebook (GPU)** önerilir
- Görüntü boyutu: `224×224`
- Batch size: `32`
- Epoch: `20` (EarlyStopping ile erken durma)
- Optimizer: `Adam(lr=1e-3)`
- Reprodüksiyon: `SEED = 42`
- Kayıt dosyaları (çıktılar):
  - `best_cnn.keras` — en iyi model ağırlıkları
  - `label_map.json` — sınıf indeks–isim eşlemeleri
  - `history.csv` — epoch metrikleri (acc/loss)

## Nasıl Çalıştırılır (Kaggle)
1. Kaggle Notebook’ta **Settings → Accelerator → GPU** seçin.  
2. **Add Data** → `ahmedelsany/car-brand-classification-dataset` ekleyin.  
3. Notebook’u **Restart & Run All** ile çalıştırın.  
4. Çalışma bittiğinde çıktı klasöründe `best_cnn.keras`, `label_map.json`, `history.csv` dosyaları oluşur.

## Eğitim ve Değerlendirme
Notebook sonunda otomatik olarak şu çıktılar üretilir:
- **Accuracy/Loss** grafiklerinin çizimi (epoch bazında)
- **Confusion Matrix** ve **Classification Report** (test setinde)
- En iyi ağırlıkların diske kaydı: `best_cnn.keras`

Tek görsel üzerinde hızlı tahmin için:
```python
# notebook içinde:
predict_image("<gorsel_yolu>.jpg", top_k=5)
