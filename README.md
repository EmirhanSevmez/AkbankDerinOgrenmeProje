# Car Brand Classification — CNN (Akbank Derin Öğrenme Bootcamp)

Bu repo, **Car Brand Classification (33 sınıf)** problemi için hazırlanan derin öğrenme projesinin kod ve dokümantasyonunu içerir. Proje; veri önişleme, veri çoğaltma (augmentation), **CNN** tabanlı model eğitimi, değerlendirme (accuracy/loss grafikleri, **Confusion Matrix** & **Classification Report**) ve **Grad-CAM** görselleştirmelerini kapsar. Opsiyonel olarak **Transfer Learning** ve küçük bir **HPO** (hyperparameter optimization) denemesi de eklenebilir.


# Giriş

- **Amaç:** Araç marka görsellerini (Audi, BMW, Ford, Mercedes, Tesla vb.) çok sınıflı bir CNN ile sınıflandırmak.
- **Veri seti:** Kaggle — `ahmedelsany/car-brand-classification-dataset` (yaklaşık 16K görsel, 33 sınıf).
- **Önişleme:** 224×224 yeniden boyutlandırma, `[0,1]` normalize; **train/val/test = 80/10/10 (stratified)**.
- **Augmentation:** RandomFlip, RandomRotation, RandomZoom, RandomContrast.
- **Model (Baseline CNN):** Conv2D + MaxPooling blokları, GlobalAveragePooling2D, Dropout(0.4), Dense(softmax).
- **Callback’ler:** ModelCheckpoint (en iyi `val_accuracy`), ReduceLROnPlateau, EarlyStopping.
- **Opsiyonel:** EfficientNetB0 / ResNet50 ile Transfer Learning, küçük HPO ızgarası (lr, dropout, batch).


# Metrikler

- Eğitim sonrası **Accuracy/Loss** grafikleri
- Test setinde **Confusion Matrix** ve **Classification Report**
- Örnek tek görüntü tahmini: `predict_image("<path>.jpg", top_k=5)`

# Ekler

- (Opsiyonel) **UI/**: Streamlit ile basit bir arayüz ve demo (sınıf tahmini + Grad-CAM görselleri).  

# Sonuç ve Gelecek Çalışmalar

- Sonuçların özeti (en iyi doğruluk, hata analizi).
- Gelecek adımlar:
  - Daha güçlü backbone ile **fine-tuning** (EfficientNet/ResNet).
  - **Keras Tuner** ile kapsamlı HPO.
  - Veri dengeleme/temizleme, domain-spesifik augmentasyon.
  - (Opsiyonel) Streamlit/Gradio ile web demo, Docker paketleme.

# Linkler

- **Kaggle Notebook:** _linkinizi buraya ekleyin_  
- (İsteğe bağlı) Ek deney notları / raporlar

---

## Nasıl Çalıştırılır (Kaggle)

1. **Settings → Accelerator → GPU** seçin.  
2. **Add Data** → `ahmedelsany/car-brand-classification-dataset` ekleyin.  
3. Notebook’u **Restart & Run All** ile çalıştırın.  
4. Çıktılar:
   - `best_cnn.keras` — en iyi model ağırlıkları  
   - `label_map.json` — sınıf indeks-isim eşlemeleri  
   - `history.csv` — epoch metrikleri (acc/loss)


