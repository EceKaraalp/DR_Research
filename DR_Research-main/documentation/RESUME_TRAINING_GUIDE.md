# Resume Training Guide - Kesintide Eğitimi Devam Ettirme

## 📋 Özet

Yeni notebook'ta implementedilen **Checkpoint Resume** mekanizması sayesinde eğitim kesintiye uğrarsa kaydedilen noktadan devam edebilirsiniz.

---

## 🔄 Nasıl Çalışır?

### Checkpoint Kaydetme

```
Her 5 epoch'ta:
├─ Model weights (state_dict)
├─ Optimizer state (learning rate, momentum)
├─ Scheduler state (epoch tracking)
├─ Training history (loss, accuracy, etc.)
├─ Best metric & patience counter
└─ Dosya: checkpoint_fold{N}.pth
```

### Resume Mekanizması

```
Eğitim başlatıldığında:
├─ Checkpoint var mı? → Evet: Yükle + Devam et
├─ Checkpoint yok → Sıfırdan başla
└─ Başarıyla biterse → Checkpoint otomatik sil
```

---

## 💾 Kullanım

### **SEÇENEK 1: Otomatik Resume (Tavsiye Edilen)**

Cell 14'de eğitimi her çalıştırdığınızda, **varsa checkpoint'ten otomatik devam eder**:

```python
# Cell 14 başında:
RESUME_TRAINING = False  # ← Check point varsa otomatik devam eder

# Eğitimi çalıştır:
trainer.fit(resume=True)  # ✅ Otomatik resume

# Fold 1 - Checkpoint bulundu → Devam etti
# Fold 2 - Checkpoint yok → Sıfırdan başladı
# Fold 3 - Best model var → Atla
```

### **SEÇENEK 2: Manuel Resume Kontrolü**

Cell 14'de açıkça kullanmak istiyorsanız:

```python
RESUME_TRAINING = True  # ← Tüm foldlar için resume enable

# Sonra eğitimi çalıştır:
trainer.fit(resume=RESUME_TRAINING)  # ✅ Tüm checkpoint'lerden devam et
```

---

## 📁 Checkpoint Dosyaları

```
results_holdout_evaluation/models/
├── best_model_fold0.pth          ← Final trained model
├── checkpoint_fold0.pth          ← Resume checkpoint (eğitim sırasında kaydedilir)
├── checkpoint_history_fold0.json ← Training history (backup)
├── best_model_fold1.pth
├── checkpoint_fold1.pth
└── ...
```

### Checkpoint İçeriği

```python
checkpoint = {
    'epoch': 25,                    # Kaydedildiği epoch
    'model_state': {...},           # Model weights
    'optimizer_state': {...},       # Optimizer (lr, momentum, etc.)
    'scheduler_state': {...},       # Learning rate scheduler state
    'best_metric': 0.7234,          # Best validation F1 so far
    'patience_counter': 3,          # Early stopping counter
    'history': {
        'train_loss': [0.5, 0.45, ...],
        'val_loss': [0.6, 0.55, ...],
        'val_f1': [0.65, 0.68, ...],
        ...
    }
}
```

---

## 🛑 Kesinti Senaryoları

### Senaryo 1: Eğitim Ortasında Kesinti

```
Fold 1 - Epoch 25'te kesinti
│
├─ 5 dakika sonra eğitim yeniden başlatılır
│
└─ ✅ Checkpoint yüklenir
   ├─ Model weights resume edilir
   ├─ Optimizer state restore edilir
   ├─ Training history korunur
   └─ Epoch 26'dan devam edilir
```

### Senaryo 2: Bir Fold Tamamlandı, Diğeri Yapılırken Kesinti

```
Fold 1 - ✅ Tamamlandı (best_model_fold0.pth kaydedildi)
Fold 2 - Epoch 40'ta kesinti → checkpoint_fold1.pth kaydedildi
Fold 3 - Başlanmadı
│
├─ Eğitim yeniden başlatılır
│
└─ ✅ Akıllı resume:
   ├─ Fold 1: Best model var → Atla, validation yap
   ├─ Fold 2: Checkpoint var → Epoch 41'den devam et
   └─ Fold 3: Reset → Sıfırdan başla
```

### Senaryo 3: Final Sonuç Kaydedildi

```
5 Fold Training - ✅ Tamamlandı
│
├─ Checkpoint dosyaları otomatik silindi
│
└─ ✅ Sadece best models kaldı:
   ├─ best_model_fold0.pth ✓
   ├─ best_model_fold1.pth ✓
   ├─ best_model_fold2.pth ✓
   ├─ best_model_fold3.pth ✓
   └─ best_model_fold4.pth ✓
```

---

## ⚡ Hızlı Rehber

### Eğitimi Başlat

```python
# Cell 14 çalıştır:
# - Otomatik checkpoint'ten devam edecek veya yeni başlayacak
# - Her fold için best model kaydedilecek
```

### Kesintide Ne Yapmalı

```
✅ Hiçbir şey yapmayın! Notebook'u tekrar açın ve Cell 14'ü çalıştırın
✅ Otomatik resume active olacak
❌ Eski checkpoint dosyalarını silmeyin!
```

### Tamamlanmış Eğitimi Kontrol Et

```python
# Checkpoint'ler silindi mi?
import os
checkpoint_files = [f for f in os.listdir('results_holdout_evaluation/models/') 
                   if 'checkpoint' in f]

if checkpoint_files:
    print("⚠️ Eğitim devam ediyor (checkpoint bulundu)")
else:
    print("✅ Eğitim tamamlandı")
```

### Tamamlanmış Eğitimi Skip Et

Cell 14'de best models var ise otomatik skip edilir:

```python
# Automatic skip logic:
for fold_idx in range(5):
    best_model = f"best_model_fold{fold_idx}.pth"
    if os.path.exists(best_model):
        print(f"✅ Fold {fold_idx + 1} already done - skipping")
        continue
    # Else: train fold
```

---

## 📊 GPU Memory & Checkpoint Size

| Dosya | Boyut |
|-------|-------|
| best_model_fold{N}.pth | ~200 MB |
| checkpoint_fold{N}.pth | ~200 MB |
| checkpoint_history_fold{N}.json | <1 MB |

**Total for 5 folds:** ~2 GB

---

## ⚠️ Dikkat Edilecek Noktalar

### ✅ Yapılması Gerekenler

- ✅ Checkpoint dosyalarını saklamak
- ✅ Eğitim sırasında kesintide sakin kalı
- ✅ Aynı config ile resume etmek
- ✅ Batch size değişmekkse checkpoint'i sil

### ❌ Yapılmaması Gerekenler

- ❌ Checkpoint dosyalarını elle düzenlemek
- ❌ Eğitim sırasında model dosyaları silmek
- ❌ Farklı GPU ile resume etmek (compatible olmayabilir)
- ❌ Config değiştirip resume etmek

---

## 🔧 Manual Resume (Advanced)

Eğer manuel olarak temel almak istiyorsanız:

```python
# Checkpoint yükle
checkpoint = torch.load('results_holdout_evaluation/models/checkpoint_fold0.pth')

# Training history al
history = checkpoint['history']
start_epoch = checkpoint['epoch'] + 1
best_f1 = checkpoint['best_metric']

print(f"Resuming from epoch {start_epoch}")
print(f"Best F1 so far: {best_f1:.4f}")
print(f"Epochs trained: {len(history['train_loss'])}")
```

---

## 📞 Troubleshooting

### Problem: "Checkpoint not found"
```
Sorun: Checkpoint dosyası silinmiş
Çözüm: Eğitim sıfırdan başlayacak (yeni checkpoint oluşturulacak)
```

### Problem: "Failed to load checkpoint"
```
Sorun: Checkpoint dosyası corrupt
Çözüm:
  1. checkpoint_fold{N}.pth dosyasını sil
  2. Eğitimi yeniden başlat
  3. Best model varsa ondan devam et
```

### Problem: "Out of Memory"
```
Sorun: CUDA OOM hatasıkesintiye neden olmuş
Çözüm:
  1. Batch size azalt: config.BATCH_SIZE = 8
  2. Checkpoint devam edecek (weights kurtarılacak)
  3. Eğitimi yeniden başlat
```

---

## 📈 Örnek Çıkış

```
═══════════════════════════════════════════════════
FOLD 1/5 TRAINING
═══════════════════════════════════════════════════

Loading checkpoint...
✅ Resumed from checkpoint (epoch 26)
   Best F1 so far: 0.7234
   Patience counter: 3/15

Continuing training...

E27 | TL=0.3245 VL=0.3612 | TA=0.8456 VA=0.8123 | F1=0.7290 | QWK=0.8845
E30 | TL=0.3120 VL=0.3578 | TA=0.8512 VA=0.8156 | F1=0.7312 | QWK=0.8867
E40 | TL=0.2890 VL=0.3501 | TA=0.8634 VA=0.8234 | F1=0.7401 | QWK=0.8934
...checkpoint saved...
E60 | TL=0.2456 VL=0.3445 | TA=0.8723 VA=0.8301 | F1=0.7534 | QWK=0.8967

⚠ Early stopping at epoch 68

✓ Fold 1 done (best F1: 0.7534)
```

---

## 📚 Kaynaklar

| Dosya | Konu |
|-------|------|
| [Trainer sınıfı](../dr_advanced_holdout_evaluation.ipynb#VSC-5e67bf04) | Checkpoint mekanizması implementasyonu |
| [Training cell](../dr_advanced_holdout_evaluation.ipynb#VSC-012f25b9) | Resume şunu nasıl kullanacağını |
| [HOLDOUT_TEST_EVALUATION_README.md](../documentation/HOLDOUT_TEST_EVALUATION_README.md) | Genel eğitim protokolü |

---

## 🎯 Özet

| Madde | Açıklama |
|------|----------|
| **Resume** | ✅ Otomatik, manuel kontrol mümkün |
| **Checkpoint**| ✅ Her 5 epoch'ta kaydedilir |
| **Dosyalar** | ✅ models/ klasöründe saklanır |
| **Memory** | ✅ ~200 MB per fold |
| **Güvenlik** | ✅ Training history backup edilir |
| **Silme** | ✅ Eğitim bitince otomatik silinir |

**Status:** ✨ Production Ready

---

**Last Updated:** March 3, 2026  
**Version:** 1.0
