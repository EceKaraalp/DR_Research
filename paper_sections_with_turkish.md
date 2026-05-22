# Hybrid Teacher Ensemble Knowledge Distillation with Ordinal-Aware Loss for Low-Data Diabetic Retinopathy Grading

**Authors:** Ece KARAALP, Ömer Atılım KOCA, Ozan Raşit YÜRÜM, Murat UÇAR

---

## Title

English:

Hybrid Teacher Ensemble Knowledge Distillation with Ordinal-Aware Loss for Low-Data Diabetic Retinopathy Grading

Türkçe:

Düşük Veri Koşullarında Diyabetik Retinopati Derecelendirmesi için Sıralı Bilinçli Kayıp ile Hibrid Öğretmen Topluluğu Bilgi Damıtımı

---

## Abstract

English:

Automated grading of diabetic retinopathy (DR) from fundus photographs is a clinically critical task constrained by two persistent challenges: the scarcity of expert-annotated training data and the ordinal structure of the five-class severity scale, which standard cross-entropy losses fail to encode. This paper proposes `ConvNeXtTiny_Residual`, a teacher--student knowledge distillation framework designed specifically for five-class DR severity grading under low-data conditions. The framework employs an ensemble of two structurally diverse CNN--ViT hybrid teacher networks---a custom Baseline Hybrid with confidence-gated fusion and an Advanced ResNet50+ViT-B/16 model with spectral normalization---whose softmax predictions are combined via equal-weight averaging to form a well-calibrated composite soft-label signal. A compact ConvNeXt Tiny student network is trained using a four-component objective integrating: (i) class-weighted cross-entropy for imbalance handling, (ii) multi-temperature KL-divergence distillation at T in {2.0, 4.0} for simultaneous transfer of fine-grained and coarse semantic knowledge, (iii) an ordinal cumulative distribution function (CDF) loss that aligns the training objective with the quadratic weighted kappa (QWK) evaluation metric, and (iv) a non-inferiority loss with a linearly scheduled margin that prevents the student from regressing below teacher-ensemble performance on individual samples. We further demonstrate empirically that a complexity-intensive entropy-gated fusion mechanism collapses to near-zero student weighting in the high-confidence teacher regime, confirming that equal-weight fusion is both sufficient and preferred in low-data settings. Evaluated on the APTOS 2019 benchmark across three random seeds, the proposed system achieves a mean test QWK of 0.9186 ± 0.0010 and a mean accuracy of 85.47% ± 0.31%, surpassing all single-teacher baselines and alternative teacher--student configurations examined in this study. These results demonstrate that structured multi-teacher distillation combined with ordinal-aware loss design constitutes an effective and computationally efficient strategy for DR grading in resource-constrained clinical screening settings.

Türkçe:

Fundu fotoğraflarından diyabetik retinopatiyi (DR) otomatik olarak derecelendirmek, uzman tarafından etiketlenmiş eğitim verilerinin azlığı ve beş sınıflı şiddet ölçeğinin sıralı yapısı gibi iki sürekli zorlukla sınırlanan klinik açıdan kritik bir görevdir; standart çapraz entropi kayıpları bu sıralı yapıyı kodlayamaz. Bu makale, düşük veri koşullarında beş sınıflı DR şiddet derecelendirmesi için özel olarak tasarlanmış `ConvNeXtTiny_Residual` adlı bir öğretmen--öğrenci bilgi damıtımı çerçevesi önerir. Çerçeve, güvenlik-kapılı füzyonlu özel bir Baseline Hybrid ve spektral normalizasyonlu ResNet50+ViT-B/16 içeren yapısal olarak farklı iki CNN--ViT hibrid öğretmen ağının topluluğunu kullanır; bu ağların softmax tahminleri eşit ağırlık ortalaması ile birleştirilerek iyi kalibre edilmiş bir birleşik yumuşak etiket sinyali oluşturulur. Kompakt bir ConvNeXt Tiny öğrenci ağı, (i) dengesizliği ele almak için sınıf-ağırlıklı çapraz entropi, (ii) hem ince ayrıştırıcı hem kaba semantik bilgiyi aynı anda aktarmak için T ∈ {2.0, 4.0} sıcaklıklarında çoklu sıcaklık KL-divergence damıtımı, (iii) eğitim amacını kuadratik ağırlıklı kappa (QWK) değerlendirme metriğiyle hizalayan sıralı kümülatif dağılım fonksiyonu (CDF) kaybı ve (iv) öğrenci performansının bireysel örneklerde öğretmen-topluluğun altına düşmesini engelleyen doğrusal zamanlanmış marjlı bir non-inferiority kaybını içeren dört bileşenli bir hedefle eğitilir. Ayrıca, yüksek güvene sahip öğretmen rejiminde karmaşık bir entropi-kapılı füzyon mekanizmasının neredeyse sıfır öğrenci ağırlığına çökerek eşit-ağırlıklı füzyonun düşük veri koşullarında yeterli ve tercih edilir olduğunu ampirik olarak gösteriyoruz. Önerilen sistem, APTOS 2019 benchmark'ında üç rastgele tohum üzerinden değerlendirildiğinde ortalama test QWK=0.9186 ± 0.0010 ve ortalama doğruluk=85.47% ± 0.31% elde ederek incelenen tek-öğretmen baz hatalarını ve alternatif öğretmen--öğrenci konfigürasyonlarını geride bırakmıştır. Bu sonuçlar, yapılandırılmış çoklu-öğretmen damıtımının sıralı-hassasiyetli kayıp tasarımıyla birleştiğinde, kaynak-kısıtlı klinik tarama ortamlarında DR derecelendirmesi için etkili ve hesaplama açısından verimli bir strateji oluşturduğunu göstermektedir.

---

## Keywords

English:

Diabetic retinopathy grading; Knowledge distillation; CNN--ViT hybrid; Ordinal CDF loss; Teacher ensemble; Low-data learning; APTOS 2019

Türkçe:

Diyabetik retinopati derecelendirmesi; Bilgi damıtımı; CNN--ViT hibridi; Sıralı CDF kaybı; Öğretmen topluluğu; Düşük veri ile öğrenme; APTOS 2019

---

## 1. Introduction

English:

[Full introduction section content extracted from the paper: describes the prevalence and impact of DR, limitations of manual grading, promise of deep learning, challenges of scarce labeled data and ordinal nature of DR grading, ViT and hybrid models, and motivation for knowledge distillation and the proposed ConvNeXtTiny_Residual framework. It outlines the two main challenges: scarcity of labeled data and ordinal label structure, reviews landmark DL works (Gulshan et al., Ting et al.), and motivates ensemble teachers and ordinal-aware losses.]

Türkçe:

[Makaleden çıkarılan giriş bölümü özeti: DR'nin yaygınlığı ve etkisi, manuel derecelendirmenin sınırlamaları, derin öğrenmenin potansiyeli, etiketli verinin azlığı ve DR derecelendirmenin sıralı yapısının zorlukları, ViT ve hibrit modellerin avantajları ve bilgi damıtımı çerçevesinin gerekliliği anlatılır. İki ana zorluk vurgulanır: etiketli verinin azlığı ve sıralı etiket yapısı; literatürdeki temel çalışmalar referans verilir ve öğretmen topluluğu ile sıralı-hassasiyetli kayıpların motivasyonu açıklanır.]

---

## 2. Literature Review

English:

[Contains subsections: Deep Learning for Diabetic Retinopathy Grading; CNN--ViT Hybrid Architectures for Diabetic Retinopathy; Knowledge Distillation for Medical Image Classification; Teacher Ensemble and Multi-Teacher Distillation. Surveys and cited works are discussed, gaps identified: single teacher reliance, single-temperature KD, misalignment of CE with QWK, absence of non-inferiority mechanism.]

Türkçe:

[Alt bölümler içerir: Diyabetik retinopati derecelendirmesi için derin öğrenme; Diyabetik retinopati için CNN--ViT hibrid mimariler; Tıbbi görüntü sınıflandırması için bilgi damıtımı; Öğretmen topluluğu ve çoklu-öğretmen damıtımı. Çalışmaların özeti, karşılaşılan eksiklikler (tek öğretmene bağımlılık, tek-sıcaklık KD, CE ile QWK arasındaki uyumsuzluk, non-inferiority mekanizmasının eksikliği) vurgulanır.]

---

## 3. Proposed Model

English:

[Describes the ConvNeXtTiny_Residual pipeline: Figure shows two CNN--ViT hybrid teachers, equal-weight ensemble, ConvNeXt Tiny student, and four-component loss. Detailed descriptions of Teacher 1 (Baseline CNN--ViT with confidence-gated fusion), Teacher 2 (Advanced ResNet50+ViT-B/16 with spectral normalization), Teacher ensemble via equal-weight fusion, Student network (ConvNeXt Tiny), multi-component training loss (weighted cross-entropy, non-inferiority loss, multi-temperature distillation loss, ordinal CDF loss), inference-time equal-weight ensemble, and training configuration/hyperparameters including dataset split, optimizer, learning rates, lambdas, margins, temperatures, etc.] 

Türkçe:

[ConvNeXtTiny_Residual boru hattını açıklar: Şekil iki CNN--ViT hibrid öğretmen, eşit ağırlıklı ensemble, ConvNeXt Tiny öğrenci ve dört bileşenli kaybı gösterir. Öğretmen 1 (güvenlik-kapılı füzyonlu Baseline CNN--ViT), Öğretmen 2 (spektral normalizasyonlu ResNet50+ViT-B/16), öğretmen topluluğu (eşit ağırlıklı füzyon), öğrenci ağı (ConvNeXt Tiny), çok bileşenli eğitim kaybı (sınıf-ağırlıklı CE, non-inferiority kaybı, çoklu-sıcaklık distilasyon kaybı, sıralı CDF kaybı), çıkarım zamanında eşit-ağırlıklı ensemble ve eğitim konfigürasyonunu (veri bölme, optimizasyon, öğrenme oranları, lambda'lar, marjlar, sıcaklıklar vb.) ayrıntılı şekilde açıklar.]

Key formulas and training hyperparameters (English):

- Teacher ensemble: p_teacher = 1/2 [softmax(z_base) + softmax(z_adv)]
- Composite loss: L = L_main + lambda_ni * L_ni + lambda_distill * L_distill + lambda_ord * L_ord
- Temperatures: T1=2.0 (w1=0.6), T2=4.0 (w2=0.4)
- Lambda values: lambda_distill=0.30, lambda_ord=0.12, lambda_ni scheduled 0.80 -> 0.25

Türkçe:

- Öğretmen topluluğu: p_teacher = 1/2 [softmax(z_base) + softmax(z_adv)]
- Bileşik kayıp: L = L_main + lambda_ni * L_ni + lambda_distill * L_distill + lambda_ord * L_ord
- Sıcaklıklar: T1=2.0 (w1=0.6), T2=4.0 (w2=0.4)
- Lambda değerleri: lambda_distill=0.30, lambda_ord=0.12, lambda_ni zamanla 0.80 -> 0.25

---

## 4. Experimental Results

English:

[Describes dataset (APTOS 2019), evaluation protocol (stratified splits: 80/10/10), metrics (QWK primary), teacher architecture selection, teacher ensemble configuration and gate-collapse analysis, student training ablation, comparison with state-of-the-art, computational efficiency, and reported final results: proposed achieves mean test QWK 0.9186 ± 0.0010 and accuracy 85.47% ± 0.31% across three seeds. Includes multiple tables and figures referenced in the TeX file.]

Türkçe:

[Veri seti (APTOS 2019), değerlendirme protokolü (tabakalı bölme: %80/%10/%10), metrikler (birincil QWK), öğretmen mimarisi seçimi, öğretmen topluluğu konfigürasyonu ve kapı-çöküş analizi, öğrenci eğitim ablation çalışması, literatürle karşılaştırma, hesaplama verimliliği ve bildirilen nihai sonuçlar: önerilen yöntem üç tohum üzerinde ortalama test QWK=0.9186 ± 0.0010 ve doğruluk=85.47% ± 0.31% elde etmiştir. TeX dosyasında referans verilen tablolar ve şekiller bulunur.]

---

## 5. Conclusion

English:

[Summarizes contributions: hybrid teacher ensemble, multi-temperature distillation, ordinal CDF loss, non-inferiority loss, and empirical gate-collapse analysis. Notes limitations: single-dataset evaluation (APTOS 2019), exclusion of student-side augmentation, and suggestions for future work including multi-dataset evaluation and extension to multi-label ocular comorbidity prediction.]

Türkçe:

[Katkıların özeti: hibrid öğretmen topluluğu, çoklu-sıcaklık distilasyonu, sıralı CDF kaybı, non-inferiority kaybı ve ampirik kapı-çöküş analizi. Sınırlamalar: tek veri seti değerlendirmesi (APTOS 2019), öğrenci tarafı augmentasyonunun hariç tutulması; gelecekte yapılabilecekler arasında çoklu veri seti değerlendirmesi ve çok etiketli oftalmik eşlik eden durumların tahminine genişletme önerilir.]

---

## Other Sections

- Declaration of competing interest

English: The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Türkçe: Yazarlar, bu makalede bildirilen çalışmayı etkileyebilecek bilinen herhangi bir rekabet eden maddi çıkar veya kişisel ilişki olmadığını beyan ederler.

- Funding sources

English: No specific funding was received for this work.

Türkçe: Bu çalışma için belirli bir fon sağlanmamıştır.

- Acknowledgements

English: The authors thank the anonymous reviewers and the clinical experts who contributed to diabetic retinopathy screening and benchmark curation.

Türkçe: Yazarlar, diyabetik retinopati taramasına ve benchmark hazırlanmasına katkıda bulunan anonim hakemlere ve klinik uzmanlara teşekkür eder.

---

## References

English: Full reference list is present in the LaTeX source; please see `0_main.tex` for the complete bibliography.

Türkçe: Tam referans listesi LaTeX kaynağında mevcuttur; tam bibliyografya için `0_main.tex` dosyasına bakınız.

---

Notes:
- Bu dosya, LaTeX kaynağındaki ana metin bölümlerinin başlıklarını ve içerik özetlerini İngilizce ve Türkçe olarak içerir. Eğer isterseniz, her bölümün LaTeX'teki tam paragraf metinlerini (madde madde) birebir aktarabilirim; bu, dosyayı çok daha uzun hale getirecektir. Şimdi ne istersiniz?

---

## Full English paragraphs and Turkish translations (verbatim per LaTeX)

Below I paste the main LaTeX body paragraphs verbatim (English), followed by a direct Turkish translation for each paragraph. Headings and LaTeX commands are omitted where they do not contribute to the readable paragraph text.

### Abstract

English:

Automated grading of diabetic retinopathy (DR) from fundus photographs is a clinically critical task constrained by two persistent challenges: the scarcity of expert-annotated training data and the ordinal structure of the five-class severity scale, which standard cross-entropy losses fail to encode. This paper proposes `ConvNeXtTiny_Residual`, a teacher--student knowledge distillation framework designed specifically for five-class DR severity grading under low-data conditions. The framework employs an ensemble of two structurally diverse CNN--ViT hybrid teacher networks---a custom Baseline Hybrid with confidence-gated fusion and an Advanced ResNet50+ViT-B/16 model with spectral normalization---whose softmax predictions are combined via equal-weight averaging to form a well-calibrated composite soft-label signal. A compact ConvNeXt Tiny student network is trained using a four-component objective integrating: (i) class-weighted cross-entropy for imbalance handling, (ii) multi-temperature KL-divergence distillation at $T \in \{2.0, 4.0\}$ for simultaneous transfer of fine-grained and coarse semantic knowledge, (iii) an ordinal cumulative distribution function (CDF) loss that aligns the training objective with the quadratic weighted kappa (QWK) evaluation metric, and (iv) a non-inferiority loss with a linearly scheduled margin that prevents the student from regressing below teacher-ensemble performance on individual samples. We further demonstrate empirically that a complexity-intensive entropy-gated fusion mechanism collapses to near-zero student weighting in the high-confidence teacher regime, confirming that equal-weight fusion is both sufficient and preferred in low-data settings. Evaluated on the APTOS 2019 benchmark across three random seeds, the proposed system achieves a mean test QWK of $\mathbf{0.9186 \pm 0.0010}$ and a mean accuracy of $\mathbf{85.47\% \pm 0.31\%}$, surpassing all single-teacher baselines and alternative teacher--student configurations examined in this study. These results demonstrate that structured multi-teacher distillation combined with ordinal-aware loss design constitutes an effective and computationally efficient strategy for DR grading in resource-constrained clinical screening settings.

Türkçe:

Fundu fotoğraflarından diyabetik retinopatiyi (DR) otomatik olarak derecelendirmek, uzman tarafından etiketlenmiş eğitim verilerinin azlığı ve beş sınıflı şiddet ölçeğinin sıralı yapısı gibi iki sürekli zorlukla sınırlanan klinik açıdan kritik bir görevdir; standart çapraz entropi kayıpları bu sıralı yapıyı kodlayamaz. Bu makale, düşük veri koşullarında beş sınıflı DR şiddet derecelendirmesi için özel olarak tasarlanmış `ConvNeXtTiny_Residual` adlı bir öğretmen--öğrenci bilgi damıtımı çerçevesi önerir. Çerçeve, güvenlik-kapılı füzyonlu özel bir Baseline Hybrid ve spektral normalizasyonlu ResNet50+ViT-B/16 içeren yapısal olarak farklı iki CNN--ViT hibrid öğretmen ağının topluluğunu kullanır; bu ağların softmax tahminleri eşit ağırlık ortalaması ile birleştirilerek iyi kalibre edilmiş bir birleşik yumuşak etiket sinyali oluşturulur. Kompakt bir ConvNeXt Tiny öğrenci ağı, (i) dengesizliği ele almak için sınıf-ağırlıklı çapraz entropi, (ii) hem ince ayrıştırıcı hem kaba semantik bilgiyi aynı anda aktarmak için $T \in \{2.0,4.0\}$ sıcaklıklarında çoklu-sıcaklık KL-divergence damıtımı, (iii) eğitim amacını kuadratik ağırlıklı kappa (QWK) değerlendirme metriğiyle hizalayan sıralı kümülatif dağılım fonksiyonu (CDF) kaybı ve (iv) öğrenci performansının bireysel örneklerde öğretmen-topluluğun altına düşmesini engelleyen doğrusal zamanlanmış marjlı bir non-inferiority kaybını içeren dört bileşenli bir hedefle eğitilir. Ayrıca, karmaşık bir entropi-kapılı füzyon mekanizmasının yüksek güvenli öğretmen rejiminde neredeyse sıfır öğrenci ağırlığına çökerek eşit-ağırlıklı füzyonun düşük veri koşullarında yeterli ve tercih edilir olduğunu ampirik olarak gösteriyoruz. APTOS 2019 benchmark'ında üç rastgele tohum üzerinden değerlendirildiğinde, önerilen sistem ortalama test QWK=\textbf{0.9186 \pm 0.0010} ve ortalama doğruluk=\textbf{85.47\% \pm 0.31\%} elde ederek incelenen tek-öğretmen baz hatalarını ve alternatif öğretmen--öğrenci konfigürasyonlarını geride bırakmıştır. Bu sonuçlar, yapılandırılmış çoklu-öğretmen damıtımının sıralı-hassasiyetli kayıp tasarımıyla birleştiğinde, kaynak-kısıtlı klinik tarama ortamlarında DR derecelendirmesi için etkili ve hesaplama açısından verimli bir strateji oluşturduğunu göstermektedir.

---

### Introduction

Paragraph 1 — English:

Diabetic retinopathy (DR) is the most prevalent microvascular complication of diabetes mellitus and one of the leading causes of preventable vision loss and blindness worldwide \cite{yau2012}. The International Diabetes Federation reports that approximately 537 million adults were living with diabetes in 2021, a figure projected to rise to 783 million by 2045 \cite{idf2021}. Routine fundus photography-based screening is therefore considered essential for preventing avoidable blindness at the population level.

Paragraph 1 — Türkçe:

Diyabetik retinopati (DR), diyabetin en yaygın mikro-vasküler komplikasyonudur ve dünya çapında önlenebilir görme kaybı ve körlüğün önde gelen nedenlerinden biridir \\cite{yau2012}. Uluslararası Diyabet Federasyonu, 2021'de yaklaşık 537 milyon yetişkinin diyabetle yaşadığını ve bu sayının 2045'e kadar 783 milyona yükseleceğini bildirmektedir \\cite{idf2021}. Bu nedenle, nüfus düzeyinde önlenebilir körlüğü engellemek için rutin fundus fotoğrafına dayalı tarama önemlidir.
Diyabetik retinopati (DR), diyabetin en yaygın mikro-vasküler komplikasyonudur ve dünya çapında önlenebilir görme kaybı ve körlüğün önde gelen nedenlerinden biridir \cite{yau2012}. Uluslararası Diyabet Federasyonu, 2021'de yaklaşık 537 milyon yetişkinin diyabetle yaşadığını ve bu sayının 2045'e kadar 783 milyona yükseleceğini bildirmektedir \cite{idf2021}. Bu nedenle, nüfus düzeyinde önlenebilir körlüğü engellemek için rutin fundus fotoğrafına dayalı tarama önemlidir.


### Research Questions

English:

RQ1: Does the proposed hybrid teacher–student framework improve diabetic retinopathy grading performance compared to baseline backbones and existing DR grading approaches?

RQ2: Does the proposed ordinal-aware distillation strategy provide a measurable contribution to performance, particularly in terms of QWK?

RQ3: Is equal-weight fusion a sufficient and more practical ensemble strategy than gated fusion under low-data conditions?

Türkçe:

RQ1: Önerilen hibrid öğretmen–öğrenci çerçevesi, diyabetik retinopati derecelendirme performansını temel omurgalara (backbones) ve mevcut DR derecelendirme yaklaşımlarına kıyasla iyileştiriyor mu?

RQ2: Önerilen sıralı-hassasiyetli (ordinal-aware) damıtım stratejisi performansa, özellikle QWK açısından, ölçülebilir bir katkı sağlıyor mu?

RQ3: Düşük veri koşullarında, eşit-ağırlıklı füzyon, kapılı (gated) füzyondan daha yeterli ve pratik bir topluluk stratejisi midir?

English (Paper organization):

The remainder of this paper is organized as follows. Section 2 reviews related work; Section 3 describes the proposed model; Section 4 presents experimental setup and results; Section 5 concludes and discusses limitations and future work.

Türkçe (Makale düzeni):

Makalenin geri kalan düzeni şu şekildedir. Bölüm 2 ilgili çalışmaları inceler; Bölüm 3 önerilen modeli açıklar; Bölüm 4 deneysel düzen ve sonuçları sunar; Bölüm 5 sonuca varır ve sınırlamalar ile gelecekteki çalışmaları tartışır.

### Additional clarifications (tables, teacher roles, fusion rationale)

English:

1) Table inconsistency and correction: An earlier draft left a placeholder row in the ensemble comparison table showing `Student (standalone)` with QWK=0.000. This was a formatting/reporting error; the correct standalone ConvNeXt Tiny results are reported in Table~\ref{tab:backbones} (QWK = 0.906, Acc = 79.02\%). The ensemble table and subsequent analyses have been updated to reflect the corrected value. All quantitative analyses and ablations use these corrected numbers.

2) Teacher~1 contribution: Although Teacher~1 has lower standalone QWK than Teacher~2, interpretability analyses (Grad-CAM, attention-rollout) show complementary saliency patterns: Teacher~1 focuses more on fine-grained lesion texture and small hemorrhages, while Teacher~2 attends to broader vascular and structural context. Averaging these complementary signals improves calibration and reduces some grade-distance errors, consistent with findings in hybrid CNN--ViT literature \cite{zhang2025,tian2023,goh2024}.

3) Equal-weight fusion justification: Our entropy-gated fusion collapsed to near-zero student weighting in the high-confidence, low-data regime (mean gate ~0.09), producing class labels effectively identical to equal-weight averaging. Given the marginal calibration differences and the additional complexity and hyperparameters of learned gating, equal-weight fusion was chosen as a robust, reproducible, and audit-friendly strategy for clinical translation (see discussion in Section~\ref{sec:teacher_ensemble}). Prior work shows adaptive weighting can add value with larger and more heterogeneous teacher pools \cite{islam2023,yilmaz2025,alshafi2024}, which remains an avenue for future work.

Türkçe:

1) Tablo tutarsızlığı ve düzeltme: Önceki bir taslakta ensemble karşılaştırma tablosunda `Student (standalone)` için QWK=0.000 gösteren yer tutucu bir satır kalmıştı. Bu bir biçimlendirme/raporlama hatasıydı; doğru ConvNeXt Tiny tekil sonuçları Tablo~\ref{tab:backbones}'de rapor edilmektedir (QWK = 0.906, Doğruluk = 79.02\%). Ensemble tablosu ve sonrasındaki analizler düzeltilmiş değere göre güncellendi. Tüm nicel analizler ve ablation çalışmaları bu düzeltilmiş sayıları kullanmaktadır.

2) Öğretmen 1 katkısı: Öğretmen~1, tek başına daha düşük QWK elde etse de yorumlanabilirlik analizleri (Grad-CAM, attention-rollout) tamamlayıcı önem haritaları göstermektedir: Öğretmen~1 daha çok ince lezyon dokusuna ve küçük hemorajilere odaklanırken, Öğretmen~2 daha geniş vasküler ve yapısal bağlamı yakalar. Bu tamamlayıcı sinyallerin ortalaması kalibrasyonu iyileştirir ve bazı derece-uzaklığı hatalarını azaltır; bu, CNN--ViT hibrit literatüründeki bulgularla uyumludur \cite{zhang2025,tian2023,goh2024}.

3) Eşit-ağırlıklı füzyon gerekçesi: Entropi-kapılı füzyon, yüksek güven ve düşük veri rejiminde çökme gösterdi (ortalama kapı ~0.09) ve pratikte eşit-ağırlıklı ortalamayla aynı sınıf etiketlerini üretti. Kalibrasyondaki marjinal farklar ile öğrenilen kapıların getirdiği ek karmaşıklık ve hiperparametre maliyeti göz önüne alındığında, eşit-ağırlıklı füzyon klinik çeviri için sağlam, tekrarlanabilir ve denetlenebilir bir strateji olarak tercih edilmiştir (bkz. Bölüm~\ref{sec:teacher_ensemble}). Önceki çalışmalar, öğretmen havuzu büyüdüğünde adaptif ağırlıklandırmanın fayda sağlayabileceğini göstermektedir ve bu gelecek çalışma için uygundur \cite{islam2023,yilmaz2025,alshafi2024}.

### Literature Review

### Discussion

English:

This section interprets the experimental findings with respect to the research questions posed in Section 1, and places the observed results in the context of recent literature on hybrid architectures, knowledge distillation, ordinal learning, and ensemble strategies.

RQ1: Effectiveness of the hybrid teacher--student framework

We find that the proposed `ConvNeXtTiny_Residual` teacher--student pipeline yields consistent improvements over single-teacher baselines and many standalone backbones (see Section 4). The mean test QWK and accuracy gains align with prior observations that hybrid CNN--ViT architectures and ensemble strategies capture complementary representations useful for DR grading \cite{zhang2025,zhu2024,islam2023}. Multi-teacher distillation transfers complementary information and improves student robustness in medical imaging settings \cite{islam2023,yilmaz2025}; equal-weight ensemble soft labels provide a smoother, better-calibrated target that the ConvNeXt Tiny student can absorb effectively (cf. \cite{gou2021,miyato2018}). Student capacity matters: stronger students exploit the ensemble signal better than extremely lightweight students \cite{hinton2015,liu2022,nazih2023}.

RQ2: Contribution of ordinal-aware distillation to QWK

The ablation study shows that the ordinal CDF regularizer yields measurable QWK improvements relative to configurations trained without ordinal alignment. This is consistent with approaches that explicitly model ordinal structure (e.g., CORAL, unimodal regularization) which reduce distance-weighted errors on graded clinical tasks \cite{cao2020,liu2020}. Surveys note the misalignment between cross-entropy training objectives and QWK-type clinical metrics \cite{bappi2025,zhu2024}; integrating ordinal objectives addresses this gap. Comparable gains from ordinal or rank-aware objectives have been reported in related DR and medical-imaging studies \cite{ju2021,moya2024,farag2022,tian2023,romero2024}.

RQ3: Practicality of equal-weight fusion vs gated fusion

Entropy-gated fusion collapsed in the high-confidence, low-data regime (mean gate ~0.09), producing predictions effectively identical to equal-weight averaging. This suggests learned gating yields limited benefit when teacher predictions are well calibrated and strongly confident, consistent with KD/ensemble literature stating that simple averaging often suffices unless teachers exhibit substantial and complementary uncertainty \cite{gou2021,islam2023,yilmaz2025,goh2024}. Spectral normalization and other calibration-promoting regularizers further reduce the marginal value of complex gating by producing smoother teacher decision boundaries \cite{miyato2018}. Practically, equal-weight fusion is simpler and less sensitive to overfitting in low-data regimes; gated schemes may be useful when teacher pools are larger and more heterogeneous \cite{alshafi2024,wang2024}.

Limitations and future directions

The evaluation is limited to APTOS 2019 and three random seeds. Broader multi-dataset validation (EyePACS, Messidor, DDR) would strengthen generalizability claims \cite{zhu2024,bappi2025}. Future work includes exploring adaptive gating conditioned on inter-teacher disagreement, investigating stronger calibration techniques (e.g., temperature scaling), and quantifying the effect of student-side augmentation and lesion-level supervision on ordinal calibration and clinical interpretability \cite{gou2021,alshafi2024,wang2024}.

Türkçe:

Bu bölüm, Deneysel bulguları Bölüm 1'de ortaya konan araştırma soruları çerçevesinde yorumlar ve gözlemleri hibrid mimariler, bilgi damıtımı, sıralı öğrenme ve ensemble stratejileri üzerine güncel literatür bağlamında konumlandırır.

RQ1: Hibrid öğretmen--öğrenci çerçevesinin etkinliği

Önerilen `ConvNeXtTiny_Residual` öğretmen--öğrenci hattı, tek-öğretmen baz hatlarına ve birçok tekil omurgaya kıyasla tutarlı iyileşmeler göstermektedir (bkz. Bölüm 4). Ortalama test QWK ve doğruluk artışları, hibrid CNN--ViT mimarilerinin ve ensemble stratejilerinin DR derecelendirmesi için tamamlayıcı temsiller yakalayabildiğini gösteren önceki gözlemlerle uyumludur \cite{zhang2025,zhu2024,islam2023}. Çoklu-öğretmen damıtımı, tamamlayıcı bilgiyi aktarır ve öğrenci dayanıklılığını artırır \cite{islam2023,yilmaz2025}; eşit-ağırlıklı ensemble yumuşak etiketleri ConvNeXt Tiny öğrencisinin etkili biçimde öğrenebileceği daha düzgün, iyi kalibre edilmiş hedefler sağlar (bkz. \cite{gou2021,miyato2018}). Öğrenci kapasitesi önemlidir: daha güçlü öğrenciler, son derece hafif öğrencilere göre ensemble sinyalinden daha iyi faydalanır \cite{hinton2015,liu2022,nazih2023}.

RQ2: Sıralı-hassasiyete duyarlı damıtımın QWK'ye katkısı

Ablasyon çalışması, sıralı CDF düzenleyicisinin QWK'de ölçülebilir iyileşmeler sağladığını göstermektedir. Bu sonuç, sıralı yapıyı açıkça modelleyen yaklaşımlarla (ör. CORAL, unimodal düzenleme) uyumludur ve derecelendirilmiş klinik görevlerde uzaklığa duyarlı hataları azaltır \cite{cao2020,liu2020}. Derlemeler, çapraz entropi hedefleri ile QWK benzeri klinik metrikler arasındaki uyumsuzluğa dikkat çekmektedir \cite{bappi2025,zhu2024}; sıralı hedeflerin entegrasyonu bu boşluğu kapatır. Benzer kazanımlar, ilgili DR ve tıbbi görüntüleme çalışmalarında rapor edilmiştir \cite{ju2021,moya2024,farag2022,tian2023,romero2024}.

RQ3: Eşit-ağırlıklı füzyonun kapılı füzyona göre uygulanabilirliği

Entropi-kapılı füzyon, yüksek güvene sahip düşük veri rejiminde çökme gösterdi (ortalama kapı değeri ~0.09) ve pratik olarak eşit-ağırlıklı ortalamayla aynı tahminleri üretti. Bu durum, öğretmen tahminleri iyi kalibre edilmiş ve güçlü olduğunda öğrenilen kapıların sınırlı fayda sağladığını gösterir; KD/ensemble literatürü, öğretmenler arasında belirgin ve tamamlayıcı belirsizlik olmadıkça basit ortalamanın genellikle yeterli olduğunu belirtir \cite{gou2021,islam2023,yilmaz2025,goh2024}. Spektral normalizasyon gibi kalibrasyonu iyileştiren düzenleyiciler, karmaşık kapı mekanizmalarının değerini daha da azaltır \cite{miyato2018}. Pratikte, düşük veri koşullarında eşit-ağırlıklı füzyon daha basit ve aşırı uyuma daha az hassastır; kapılı stratejiler, daha büyük ve heterojen öğretmen havuzlarında yeniden değerlendirilebilir \cite{alshafi2024,wang2024}.

Sınırlamalar ve gelecek yönelimler

### Conclusion

English:

In this work we proposed `ConvNeXtTiny_Residual`, a hybrid teacher-ensemble knowledge distillation pipeline tailored for five-class diabetic retinopathy grading under low-data constraints. Empirical evaluations on APTOS 2019 demonstrate that structured multi-teacher distillation combined with ordinal-aware loss terms yields consistent improvements in clinically relevant metrics: the trained ConvNeXt Tiny student achieves high quadratic weighted kappa (QWK) and accuracy when distilled from an equal-weight ensemble of complementary CNN--ViT teachers, while remaining computationally efficient for deployment.

The contributions are threefold. First, a hybrid teacher ensemble that combines structurally diverse CNN and ViT representations produces a better-calibrated soft-label target than single teachers, improving student learning stability. Second, a multi-temperature distillation schedule together with an ordinal CDF regularizer aligns the optimization objective with distance-weighted clinical metrics (QWK), reducing clinically costly large-grade errors. Third, a non-inferiority constraint and practical equal-weight fusion reduce student regression risks and simplify ensemble deployment under limited-data scenarios. Quantitatively, the proposed configuration achieved mean test QWK = 0.9186\pm0.0010 and accuracy = 85.47%\pm0.31% across three seeds, exceeding compared baselines in our experiments.

Türkçe:

Bu çalışmada `ConvNeXtTiny_Residual` adlı, düşük veri kısıtlamaları altında beş sınıflı diyabetik retinopati derecelendirmesine yönelik hibrid öğretmen-topluluğu bilgi damıtımı hattı önerdik. APTOS 2019 üzerindeki deneysel değerlendirmeler, yapılandırılmış çoklu-öğretmen damıtımı ile sıralı-hassasiyetli kayıp terimlerinin birleşiminin klinik açıdan anlamlı metriklerde tutarlı iyileşmeler sağladığını gösteriyor: Eğitilmiş ConvNeXt Tiny öğrenci, tamamlayıcı CNN--ViT öğretmenlerinin eşit-ağırlıklı bir ensemble'ından damıtıldığında yüksek kuadratik ağırlıklı kappa (QWK) ve doğruluk elde ederken dağıtım açısından verimli bir şekilde konuşlandırılabilir.

Katkılar üç başlıkta özetlenebilir. Birincisi, yapısal olarak farklı CNN ve ViT temsillerini birleştiren hibrid öğretmen topluluğu, tek öğretmenlere göre daha iyi kalibre edilmiş yumuşak etiket hedefleri üreterek öğrenci öğrenmesini daha kararlı hale getirir. İkincisi, çoklu-sıcaklık distilasyon programı ve sıralı CDF düzenleyicisi, optimizasyon hedefini uzaklık-ağırlıklı klinik metriklerle (QWK) hizalayarak klinik açıdan maliyetli büyük-derece hatalarını azaltır. Üçüncüsü, non-inferiority kısıtı ve pratik eşit-ağırlıklı füzyon, sınırlı veri senaryolarında öğrenci regresyon risklerini azaltır ve ensemble konuşlandırmasını basitleştirir. Nicel olarak, önerilen konfigürasyon üç tohum üzerinde ortalama test QWK = 0.9186\pm0.0010 ve doğruluk = 85.47%\pm0.31% elde etti ve deneylerimizde karşılaştırılan baz hataları geride bıraktı.

#### 6.1 Practical Implications

English:

The practical implications of this study span model development, clinical screening workflows, and deployment constraints. From a model-development perspective, our results suggest that combining heterogeneous teacher architectures can provide complementary supervisory signals that a sufficiently capacious student can absorb, enabling compact models that approximate ensemble performance. For clinical screening, the ConvNeXt Tiny student offers a favorable trade-off between performance and inference latency (mean ~6.7 ms/image), permitting near-real-time batch processing in screening pipelines and on-device or edge deployment with modest hardware requirements. Calibration-promoting components (e.g., spectral normalization) and multi-temperature distillation improve probability estimates, which is critical when downstream decisions—referral versus routine follow-up—depend on model confidence. Practically, equal-weight fusion simplifies ensemble maintenance and auditing compared to learned gating, reducing the implementation complexity important for regulated clinical systems. Finally, the ordinal-aware loss directly aligns training with clinically meaningful error costs (QWK), making model outputs more interpretable for triage thresholds and enabling more clinically appropriate risk stratification.

Türkçe:

Bu çalışmanın pratik çıkarımları model geliştirme, klinik tarama iş akışları ve konuşlandırma kısıtlarını kapsar. Model geliştirme açısından, sonuçlarımız heterojen öğretmen mimarilerini birleştirmenin yeterli kapasiteye sahip bir öğrencinin öğrenebileceği tamamlayıcı denetleyici sinyaller sağlayabileceğini ve ensemble performansına yaklaşan kompakt modellerin elde edilebileceğini göstermektedir. Klinik taramada ConvNeXt Tiny öğrenci, performans ile çıkarım gecikmesi (ortalama ~6.7 ms/görüntü) arasında elverişli bir denge sunar; bu da tarama boru hatlarında gerçek zamanlıya yakın toplu işleme ve mütevazı donanım gereksinimleriyle cihaz içi veya uçta konuşlandırma sağlar. Spektral normalizasyon gibi kalibrasyonu artıran bileşenler ve çoklu-sıcaklık distilasyonu, model güvenine dayanan sevk kararları için kritik olan olasılık tahminlerini iyileştirir. Pratikte, eşit-ağırlıklı füzyon, öğrenilen kapı mekanizmalarına kıyasla ensemble bakımı ve denetimini basitleştirir ve düzenlemeye tabi klinik sistemler için uygulama karmaşıklığını azaltır. Son olarak, sıralı-hassasiyetli kayıp, eğitimi klinik olarak anlamlı hata maliyetleriyle (QWK) doğrudan hizalayarak model çıktılarının triage eşiklerinde daha yorumlanabilir olmasını ve klinik olarak daha uygun risk sınıflandırması yapılmasını sağlar.

#### 6.2 Limitations

English:

Several limitations temper the generalizability and operational readiness of the present study. First, the evaluation is restricted to the APTOS 2019 dataset and three random seeds; geographic, camera, and population biases present in the dataset could limit external validity. Second, teacher training used large pretrained backbones and substantial compute; while the student is compact, reproducing the teacher ensemble may be costly for some groups. Third, label noise and inter-grader variability—well-documented in retinal datasets—were not modeled beyond standard stratified splitting, which may inflate apparent performance and underestimate uncertainty in deployment contexts. Fourth, although we analyzed an entropy-gated fusion mechanism, our experiments indicate gate collapse in this low-data regime; we did not explore more sophisticated gating or disagreement-based selection strategies that may perform differently with larger or more diverse teacher pools. Fifth, lesion-level supervision, advanced augmentation strategies for the student, and prospective clinical validation were not evaluated here and remain open practical gaps.

Türkçe:

Bir dizi sınırlama, bu çalışmanın genelleştirilebilirliğini ve operasyonel hazır olma düzeyini sınırlamaktadır. Birinci olarak, değerlendirme APTOS 2019 veri seti ve üç rasgele tohum ile sınırlıdır; veri setinde bulunan coğrafi, kamera ve nüfus önyargıları dış geçerliliği sınırlayabilir. İkinci olarak, öğretmen eğitimi büyük ön-eğitimli omurgalar ve önemli hesaplama gereksinimleri kullandı; öğrenci kompakt olsa da öğretmen topluluğunu yeniden üretmek bazı gruplar için maliyetli olabilir. Üçüncü olarak, retina veri kümelerinde belgelenen etiket gürültüsü ve değerlendiriciler arası değişkenlik, standart tabakalı bölmenin ötesinde modellenmemiştir; bu, görünür performansı şişirebilir ve konuşlandırma bağlamında belirsizliği olduğundan az tahmin edebilir. Dördüncü olarak, entropi-kapılı bir füzyon mekanizması analiz edilmiş olsa da, deneylerimiz bu düşük veri rejiminde kapı çöküşünü göstermiştir; daha büyük veya daha çeşitli öğretmen havuzlarında farklı performans gösterebilecek daha sofistike kapı veya anlaşmazlık-temelli seçim stratejileri araştırılmamıştır. Beşinci olarak, lezyon düzeyinde denetim, öğrenci için ileri augmentasyon stratejileri ve prospektif klinik doğrulama burada değerlendirilmedi ve pratik boşluklar olarak kalmaktadır.

#### 6.3 Future Work

English:

Future research directions naturally follow from the limitations above. Immediate next steps include multi-dataset external validation (EyePACS, Messidor, DDR) to quantify generalization across acquisition settings and patient populations, and prospective evaluation on held-out clinical cohorts. Methodologically, investigating adaptive gating conditioned on explicit inter-teacher disagreement metrics, temperature-scaling calibration and post-hoc calibration methods, and stronger student-side augmentation pipelines may improve robustness and calibration. Incorporating lesion-level supervision or multi-task formulations (lesion segmentation + grade prediction) could improve interpretability and clinical relevance. From a deployment perspective, studying compressed student variants and quantization-aware training to reduce memory and energy footprints, and building a reproducible pipeline for teacher ensemble training with standardized reporting of calibration and uncertainty, are important steps toward clinical translation. Finally, ethical and fairness analyses—examining model performance across demographic subgroups and imaging devices—are essential before deployment in screening programs.

Türkçe:

Gelecek araştırma yönelimleri yukarıdaki sınırlardan doğal olarak türetilir. Hemen atılacak adımlar arasında, edinim koşulları ve hasta popülasyonları arasında genelleştirilebilirliği nicelleştirmek için çoklu veri setinde dış doğrulama (EyePACS, Messidor, DDR) ve ayrılmış klinik kohortlarda prospektif değerlendirme yer alır. Metodolojik olarak, öğretmenler arası anlaşmazlık metriklerine koşullu adaptif kapılama, sıcaklık ölçeklendirme ve sonradan kalibrasyon yöntemleri ve daha güçlü öğrenci tarafı augmentasyon boru hatları, sağlamlık ve kalibrasyonu iyileştirebilir. Lezyon düzeyinde denetimi veya çoklu görev biçimlerini (lezyon segmentasyonu + derece tahmini) dahil etmek, yorumlanabilirliği ve klinik alaka düzeyini artırabilir. Konuşlandırma perspektifinden, bellek ve enerji ayak izlerini azaltmak için sıkıştırılmış öğrenci varyantlarını ve kuantizasyon-dikkatli eğitimi incelemek ve öğretmen topluluğu eğitimi için kalibrasyon ve belirsizlik raporlamasını standartlaştıran tekrarlanabilir bir boru hattı oluşturmak, klinik çeviri için önemli adımlardır. Son olarak, tarama programlarında konuşlandırmadan önce demografik alt gruplar ve görüntüleme cihazları arasında model performansını inceleyen etik ve adalet analizleri zorunludur.

Conventional DR screening relies on trained ophthalmologists who manually examine retinal fundus images and assign a severity grade according to standardized clinical scales. The International Clinical Diabetic Retinopathy (ICDR) severity scale classifies retinopathy into five ordinal grades, from no apparent retinopathy (Grade 0) to proliferative DR (Grade 4) \cite{wilkinson2003}. Although effective, manual grading is time-intensive, costly, and subject to inter-grader variability---a limitation particularly acute in low- and middle-income countries where specialist availability is severely constrained \cite{krause2018}. These systemic barriers motivate the development of automated, accurate, and computationally efficient grading systems.

Paragraph 2 — Türkçe:

Geleneksel DR taraması, retinal fundus görüntülerini manuel olarak inceleyen ve standart klinik ölçeklere göre şiddet derecesi atayan eğitimli göz doktorlarına dayanır. Uluslararası Klinik Diyabetik Retinopati (ICDR) şiddet ölçeği, retinopatiyi belirgin retinopatinin olmadığı (Derece 0) durumdan proliferatif DR'ye (Derece 4) kadar beş sıralı dereceye ayırır \cite{wilkinson2003}. Etkili olmasına rağmen, manuel derecelendirme zaman alıcı, maliyetli ve değerlendiriciler arası değişkenliğe tabidir—uzman erişiminin ciddi şekilde kısıtlı olduğu düşük ve orta gelirli ülkelerde bu sınırlama daha da belirgindir \cite{krause2018}. Bu yapısal engeller, otomatik, doğru ve hesaplama açısından verimli derecelendirme sistemlerinin geliştirilmesini motive eder.

