from pathlib import Path
import re


FILE_PATH = Path(r'c:\Users\user\Desktop\DR_Research\PAPER_DRAFT_INTRO_LITERATURE_PROPOSED_MODEL.md')
REMOVED_REFS = {3, 4, 17}


def renumber_reference(number: int) -> int | None:
    if number in REMOVED_REFS:
        return None
    return number - sum(1 for removed in REMOVED_REFS if removed < number)


def subn(content: str, pattern: str, repl: str, label: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, repl, content, flags=flags)
    print(f'{"OK" if count else "WARN"}   {label}: {count}')
    return updated


def renumber_citations(content: str) -> str:
    pattern = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')

    def repl(match: re.Match[str]) -> str:
        numbers = [int(part.strip()) for part in match.group(1).split(',')]
        mapped = [str(new) for old in numbers if (new := renumber_reference(old)) is not None]
        if not mapped:
            return ''
        return '[' + ', '.join(mapped) + ']'

    return pattern.sub(repl, content)


content = FILE_PATH.read_text(encoding='utf-8')

content = subn(content, r'\n> \*\*Revision notes \(internal\):\*\*[\s\S]*$', '', 'remove internal notes')

content = subn(content, r'\[3\] Teo ZL,[^\n]*\n> \*\*Alıntı:[^\n]*\n?', '', 'remove Teo ref block')
content = subn(content, r'\[4\] Wilkinson CP,[^\n]*\n> \*\*Alıntı:[^\n]*\n?', '', 'remove Wilkinson ref block')
content = subn(content, r'\[17\] Rautaray J,[^\n]*\n> \*\*Alıntı \(×3\):[^\n]*\n?', '', 'remove Rautaray ref block')

content = subn(content, r'\s*stages—including severe non-proliferative and proliferative DR—if not detected and treated in a timely manner \[3\]\.', '.', 'remove EN Teo citation')
content = subn(content, r'önemli bir kısmı, zamanında tespit edilip tedavi edilmediği takdirde görmeyi tehdit eden evrelere—şiddetli non-proliferatif ve proliferatif DR—ilerlemektedir \[3\]\.', 'önemli bir kısmı, zamanında tespit edilip tedavi edilmediği takdirde görmeyi tehdit eden evrelere—şiddetli non-proliferatif ve proliferatif DR—ilerlemektedir.', 'remove TR Teo citation')
content = subn(content, r'widely adopted in clinical practice \[4\]\.', 'widely adopted in clinical practice.', 'remove EN Wilkinson citation')
content = subn(content, r'klinik pratikte yaygın olarak benimsenmektedir \[4\]\.', 'klinik pratikte yaygın olarak benimsenmektedir.', 'remove TR Wilkinson citation')

content = subn(content, r'Rautaray et al\. \[17\] further demonstrated that cross-architecture distillation from a ViT-based teacher into a compact CNN student is viable for DR severity classification with substantially reduced model complexity\.\s*', '', 'remove EN Rautaray intro sentence')
content = subn(content, r'Rautaray ve ark\. \[17\] ise ViT tabanlı öğretmenden kompakt CNN öğrencisine çapraz mimari damıtmanın, model karmaşıklığını önemli ölçüde azaltarak DR şiddet sınıflandırmasında uygulanabilir olduğunu göstermiştir\.\s*', '', 'remove TR Rautaray intro sentence')
content = subn(content, r'cost-sensitive learning \[17\], and ', '', 'remove EN cost-sensitive phrase')
content = subn(content, r'maliyet duyarlı öğrenme \[17\] ve ', '', 'remove TR cost-sensitive phrase')
content = subn(content, r'Rautaray et al\. \[17\] demonstrated in the DR domain that a strong ViT-based teacher \(FastViT-MA26\) can effectively transfer discriminative knowledge to a compact student \(EfficientNet-B0\) on APTOS 2019—underscoring the value of structured knowledge transfer from a high-capacity teacher\.\s*', '', 'remove EN Rautaray ensemble sentence')
content = subn(content, r'Rautaray ve ark\. \[17\], DR alanında güçlü bir ViT tabanlı öğretmenin \(FastViT-MA26\) APTOS 2019 üzerinde kompakt bir öğrenciye \(EfficientNet-B0\) yüksek ayrımcı bilgi etkin biçimde aktarabildiğini göstermiş; yüksek kapasiteli öğretmenden yapılandırılmış bilgi aktarımının değerini vurgulamıştır\.\s*', '', 'remove TR Rautaray ensemble sentence')
content = subn(content, r'\| Rautaray et al\. \[17\] \| FastViT-MA26→EfficientNet-B0 KD \| — \| \*\*0\.970\*\*§ \| — \| WKS; cost-sensitive learning \|\n', '', 'remove Rautaray table row')
content = subn(content, r'§ Weighted kappa score \(WKS\) with cost-sensitive learning; direct comparison with QWK requires caution\.  \n', '', 'remove WKS footnote')
content = subn(content, r'Rautaray et al\. \[17\] report a higher WKS \(0\.970\), but this metric incorporates cost-sensitive class weights that systematically differ from the symmetric QWK weighting used here\.\s*', '', 'remove Rautaray comparison sentence')

content = renumber_citations(content)
print('OK   renumber citations globally')

content = subn(content, r'10\.1109/CVPR\.2022\.01167', '10.1109/CVPR52688.2022.01167', 'fix ConvNeXt DOI')
content = subn(content, r"Azzou et al\. \[22\] evaluated a hybrid combining ResNet50's granular feature extraction with ViT's global relational reasoning on the APTOS dataset, finding the hybrid outperformed both standalone models with 98% average precision across all classes—a gain of \+7% over the standalone ViT for early DR\.", "Azzou et al. [22] evaluated a hybrid combining ResNet50's granular feature extraction with ViT's global relational reasoning on the APTOS dataset in a simplified three-class formulation, finding the hybrid outperformed both standalone models with 98% average precision across all classes—a gain of +7% over the standalone ViT for early DR.", 'clarify Azzou EN scope')
content = subn(content, r"Azzou ve ark\. \[22\], APTOS veri kümesi üzerinde ResNet50'nin ayrıntılı öznitelik çıkarımını ViT'in küresel ilişkisel akıl yürütmesiyle birleştiren bir hibridi değerlendirmiş; hibridin erken DR için bağımsız ViT'e kıyasla \+%7 kazanımla tüm sınıflarda %98 ortalama hassasiyetle her iki bağımsız modeli de geride bıraktığını tespit etmiştir\.", "Azzou ve ark. [22], APTOS veri kümesi üzerinde sadeleştirilmiş üç sınıflı bir kurulumda ResNet50'nin ayrıntılı öznitelik çıkarımını ViT'in küresel ilişkisel akıl yürütmesiyle birleştiren bir hibridi değerlendirmiş; hibridin erken DR için bağımsız ViT'e kıyasla +%7 kazanımla tüm sınıflarda %98 ortalama hassasiyetle her iki bağımsız modeli de geride bıraktığını tespit etmiştir.", 'clarify Azzou TR scope')
content = subn(content, r'Nazih et al\. \[24\] demonstrated that a ViT applied directly to DR severity grading, optimized with AdamW and trained with focal loss and class weights to address imbalance, achieves competitive F1-score of 0\.825 and AUC of 0\.956\.', 'Nazih et al. [24] demonstrated that a ViT applied directly to DR severity grading, optimized with AdamW and trained with focal loss and class weights to address imbalance, achieved an AUC of 0.956.', 'remove Nazih EN F1')
content = subn(content, r"Nazih ve ark\. \[24\], AdamW ile optimize edilen ve dengesizliği gidermek amacıyla odak kaybı ile sınıf ağırlıkları kullanılarak eğitilen, DR şiddet derecelendirmesine doğrudan uygulanan bir ViT'in rekabetçi F1-skoru = 0\.825 ve AUC = 0\.956 elde ettiğini göstermiştir\.", "Nazih ve ark. [24], AdamW ile optimize edilen ve dengesizliği gidermek amacıyla odak kaybı ile sınıf ağırlıkları kullanılarak eğitilen, DR şiddet derecelendirmesine doğrudan uygulanan bir ViT'in AUC = 0.956 elde ettiğini göstermiştir.", 'remove Nazih TR F1')
content = subn(content, r'\[27\] Xue D, Feng X\.', '[27] Luo L, Xue D, Feng X.', 'fix Luo first author')
content = subn(content, r'Moya-Albor et al\. \[29\] employed an Inception-v3 teacher–student pipeline with a novel combination of KL divergence and categorical cross-entropy as the distillation loss, achieving average training accuracy of 99% and validation accuracy of 97% on DR lesion classification—directly paralleling the loss formulation adopted in the present work\.', 'Moya-Albor et al. [29] employed an Inception-v3 teacher–student pipeline with a novel combination of KL divergence and categorical cross-entropy as the distillation loss, achieving average training accuracy of 99% and validation accuracy of 97% on DR lesion classification rather than five-class severity grading; this supports the use of combined distillation losses but is not a directly comparable grading benchmark.', 'clarify Moya-Albor EN scope')
content = subn(content, r'Moya-Albor ve ark\. \[29\], damıtma kaybı olarak KL diverjansı ile kategorik çapraz entropi kombinasyonunu kullanan Inception-v3 öğretmen–öğrenci boru hattını kullanmış; DR lezyon sınıflandırmasında ortalama %99 eğitim ve %97 doğrulama doğruluğu elde etmiştir; bu durum, söz konusu çalışmada benimsenen kayıp formülasyonuyla doğrudan örtüşmektedir\.', 'Moya-Albor ve ark. [29], damıtma kaybı olarak KL diverjansı ile kategorik çapraz entropi kombinasyonunu kullanan Inception-v3 öğretmen–öğrenci boru hattını kullanmış; beş sınıflı şiddet derecelendirmesi yerine DR lezyon sınıflandırmasında ortalama %99 eğitim ve %97 doğrulama doğruluğu elde etmiştir; bu durum birleşik damıtma kayıplarının kullanımını desteklese de doğrudan karşılaştırılabilir bir derecelendirme kıyaslaması sunmamaktadır.', 'clarify Moya-Albor TR scope')
content = subn(content, r'\| Nazih et al\. \[24\] \| ViT \+ focal loss \+ AdamW \| — \| — \| 0\.825 \| AUC = 0\.956 \|', '| Nazih et al. [24] | ViT + focal loss + AdamW | — | — | — | AUC = 0.956; different evaluation setup |', 'adjust Nazih table row')
content = subn(content, r'\| Ikram & Imran \[21\] \| ResViT FusionNet \| 93\.01 \| 0\.894‡ \| — \| LIME/Grad-CAM; kappa reported \|', '| Ikram & Imran [21] | ResViT FusionNet | 93.01 | 0.894‡ | — | LIME/Grad-CAM; kappa reported; different APTOS 2019 Kaggle variant |', 'add Ikram table note')

content = re.sub(r'\n{3,}', '\n\n', content)

FILE_PATH.write_text(content, encoding='utf-8')
print('OK   saved normalized draft')

all_refs = sorted({int(value) for value in re.findall(r'\[(\d+)\]', content)})
reference_entries = sorted({int(match.group(1)) for match in re.finditer(r'^\[(\d+)\] ', content, flags=re.MULTILINE)})
print('All bracket refs:', all_refs)
print('Reference entry refs:', reference_entries)
print('Max ref:', max(all_refs) if all_refs else 'none')
print('Has Teo:', 'Teo ZL' in content)
print('Has Wilkinson:', 'Wilkinson CP' in content)
print('Has Rautaray:', 'Rautaray' in content)
print('Has updated ConvNeXt DOI:', '10.1109/CVPR52688.2022.01167' in content)
print('Has Luo first author:', '[27] Luo L, Xue D, Feng X.' in content)