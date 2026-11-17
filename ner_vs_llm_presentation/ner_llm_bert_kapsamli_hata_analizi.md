# Information Extraction: Traditional NER vs. LLM Parsing
## 9 deneyin tam hata analizi ve BERT `simple` / `first` karşılaştırması

**Analiz tarihi:** 16 Kasım 2026  
**Kaynaklar:** Kullanıcının sağladığı `results.zip` ve `bert_analysis.zip` arşivlerinin gerçek CSV/JSON çıktıları; `english_test.jsonl` gold etiketleri ve kaynak metinleri. Kayıtlı model tahminleri incelendi; yeniden model çalıştırılmadı.  
**Kontrol:** 50 örneğin gold alanları yüklenen ana JSONL ile eşleşiyor. BERT `simple` ve `first` için hem CPU hem MPS ayrıntılı tahminleri mevcut. F1 değerleri kayıtlı TP/FP/FN sayılarından yeniden doğrulandı. `first` seçimi test setindeki hatalar görüldükten sonra yapıldığı için test seti üzerinde optimizasyon sınırlılığı vardır.

## 1. Araştırma sorusu ve deney düzeni

**Ana soru:** Göreve özel hazırlanmış NER modelleri (spaCy ve NER için fine-tune edilmiş BERT), prompt ile yapılandırılmış çıkarım yapan bir LLM'ye (Qwen3:4b-instruct) göre doğruluk, esneklik ve hız açısından nasıl farklılaşır?

- **Ana test:** 50 sentetik İngilizce paragraf, 5 grup × 10 örnek.
- **Ek dil testi:** 10 Türkçe paragraf, 10 İngilizce eşlenik metin.
- **Geliştirme kümesi:** 7 ayrı metin; ana test F1 hesabına katılmadı.
- **Ana etiketler:** PERSON, ORG, LOC. BERT `PER` etiketini `PERSON` olarak eşliyor; BERT `MISC` temel değerlendirmeye dahil edilmiyor. spaCy `GPE` ve `LOC` etiketleri `LOC` olarak normalize ediliyor.
- **Başarı ölçümü:** Varlığın türü ile başlangıç ve bitiş karakter sınırının eşleştiği, **exact-span typed micro Precision / Recall / F1**.
- **Önemli sınır:** Bütün test metinleri sentetik; elde edilen sonuçlar daha geniş gerçek dünya NER performansı hakkında kesin hüküm değildir.

## 2. Ana model karşılaştırması (50 İngilizce test metni)

| Model ve çalışma | Precision | Recall | Micro-F1 | TP | FP | FN | Ortalama gecikme |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BERT NER, `first`, Mac CPU** | 0.9617 | 0.9778 | **0.9697** | 176 | 7 | 4 | 0.0587 sn |
| **BERT NER, `first`, Mac MPS** | 0.9617 | 0.9778 | **0.9697** | 176 | 7 | 4 | 0.0306 sn |
| BERT NER, `simple`, Mac CPU | 0.8549 | 0.9167 | 0.8847 | 165 | 28 | 15 | 0.0484 sn |
| BERT NER, `simple`, Mac MPS | 0.8549 | 0.9167 | 0.8847 | 165 | 28 | 15 | 0.0252 sn |
| **spaCy `en_core_web_sm`, CPU** | 0.8793 | 0.8500 | 0.8644 | 153 | 21 | 27 | 0.0186 sn |
| **Qwen3:4b-instruct**, schema-constrained | 0.7241 | 0.8167 | 0.7676 | 147 | 56 | 33 | 3.7082 sn |

**Yorum:** Hazır NER BERT + `first` bu sabit testte en yüksek Precision, Recall ve F1 değerlerini verdi. `simple` kullanıldığında ilk F1 0.8847 idi. spaCy CPU en düşük ölçülen ortalama gecikmeye ulaştı. BERT `first` üzerinde MPS, CPU'dan yaklaşık **1.92×** hızlıydı (30.57 ms / 58.69 ms); bu oran tek koşudan ve sistem yükünden etkilenebilir. Qwen generative yaklaşımı daha fazla zaman aldı. Aynı 50 metin kullanılsa da backend/cihaz farkları, kalıcı model yükleme ve tek koşu nedeniyle bunlar kontrollü donanım benchmark'ı değil, gözlenen pratik gecikmelerdir.

**Kritik metodoloji sınırlaması:** BERT `first` stratejisi, `simple` sonuçları analiz edildikten sonra test kümesinde denendi. **0.9697 bu veri seti üzerinde sonradan seçilmiş stratejinin gözlenen F1'idir**, yeni/bağımsız metinlerde doğrulanmış genel başarı tahmini değildir.

**Ana veri grupları:**

| İngilizce alt grup (10'ar metin) | spaCy F1 | Qwen F1 | BERT `first` F1 |
|---|---:|---:|---:|
| standard | 0.868 | 0.711 | **1.000** |
| ambiguity | 0.814 | 0.725 | **0.968** |
| multi_entity | 0.876 | **0.882** | **0.956** |
| extended | 0.842 | 0.716 | **0.962** |
| challenging | **0.926** | 0.781 | **0.963** |

Qwen `multi_entity` alt grubunda çok küçük bir farkla daha iyi çıktı; 10 metinlik alt gruptaki fark güçlü bir genelleme değildir. `extended` alt grubunda da ana Accuracy deneyinin F1 hesabı yalnızca **PERSON/ORG/LOC** için yapıldı; JOB_TITLE, DATE ve PRODUCT bu F1'e eklenmedi.

## 3. Deney 1 — Accuracy

**Amaç:** spaCy ve Qwen'in aynı 50 metinde doğru tür ve karakter sınırıyla NER çıkarımı yapma başarısını karşılaştırmak.

**Sonuç:** spaCy F1 **0.8644** (TP 153, FP 21, FN 27); Qwen F1 **0.7676** (TP 147, FP 56, FN 33). spaCy'nin precision avantajı belirgin. Qwen'in bazı yanlış veya yinelenen tahminleri FP sayısını artırıyor.

**Gözlenen hata sınıflaması** (ana accuracy `detail.csv` içindeki gold ve predicted aralıklarının eşleştirilmesiyle):

| Ayrıklaştırılmış hata tipi | spaCy | Qwen | Açıklama |
|---|---:|---:|---|
| Tam sınırda **yanlış tür** eşleştirmesi | 18 | 1 | Aynı aralık, farklı etiket; 1 FP + 1 FN |
| **Örtüşen ama tam eşleşmeyen** span çifti | 0 | 8 | Sınır veya tür uyuşmazlığı; 1 FP + 1 FN |
| Ekstra tahmin (eşleşmeyen `predicted`) | 3 | 32 | FP |
| Kaçırılan gold (eşleşmeyen) | 9 | 24 | FN |
| Kaynakla eşleştirilemeyen/yinelenen çıktı (`ungrounded`) | 0 | 15 | Değerlendirme protokolünde ek FP |

Bu tablonun sınıfları, ham tahmin ve altın standardı tek tek eşleştiren **analitik ayrıştırmadır**; modelin iç nedenini kanıtlamaz. Tür hatası ve span hatası 1'er FP ve FN sayılır; Qwen FP = 1 + 8 + 32 + 15 = 56; FN = 1 + 8 + 24 = 33. spaCy FP = 18 + 3 = 21; FN = 18 + 9 = 27.

**Örnek EN019 (Orange):** Gold: iki ayrı `Orange → ORG`, `Lyon → LOC`, `Theo Lambert → PERSON`. spaCy `Lyon → ORG` tahmin etti ve Orange'ları kaçırdı (F1 0.333). Qwen bir Orange'ı doğru buldu, ancak `Theo` ve `Theo Lambert` ifadelerini ayrı PERSON olarak çıkardı ve ikinci Orange'ı kaçırdı (F1 0.750). Bu örnek spaCy'de varlık türünü yanlış belirleme, Qwen'de yinelenen/örtüşen kişi adı hatası gösterir.

**Örnek EN001:** Gold `Sarah Johnson → PERSON` iken Qwen `Sarah → PERSON` çıkardı; tam başlangıç/bitiş sınırını koruyamadı. Ayrıca `everyone → PERSON` gibi gereksiz çıktı üretti ve Google/London'u birden fazla kez tahmin ederek eşleştirilemeyen tekrarlar yarattı.

## 4. Deney 2 — Flexibility

**Amaç:** Ana NER modellerindeki standart etiketler dışında `JOB_TITLE` gibi yeni/ek kategorilerde prompt tabanlı çıkarım başarısını incelemek.

**Sonuç:** Qwen, 10 metinde `JOB_TITLE` için **Precision 0.7692, Recall 1.0000, F1 0.8696**; 10 TP, 3 FP, 0 FN. `en_core_web_sm` kontrol noktası `JOB_TITLE` etiketini sunmuyor; buna F1=0 demek metodolojik olarak yanlış olur. Bir BERT modeli de uygun yeni etiketler için ayrıca fine-tune edilebilir.

**EN035:** Gold `project manager → JOB_TITLE`; Qwen doğru etiketin yanında `product names` ve `occupational titles` ifadelerini de yanlış JOB_TITLE saydı (F1 0.50). **EN039:** `marketing specialist` doğruyken ayrıca `marketing` ayrı bir JOB_TITLE olarak etiketlendi (F1 0.667). **Olası neden:** kavramsal kategori eşleştirmesinin fazla geniş yapılması ve aynı span'in alt parçalarını ayrı varlık olarak üretme.

## 5. Deney 3 — Ambiguity

**Amaç:** Orange, Apple, Amazon gibi ifadelerin bağlama göre kişi/kurum/yer veya sıradan sözcük olarak ayrılmasını değerlendirmek.

**Sonuç:** spaCy n10 F1 **0.8136** (TP24, FP4, FN7); Qwen F1 **0.7246** (TP25, FP13, FN6), Qwen `ungrounded` **4**. Qwen daha çok TP üretse de fazladan FP nedeniyle F1'i düştü.

**EN012:** `apple`, metinde meyve; gold sadece `Paris → LOC`, `Nina Brooks → PERSON`. spaCy ikisini doğru çıkarıp F1 1.0 elde etti. Qwen iki doğru varlığa ek olarak `the team → ORG` üretti ve aynı değerleri defalarca tekrarladı; F1 **0.50**. Yanlışlık varlığın metinde bulunmamasından değil, varlık olarak **fazla etiketlenmesi** ve tekrardan kaynaklandı.

## 6. Deney 4 — JSON Reliability

**Amaç:** Yalnızca prompt ile JSON istemek ile Ollama'nın JSON Schema kısıtlı üretimini karşılaştırmak.

| Koşul | Raw JSON geçerli | Schema uygunluğu | Micro-F1 | TP / FP / FN |
|---|---:|---:|---:|---|
| Prompt-only | %100 | **%94** | 0.6750 | 135 / 85 / 45 |
| Schema-constrained | %100 | **%100** | **0.7676** | 147 / 56 / 33 |

**Gerçek şema hataları:** `EN032`, `EN039`, `EN042`. Üçünde de çıktı sentaktik olarak doğru JSON olmasına rağmen izin verilen `PERSON/ORG/LOC` etiketleri dışına çıktı. EN032'de `iPhone 16 → PRODUCT`, `2025 → DATE`; EN039'da `Spotify Wrapped → PRODUCT`; EN042'de `August 4, 2025 → DATE` üretildi. JSON Schema'nın kategori enum'u bu etiketleri kabul etmiyor. Değerlendirme hattı şeması yanlış olan **tüm tahmin dizisini reddettiği için** bu üç örneğin F1'i sıfır oldu; bu, iki koşul arasındaki F1 farkının bir kısmını doğrudan açıklıyor.

**Yorum:** `raw JSON geçerliliği`, `schema uygunluğu` ve `entity doğruluğu` üç ayrı ölçüt. Şemanın doğruluğu artırması yalnızca biçim düzeltmesi değildir; farklı kısıtlı çözümleme/çıktı davranışı ve hatalı şemalı kayıtları tümden reddetme gibi değerlendirme etkileri de vardır. Bu koşullarda gözlenen farkı nedensel olarak yalnızca “şema, NER'i daha akıllı yaptı” biçiminde yorumlamamalıyız.

## 7. Deney 5 — Efficiency

**Amaç:** Yerel inference maliyetini karşılaştırmak. **Gerçek dosyada bir 50 metinlik warm repeat** bulunuyor; 3 tekrar varsayılmamalı.

- spaCy: ortalama **0.0198 sn/metin**, F1 0.8644.
- Qwen: ortalama **3.7563 sn/metin**, F1 0.7676.
- Fark: bu koşullarda spaCy yaklaşık **190×** daha hızlı.

Qwen generative decoder, token üreterek bir JSON yanıtı oluşturur; spaCy doğrudan NER spanleri döndürür. Bu mimari fark süreyi açıklayan makul bir etkendir, ancak yürütme backend'i, donanım kullanımının kapsamı, model boyutu ve istem uzunluğu da rol oynar. Tekrarlı benchmark olmadan kesin evrensel hız oranı ileri sürülmemelidir.

## 8. Deney 6 — Temperature

**Amaç:** `temperature=0` ve `temperature=1` koşullarında çıkarsamaların kararlılığını incelemek; 50 metin × 3 tekrar × 2 sıcaklık = 300 model çağrısı.

- `temperature=0.0`: F1 **0.7676 / 0.7676 / 0.7676**.
- `temperature=1.0`: F1 **0.7696 / 0.7545 / 0.7551**.
- Her alt koşulda raw JSON ve schema uygunluğu %100.
- `temperature=1` koşulunda `ungrounded` sayıları tekrarlar boyunca **11 / 15 / 19**; sıfırda **15 / 15 / 15**.

**Yorum:** Temperature 0, bu çalışmada **metrik düzeyinde** daha sabit sonuç verdi; bu, çıktı dizelerinin yüzde yüz aynı olduğunu tek başına kanıtlamaz. Temperature 1 tek bir tekrarda küçük farkla daha yüksek olsa da diğer iki tekrarda daha düşük F1 verdi. Aynı 50 metne yapılan tekrarları 300 bağımsız test metni gibi değerlendirmemek gerekir.

## 9. Deney 7 — Few-shot

**Amaç:** Zero-shot ile üç etiketli örneğin prompta eklenmesinin doğruluk ve süreye etkisi.

- Zero-shot (n50): **P 0.7241 / R 0.8167 / F1 0.7676**, TP147 / FP56 / FN33, ortalama **4.3805 sn**.
- Three-shot (n50): **P 0.7295 / R 0.8389 / F1 0.7804**, TP151 / FP56 / FN29, ortalama **6.0716 sn**.
- F1 farkı +**0.0127** (yaklaşık 1.27 yüzde puanı); süre artışı yaklaşık **%38.6**.
- Three-shot `ungrounded` 17; zero-shot 15.

**EN012 önemli karşı örnek:** Three-shot, `lunch`, `apple`, `magazine`, `May 2025` ifadelerini LOC; `technology company`, `team` ifadelerini ORG olarak yanlış yorumladı; F1 0.40'a düştü (zero-shot 0.50). Dolayısıyla ortalama iyileşme her örnek için geçerli değil. Olası neden: verilen örneklerden aşırı genelleme ve sınıf sınırlarının bulanıklaşması.

## 10. Deney 8 — Language Sensitivity

**Amaç:** Metin dilinin (EN/TR) ve prompt dilinin (EN/TR) ayrı etkilerini 2 × 2 düzende incelemek.

| Prompt | Metin | n | F1 | Ortalama süre |
|---|---|---:|---:|---:|
| İngilizce | İngilizce | 10 | 0.7105 | 3.5961 sn |
| İngilizce | Türkçe | 10 | **0.7429** | 3.6119 sn |
| Türkçe | İngilizce | 10 | 0.7397 | **3.4441 sn** |
| Türkçe | Türkçe | 10 | 0.7397 | 3.8969 sn |

**TR001:** `Londra` gold LOC iken İngilizce prompt/Türkçe metin koşulunda aynı zamanda `Londra'da` da LOC olarak tahmin edildi; Türkçedeki durum eki/karakter sınırı etkisine örnek. Türkçe prompt/Türkçe metin koşulunda `Sarah Johnson` tam kişi adı yerine `Sarah` ve `Johnson` olarak bölündü. Bunun tek nedeni dil olduğunu söyleyemeyiz; prompt, veri içeriği ve span eşleştirme birlikte rol oynayabilir. İngilizce ve Türkçe alt kümelerde gold varlık sayıları da farklıdır (sırasıyla 39 ve 38); küçük F1 farkları genellenmemeli.

## 11. Deney 9 — Unseen / Challenging

**Amaç:** Ana 50 İngilizce test paragrafının içindeki 10 zorlayıcı metinde model karşılaştırması. **Bağımsız dış veri kümesi değildir.**

- spaCy: TP25, FP2, FN2, **F1 0.9259**, ortalama 0.0265 sn.
- Qwen: TP25, FP12, FN2, **F1 0.7813**, ortalama 3.9641 sn; `ungrounded` 5.

İki model de aynı miktarda TP ve FN üretmiş; ayrım, Qwen'in FP sayısının yüksek olmasıdır.

**EN050:** Gold yalnızca `Berlin → LOC`. spaCy doğru çıkarım yaptı (F1 1.0). Qwen `Berlin → PERSON`, `2023 → ORG` üretti; aynı zamanda `Berlin → LOC` tekrarını span eşleme aşamasında kullanamadı, `ungrounded` olarak raporlandı. Bu bir **yanlış kategori + duplicate alignment** örneğidir; `Berlin` ve `2023` metinde yer aldığından “uydurulmuş varlık adı” demek doğru olmaz.

**EN041:** Gold `Google → ORG`, `Alice Monroe → PERSON`, `New York → LOC`. Qwen bunları bulmasına rağmen `Reviewer`/`reviewer` ifadelerini PERSON saydı ve aynı adı yineledi (F1 0.667); spaCy F1 1.0. Bu, gereksiz rol sözcüklerinin varlık gibi algılanmasına örnek.

**EN047:** spaCy `Netflix → LOC` yanlış türü verdi, Qwen ise `Netflix → ORG` türünü doğru verdi; ama Qwen ham çıktısındaki yinelenen Netflix/Dublin öğeleri FP cezası yarattı. Tekil örneklerde Qwen'in daha iyi olabildiği görülüyor.

## 12. Hata taksonomisi: neyi ne olarak adlandırmalıyız?

| Başlık | Gözlenebilir tanım | Kanıt niteliğinde örnek |
|---|---|---|
| **Wrong type** | Doğru yazım/aralık, yanlış etiket | EN050: Berlin `LOC` yerine `PERSON` |
| **Boundary error** | Metin parçasının `start/end` sınırı gold ile farklı | EN001: `Sarah Johnson` yerine `Sarah` |
| **Missing entity (FN)** | Gold varlık hiç doğru çıkarılmadı | EN019: spaCy Orange ORG örneklerini kaçırdı |
| **Spurious entity (FP)** | Kaynaktaki sıradan ifade varlık diye etiketlendi | EN012: `the team → ORG`; EN041: `reviewer → PERSON` |
| **Duplicate extraction** | Aynı ifade/konum yinelenerek üretildi | EN001: Google/London tekrarları |
| **Ungrounded / unalignable** | Modeldeki metin, değerlendirme eşleyicisi tarafından boş bir kaynak spanine yerleştirilemedi | EN001, EN050; çoğu örnek tekrar kaynaklı |
| **True hallucination** | Kaynakta desteklenmeyen, gerçekten uydurulmuş içerik | **Ana Accuracy'deki 15 kayıt kaynakta var. Fakat diğer koşullarda dört kaynak-dışı ifade saptandı** |
| **JSON syntax error** | `json.loads` başarısız | JSON deneyinde **yok** (%100 raw JSON geçerli) |
| **JSON Schema violation** | JSON doğru biçim, yasak etiket/alan/tür içeriyor | EN032/EN039/EN042; yasak PRODUCT/DATE etiketleri |
| **Contextual ambiguity** | Apple/Orange gibi çokanlamlı ifadenin bağlamının yanlış yorumlanması | EN012, EN019 |
| **Morphological / span issue** | Türkçe eklerle varlık sınırlarının uyuşmaması | TR001: `Londra` / `Londra'da` |

**Neden halüsinasyon sayıları dikkat gerektiriyor?** Ana Accuracy deneyindeki **15 `ungrounded`** kaydın hepsinin yüzey metni kaynak paragrafta bulundu. Çoğu, tek bir kaynak spaninin tekrar çıkarılması veya aynı konuma iki kategori bağlanması nedeniyle eşleştirilememiştir; **15 halüsinasyon** denemez. Bunun aksine diğer koşullarda kaynak metinde hiç geçmeyen **dört** string görüldü: Few-shot `EN041` → `Carla Moreno`; Few-shot `EN049` → `Lima`; Temperature=1.0 üçüncü tekrar `EN050` → `the developer` ve `the owner`. Bunlar yüzey metni açısından kaynaktan desteklenmeyen üretimlerdir, yani **hallucinated/unsupported entity strings** olarak örneklendirilebilir. `the team → ORG` gibi örnekler ise metinde geçen ama yanlış kategorilenen **semantic over-extraction** hatalarıdır. Aynı metin/koşul tekrarları bağımsız istatistik değildir.

## 13. BERT — Token aggregation ablation (`simple` vs. `first`)

**Amaç:** Aynı `dslim/bert-base-NER` modelinin tahminleri sabitken tokenların varlık spanlerine birleştirilme stratejisi sonuçları nasıl değiştiriyor? **Her iki koşul da eğitim/fine-tuning içermeyen inference**.

| Kriter | `simple` | `first` | Fark |
|---|---:|---:|---:|
| Micro Precision | 0.8549 | **0.9617** | +0.1068 |
| Micro Recall | 0.9167 | **0.9778** | +0.0611 |
| Micro F1 | 0.8847 | **0.9697** | +0.0850 (8.50 yüzde puanı) |
| TP | 165 | 176 | +11 |
| FP | 28 | 7 | −21 |
| FN | 15 | 4 | −11 |
| En az bir hata içeren metin | 14/50 | **6/50** | −8 |

**Gözlem:** İki stratejinin tahmin listesi **10 metinde** farklı. Bunların **8'inde F1 yükseldi**, ikisinde (EN027, EN038) F1 aynı kaldı ama yanlış spanler değişti. 50 örneğin hiçbirinde gözlenen F1 düşüşü yok. Bu, **bu örneklemle sınırlı** bir bulgudur.

**Düzeltilen fragmentasyon örnekleri:**

| ID | `simple` çıktısı | `first` çıktısı | F1 |
|---|---|---|---|
| EN005 | `P` + `riya Shah` | `Priya Shah` | 0.400 → 1.000 |
| EN008 | `Am` + `ina Yusuf` | `Amina Yusuf` | 0.400 → 1.000 |
| EN023 | `Fat` + `ima Ali` | `Fatima Ali` | 0.667 → 1.000 |
| EN025 | `Ya` + `ra Haddad` | `Yara Haddad` | 0.667 → 1.000 |
| EN028 | `N` + `VIDIA` | `NVIDIA` | 0.727 → 1.000 |
| EN030 | `Is` + `la Campbell` | `Isla Campbell` | 0.727 → 1.000 |
| EN035 | `Amir` | `Amira` | 0.750 → 1.000 |
| EN037 | `Z` + `ara Ahmed`; `C` + `isco` | `Zara Ahmed`; `Cisco` | 0.400 → 1.000 |

**Olası mekanizma:** WordPiece alt token etiketleri BIO açısından uyumsuz sınırlarda gruplanıyor; `first` kelimedeki ilk alt token etiketiyle kelime bütünlüğünü koruyarak parçalamayı azaltıyor. Görülen fark BERT'in ağırlıklarının daha iyi öğrenmesi değil, **post-processing/aggregation** değişimidir.

**Önemli raporlama:** `first`, `simple` hataları test üzerinde görüldükten sonra seçildi. Sunumda `BERT (simple)=0.885`, `BERT (first; post-hoc)=0.970` birlikte gösterilmeli; sonuncusu bağımsız doğrulama gibi sunulmamalı.

## 14. BERT `first` — Kalan altı hatalı metnin tam değerlendirmesi

| Örnek | Gold / beklenen | `first` çıktısı | Hata türü | Açıklama |
|---|---|---|---|---|
| EN013 | `Manaus → LOC` | `Manaus → ORG` | Yanlış kategori | Coğrafi ad kurum olarak etiketlendi |
| EN026 | `Milan → PERSON` (ikinci anılış) | `Milan → ORG` | Yanlış kategori | Kişi referansı yanlış yorumlandı; `Milan` çokanlamlı ad |
| EN027 | `Coursera → ORG` | `Coursera → PERSON` | Yanlış kategori | Eğitim platformu kişi olarak yorumlandı |
| EN036 | Gold yalnız Adobe/San Jose/Miles Turner (çekirdek) | `UX → ORG`, `Photoshop → ORG` ekstra | Çekirdek etiket kapsamı dışı yanlış pozitifler | UX meslek ifadesinin parçası, Photoshop PRODUCT; yanlış ORG tahmini |
| EN038 | Gold Amazon/Boston/Caleb Young/Caleb | `Kindle → ORG` ekstra | Ürünü kurum sanma | Gold `Kindle Paperwhite → PRODUCT` çekirdek etiketlerde yok |
| EN048 | `Oxford → LOC` (şehir anılışı) | `Oxford → ORG` | Yanlış kategori / çokanlamlılık | `Oxford University` kurum, sonraki `Oxford` şehir |

**Sayım:** **4 yanlış kategori eşleşmesi** her biri 1 FP + 1 FN; **3 ekstra ORG tahmini** ayrıca FP. Toplam **TP176 / FP7 / FN4**. Bunlar altı hatalı test metnindeki tüm mevcut çekirdek NER hatalarıdır. `first` sonuçlarında kalan **tam-sınır fragmentasyon hatası yoktur**. `UX`, `Photoshop`, `Kindle` kaynakta bulunduğu için **uydurulmuş metin halüsinasyonu değildir**.

### 14.1 En öğretici üç-model karşılaştırmaları

| ID | spaCy | BERT `first` | Qwen | Öğrettiği hata |
|---|---|---|---|---|
| **EN013** | `Leo Grant` → ORG | `Manaus` → ORG | `retail corporation` → ORG ekstra; `Amazon River` eksik | Aynı kaynakta farklı kategori ve ekstra varlık hataları |
| **EN026** | `Milan Novak` → ORG; ikinci `Milan` → LOC | ikinci `Milan` → ORG | `Milan Novak` → PERSON doğru; ayrıca `Milan` → LOC ve ikinci anılışı eksik | Aynı adın şehir/kişi karışıklığı |
| **EN027** | `Coursera` → PERSON | `Coursera` → PERSON | `Coursera` → ORG doğru; `June 2025` → LOC ekstra | Qwen tekil bir kurumu daha iyi sınıflandırırken tarihte yanlış pozitif üretiyor |
| **EN036** | Üç çekirdek varlık tam doğru | `UX`, `Photoshop` → ORG ekstra | `Photoshop` → ORG; `the organizers` → PERSON ekstra | Ürün/meslek/ortak isim yanlış pozitifleri |
| **EN038** | `Caleb Young` → ORG; ürün grubu → ORG | `Kindle` → ORG ekstra | `Caleb Young` doğru + `Caleb` örtüşmesi; `Kindle Paperwhite` → ORG ekstra | Ürün adını kurum sanma, iç içe/tekrarlı ad tahminleri |
| **EN048** | İkinci `Oxford` → ORG | İkinci `Oxford` → ORG | İlk `Oxford` → LOC (üniversite adının bir kısmı); ikinci şehir anılışı eksik | Üniversite adı/şehir adı bağlamında sınır ve tür sorunu |

**Ortak hata kümesi:** BERT `first` ile spaCy'nin birlikte hata yaptığı örnekler EN013, EN026, EN027, EN038, EN048 (**5 metin**). BERT `first` ile Qwen'in birlikte hata yaptığı **6 metnin hepsi** EN013, EN026, EN027, EN036, EN038 ve EN048. **Üç modelin aynı anda hata yaptığı beş metin** EN013, EN026, EN027, EN038, EN048. Bu ortak hatalar her modelin birebir aynı tahmini yaptığını göstermez.

## 15. Entity türüne göre ana Accuracy karşılaştırması

50 İngilizce metindeki çekirdek etiketler için ayrı ayrı exact-span entity değerlendirmesi. Qwen'in yüzey eşleyicisinin `ungrounded` olarak cezalandırdığı çıktıların FP katkısı, çıktı etiketine dağıtılmıştır.

| Model | Kişi (PERSON) F1 | Kurum (ORG) F1 | Konum (LOC) F1 |
|---|---:|---:|---:|
| spaCy | 0.8777 | 0.7925 | 0.9174 |
| **BERT `first`** | **0.9868** | **0.9346** | **0.9808** |
| Qwen | 0.6395 | 0.8065 | 0.8929 |

**Yorum:** Qwen kişi çıkarımında daha fazla parçalanmış/tekrarlı veya ekstra tahmin üretmiş. ORG ve LOC çekirdek kategorilerinde Qwen, spaCy'ye göre bazı örneklerde yaklaştı veya ORG etiketinde az farkla üstünlük sağladı, fakat tüm sınıfların mikro birleşiminde spaCy daha iyi. BERT `first` üç çekirdek türde de en yüksek F1'i verdi. Bu sonuçlar **50 sentetik test metninin** bu etiket setine özgüdür.

## 16. Yapısal hatalar, halüsinasyon ve tekrarlar için kesin ayrım

**Ana Accuracy, Qwen:** `ungrounded_total=15`. Ham kayıtların kaynak metni incelendiğinde **15 ifadenin de kaynakta en az bir kez geçtiği** görüldü. Bunlar kaynağı olmayan 15 isim değil; yinelenen çıkarım / aynı span'e tekrar eşleme / yanlış türle aynı span'i tüketme türündeki **surface alignment** sorunlarıdır. Örnek EN001'de `Google`, `London`, `Google` tekrarları; EN050'de aynı `Berlin` span'i önce PERSON, sonra LOC olarak kullanılması.

**Doğrudan kaynakta bulunmayan stringler, kaydedilmiş diğer Qwen denemelerinde:**

| Deney/koşul | Örnek | Ham modelin ürettiği ama kaynakta olmayan metin | Sınıf |
|---|---|---|---|
| Few-shot / three-shot | EN041 | `Carla Moreno` | PERSON |
| Few-shot / three-shot | EN049 | `Lima` | LOC |
| Temperature=1.0 / üçüncü tekrar | EN050 | `the developer` | PERSON |
| Temperature=1.0 / üçüncü tekrar | EN050 | `the owner` | PERSON |

Bunlar **gerçek kaynaktan desteklenmeyen string üretimi** örnekleridir ve halüsinasyona daha yakın kanıttır. Buna karşılık `the team → ORG`, `everyone → PERSON`, `Photoshop → ORG` gibi örnekler **metinde gerçekten var olan sözcüklerin varlık türüyle yanlış etiketlenmesidir** (*spurious/over-extraction*). Kaynakta bulunmayan string sayısı, tüm modeli temsil eden genel bir halüsinasyon oranına çevrilmemelidir: farklı deneyler aynı metinler üzerinde tekrarlar içerir.

**Schema kontrolü:** Prompt-only koşulunda JSON sözdizimi geçerliliği **50/50**, ancak şema uygunluğu **47/50**. Kaynak dışı etiket üreten üç örnek: **EN032 (PRODUCT/DATE), EN039 (PRODUCT), EN042 (DATE)**. Bu etiketler ana PERSON/ORG/LOC izinli enum'unda olmadığından bütün çıktılar reddedilmiş, her biri F1=0 hesaplanmıştır. Schema-constrained koşulunda sözdizimi ve şema geçerliliği **50/50**. Bu, **JSON geçerliliğinin semantik doğrulukla aynı şey olmadığını** kanıtlar.

## 17. Sonuçların sınırları ve bilimsel doğruluk kontrolleri

1. **Sentetik veri / düşük n:** 50 İngilizce ve 10 ek Türkçe metin; gerçek haber, medikal veya hukuki içeriklere performans genellemesi yapılamaz. Etiketler elle gözden geçirilmelidir.
2. **BERT için test üzerinde seçim:** `first`, `simple` hataları gözlendikten sonra kullanıldı. Yeni bir gizli test kümesi veya yalnızca development kümesi üzerinde seçim yapılıp sonra yeni test gerekir.
3. **Etiketler sabit:** P/R/F1 yalnızca `PERSON/ORG/LOC`; Qwen esneklik deneyindeki `JOB_TITLE` bu ana puanlara eklenmemeli. spaCy/BERT'in hazır kontrol noktalarının JOB_TITLE sunmaması, bütün NER modellerinin buna kategorik olarak imkânsız olduğu anlamına gelmez.
4. **İlişki çıkarımı ölçülmedi:** `Tim Cook → CEO of → Apple` gibi ilişkiler doğrudan gold şemasında bulunmuyor. Bu deneyler varlık çıkarımını ölçer, genel relation extraction kalitesini değil.
5. **JSON ile NER API farklı:** spaCy/BERT doğrudan varlık nesneleri çıkarıyor; bunlara “JSON Schema validity=0” vermek anlamsız. Bu metrik `null`/uygulanmaz.
6. **Süre yöntemleri:** spaCy CPU, BERT CPU/MPS, Qwen Ollama farklı backend'leri; sıcaklık, prompt, warm-up, sistem yükü ve tek tekrar süreleri etkiler. Tek koşuldan donanımlar arası sabit hız oranı ilan edilmemeli.
7. **Qwen `ungrounded` ayırımı:** Dedup/span eşleyici kaynakta geçen tekrarı bile FP sayar. Bu F1 hesaplama yönteminin tasarım tercihi; BERT ve spaCy'deki tekrar işlemi farklıdır. Adil kıyas için ölçüt ve tekrar cezaları açıklanmalı.
8. **`Unseen` deney adı:** Challenging alt grup ana testin içindedir, bağımsız yeni benchmark değildir. Aynı 10 metin toplam 50 içinde tekrar yer alır.
9. **Tekrar bağımsızlığı:** Temperature 300 çağrı = 300 bağımsız test cümlesi değildir; aynı 50 cümlenin tekrarlarıdır.
10. **Quantization ölçülmedi:** Ollama `ReadTimeout` nedeniyle Q4/Q8 kıyaslaması tamamlanmadı. **Hiçbir Q4/Q8 doğruluk/hız sonucu raporlanmaz.**

## 18. Sunum için seçilebilecek sonuçlar (15 dakikalık konuşma)

**Ana görsel 1:** 50 İngilizce metinde BERT `first` (0.970, post-hoc), BERT `simple` (0.885), spaCy (0.864), Qwen schema-constrained (0.768) bar grafiği.

**Ana görsel 2:** BERT aggregasyon örneği EN008: `Am` + `ina Yusuf` → `Amina Yusuf`; F1 0.400 → 1.000 (sadece EN008).

**Ana görsel 3:** JSON Reliability: prompt-only %94 schema geçerliliği/F1 0.675; schema-constrained %100/F1 0.768; ham JSON her iki koşulda %100.

**Ana görsel 4:** Qwen hata türleri: yanlış kategori, boundary, tekrar, kaynağa dayanmayan string (ör. `Carla Moreno`); sonuncunun ana Accuracy'de değil Few-shot içinde meydana geldiğini açıkça belirt.

**İngilizce konuşma özeti:**

> We evaluated task-specific NER models and an instruction-following LLM on 50 synthetic English passages using exact-span entity-level micro-F1. Our observed scores were 0.864 for spaCy and 0.768 for Qwen. BERT scored 0.885 with simple aggregation, improving to 0.970 after a post-hoc change to first-token aggregation. The main failure modes differed: BERT showed token fragmentation before aggregation adjustment; spaCy showed entity-type confusions; Qwen frequently over-extracted or repeated entities. Prompt-only JSON was always syntactically valid but failed our schema on three samples. These results are preliminary because the dataset is small and synthetic, and the BERT aggregation strategy was selected after inspecting test errors.

## Ek A. 50 test metninin tamamında bireysel F1 karşılaştırması

Bu tablo **aynı 50 metnin** üç model ve iki BERT aggregation stratejisi altında sonuçlarını listeler. Her hücre, ilgili metnin entity-level F1 değeridir (küme genelindeki micro-F1 ile karıştırılmamalıdır). Alt tablodaki **hata** işareti F1'in 1'den küçük olduğu metindir.

| ID | Grup | spaCy F1 | BERT `simple` F1 | BERT `first` F1 | Qwen F1 |
|---|---|---:|---:|---:|---:|
| EN001 | standard | 1.000 | 1.000 | 1.000 | 0.500 |
| EN002 | standard | 1.000 | 1.000 | 1.000 | 0.571 |
| EN003 | standard | 1.000 | 1.000 | 1.000 | 1.000 |
| EN004 | standard | 1.000 | 1.000 | 1.000 | 0.857 |
| EN005 | standard | 0.857 | 0.400 | 1.000 | 0.571 |
| EN006 | standard | 0.857 | 1.000 | 1.000 | 0.750 |
| EN007 | standard | 0.500 | 1.000 | 1.000 | 0.857 |
| EN008 | standard | 0.750 | 0.400 | 1.000 | 0.750 |
| EN009 | standard | 1.000 | 1.000 | 1.000 | 0.857 |
| EN010 | standard | 0.750 | 1.000 | 1.000 | 0.571 |
| EN011 | ambiguity | 0.857 | 1.000 | 1.000 | 0.857 |
| EN012 | ambiguity | 1.000 | 1.000 | 1.000 | 0.500 |
| EN013 | ambiguity | 0.750 | 0.750 | 0.750 | 0.500 |
| EN014 | ambiguity | 0.667 | 1.000 | 1.000 | 1.000 |
| EN015 | ambiguity | 1.000 | 1.000 | 1.000 | 0.571 |
| EN016 | ambiguity | 1.000 | 1.000 | 1.000 | 0.750 |
| EN017 | ambiguity | 1.000 | 1.000 | 1.000 | 1.000 |
| EN018 | ambiguity | 1.000 | 1.000 | 1.000 | 0.800 |
| EN019 | ambiguity | 0.333 | 1.000 | 1.000 | 0.750 |
| EN020 | ambiguity | 0.500 | 1.000 | 1.000 | 0.667 |
| EN021 | multi_entity | 1.000 | 1.000 | 1.000 | 0.909 |
| EN022 | multi_entity | 1.000 | 1.000 | 1.000 | 0.889 |
| EN023 | multi_entity | 1.000 | 0.667 | 1.000 | 1.000 |
| EN024 | multi_entity | 0.750 | 1.000 | 1.000 | 1.000 |
| EN025 | multi_entity | 1.000 | 0.667 | 1.000 | 1.000 |
| EN026 | multi_entity | 0.600 | 0.800 | 0.800 | 0.800 |
| EN027 | multi_entity | 0.750 | 0.750 | 0.750 | 0.889 |
| EN028 | multi_entity | 0.889 | 0.727 | 1.000 | 0.545 |
| EN029 | multi_entity | 1.000 | 1.000 | 1.000 | 1.000 |
| EN030 | multi_entity | 0.800 | 0.727 | 1.000 | 0.909 |
| EN031 | extended | 1.000 | 1.000 | 1.000 | 1.000 |
| EN032 | extended | 1.000 | 1.000 | 1.000 | 0.750 |
| EN033 | extended | 0.857 | 1.000 | 1.000 | 0.571 |
| EN034 | extended | 1.000 | 1.000 | 1.000 | 0.750 |
| EN035 | extended | 0.750 | 0.750 | 1.000 | 0.667 |
| EN036 | extended | 1.000 | 0.750 | 0.750 | 0.750 |
| EN037 | extended | 0.667 | 0.400 | 1.000 | 0.750 |
| EN038 | extended | 0.500 | 0.889 | 0.889 | 0.667 |
| EN039 | extended | 0.889 | 1.000 | 1.000 | 0.667 |
| EN040 | extended | 0.857 | 1.000 | 1.000 | 0.667 |
| EN041 | challenging | 1.000 | 1.000 | 1.000 | 0.667 |
| EN042 | challenging | 1.000 | 1.000 | 1.000 | 0.857 |
| EN043 | challenging | 1.000 | 1.000 | 1.000 | 1.000 |
| EN044 | challenging | 1.000 | 1.000 | 1.000 | 0.800 |
| EN045 | challenging | 1.000 | 1.000 | 1.000 | 1.000 |
| EN046 | challenging | 1.000 | 1.000 | 1.000 | 1.000 |
| EN047 | challenging | 0.667 | 1.000 | 1.000 | 0.750 |
| EN048 | challenging | 0.667 | 0.667 | 0.667 | 0.667 |
| EN049 | challenging | 1.000 | 1.000 | 1.000 | 0.857 |
| EN050 | challenging | 1.000 | 1.000 | 1.000 | 0.000 |

## Ek B. Yeniden üretilebilirlik ve dosya adları

- **Orijinal dokuz deney:** `results.zip` → `results/{accuracy,flexibility,ambiguity,json,efficiency,temperature,fewshot,language,unseen}_{summary.json,detail.csv}`.
- **BERT `simple`:** `bert_analysis.zip` → `results/bert/bert_cpu_detail.csv`, `bert_mps_detail.csv`, `bert_mac_summary.json`.
- **BERT `first`:** `bert_analysis.zip` → `results/bert_first/bert_cpu_detail.csv`, `bert_mps_detail.csv`, `bert_mac_summary.json`, CPU/MPS özetleri.
- **Veri:** `english_test.jsonl` → 50 metin, `entities` altın standart PERSON/ORG/LOC, `extended_entities` ayrı DATE/JOB_TITLE/PRODUCT.
- **Değerlendirme:** etiket ve hem `start` hem `end` koordinatı eşleşmedikçe TP sayılmaz. Bir yanlış etiket aynı anda **1 FP + 1 FN** üretebilir. Bazı yüzey tekrarları Qwen için ayrıca FP sayılmıştır.
- **Şema alanları:** `{"entities":[{"text":"...","label":"PERSON|ORG|LOC"}]}` (gerçek uygulama şeması); modelin ham yanıtları proje sonuçlarındaki `raw_model_response` alanında bulunur.

**Son söz:** BERT `first` en yüksek gözlenen F1'i sağlamıştır, ancak bu strateji aynı test seti üzerinden seçilmiştir. Raporun sayıları yalnızca belirtilen denemeler ve veri için geçerlidir. Kesin genellenebilir model sıralaması elde etmek için daha büyük, gerçek, bağımsız ve elle doğrulanmış bir test kümesi gerekir.
