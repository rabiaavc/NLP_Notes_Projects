"""Rebuild synthetic, human-reviewable gold JSONL records from inline annotations.
Markers: [[PERSON|Sarah Lee]], [[ORG|Google]], [[LOC|London]], ...
All examples are fictitious teaching scenarios, not factual news claims.
"""
import json
import re
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
PATTERN=re.compile(r"\[\[(PERSON|ORG|LOC|DATE|JOB_TITLE|PRODUCT)\|([^\[\]]+)\]\]")
GROUPS={}

def group(name, texts):
    assert name not in GROUPS
    GROUPS[name]=[x.strip() for x in texts]

# 50 English test passages: 3-4 sentences and 35+ words per passage.
group('standard', [
"""[[PERSON|Sarah Johnson]] arrived in [[LOC|London]] to meet engineers from [[ORG|Google]]. The team reviewed a demonstration scheduled for [[DATE|March 12, 2025]] and discussed how to document unusual customer questions. Sarah asked everyone to record the decisions before leaving the office.""",
"""At a conference in [[LOC|Berlin]], [[PERSON|Daniel Brooks]] introduced a small research project supported by [[ORG|Microsoft]]. His presentation included an update from [[DATE|2024]] and several examples of software failures. After the session, Daniel answered questions about the testing process rather than promoting a specific product.""",
"""[[PERSON|Emma Rivera]] prepared a training workshop for [[ORG|IBM]] in [[LOC|Toronto]]. The agenda for [[DATE|June 3, 2025]] included a discussion of missing data and a brief hands-on exercise. Participants were encouraged to document assumptions before choosing a machine learning method.""",
"""The local design team invited [[PERSON|Noah Carter]] to the [[ORG|Adobe]] office in [[LOC|Seattle]]. During the [[DATE|2023]] meeting, they reviewed an unfinished prototype and several usability notes. Noah suggested adding a simple summary so that new employees could understand the project without a long briefing.""",
"""[[PERSON|Priya Shah]] visited [[ORG|Samsung]] representatives in [[LOC|Seoul]] to discuss a student showcase. The proposal, originally drafted in [[DATE|2025]], described a lightweight evaluation process and a short public demonstration. Priya later sent a message explaining which measurements should be collected during the event.""",
"""A university research group in [[LOC|Boston]] hosted [[PERSON|Lucas Miller]] for a seminar organized with [[ORG|Amazon]]. The session on [[DATE|April 8, 2024]] focused on reliability in information systems. Lucas compared simple rule-based methods with models trained on labeled examples, then invited attendees to challenge his assumptions.""",
"""[[ORG|Netflix]] invited [[PERSON|Olivia Perez]] to a meeting in [[LOC|Los Angeles]]. The fictional case study referred to subscriber feedback recorded in [[DATE|2022]] and asked participants to identify recurring complaints. Olivia explained why conclusions based on only a handful of observations could be misleading.""",
"""[[PERSON|Amina Yusuf]] coordinated a volunteer technology class with [[ORG|UNICEF]] in [[LOC|Nairobi]]. The planning notes mentioned [[DATE|September 2025]] as a tentative start date. During the preparation call, Amina emphasized that the materials should be accessible to learners using older devices.""",
"""[[ORG|Spotify]] hosted a small design sprint in [[LOC|Stockholm]] with guest speaker [[PERSON|Liam Evans]]. The fictional project had first been proposed in [[DATE|2024]]. Liam asked the group to distinguish observed user behavior from assumptions made by the product team.""",
"""[[PERSON|Chloe Martin]] attended a demonstration at [[ORG|Toyota]] in [[LOC|Tokyo]]. The agenda for [[DATE|February 2025]] included sensor calibration, dashboard design, and privacy considerations. Chloe proposed a follow-up experiment that would compare different alert messages without changing the underlying data."""
])

group('ambiguity', [
"""[[ORG|Apple]] invited [[PERSON|Nina Brooks]] to a product briefing in [[LOC|Cupertino]]. The team discussed hardware reliability and mentioned plans from [[DATE|2025]], but no device was demonstrated at the meeting. Nina later wrote a report about how easily a familiar brand name can influence expectations.""",
"""During lunch in [[LOC|Paris]], [[PERSON|Nina Brooks]] sliced an apple while reading a magazine. The article described a workshop that would begin in [[DATE|May 2025]]. No technology company was involved in the lunch, and the fruit was simply part of her meal.""",
"""The [[LOC|Amazon River]] appeared in a geography lesson presented by [[PERSON|Leo Grant]] in [[LOC|Manaus]]. Students reviewed a map drawn in [[DATE|2020]] and compared several waterways. Leo reminded the class that a river name should not be confused with the name of a retail corporation.""",
"""[[ORG|Amazon]] invited [[PERSON|Leo Grant]] to a logistics exercise in [[LOC|Dublin]]. The team used mock orders created in [[DATE|2024]] to test a routing system. They discussed how an organization name can have the same surface form as a famous geographic feature.""",
"""[[PERSON|Jordan Blake]] joined a design meeting hosted by [[ORG|Intel]] in [[LOC|Austin]]. The meeting notes from [[DATE|2025]] mention a new testing checklist and several open questions. Jordan volunteered to review the checklist before the next software release.""",
"""A student group traveled to [[LOC|Jordan]] to participate in a workshop in [[LOC|Amman]]. Their coordinator, [[PERSON|Maya Collins]], circulated the schedule for [[DATE|July 2025]]. The discussion included how the same word can denote a person in one sentence and a country in another.""",
"""[[PERSON|May Chen]] gave a short talk at the [[ORG|Meta]] office in [[LOC|New York]]. Her slides summarized feedback received during [[DATE|2024]]. Although her first name is also a month, the sentence uses May to refer to the speaker rather than to a date.""",
"""The museum planned to reopen in [[DATE|May 2026]], according to a fictional notice displayed in [[LOC|Oxford]]. [[PERSON|Helen Parker]] checked the notice for ambiguous wording. Here, May names a month, and it does not refer to a person attending the event.""",
"""[[ORG|Orange]] sponsored a small fictional connectivity workshop in [[LOC|Lyon]]. [[PERSON|Theo Lambert]] reviewed the agenda in [[DATE|2024]] and suggested a simple network diagram. In this passage, [[ORG|Orange]] is presented as an organization rather than a piece of fruit.""",
"""[[PERSON|Theo Lambert]] bought an orange at a market in [[LOC|Lyon]] before attending a workshop. His notes were dated [[DATE|2024]], but the receipt mentioned no company or institution. The word orange describes food and should not be marked as an organization."""
])

group('multi_entity', [
"""[[PERSON|Aria Wilson]] and [[PERSON|Ethan Reed]] met representatives from [[ORG|Google]] and [[ORG|Microsoft]] in [[LOC|San Francisco]]. Their joint workshop was scheduled for [[DATE|October 2025]]. The group compared two ways of handling user feedback and agreed to circulate a neutral summary after the meeting.""",
"""Researchers from [[ORG|Stanford University]] and [[ORG|Oxford University]] invited [[PERSON|Owen Smith]] to a seminar in [[LOC|London]]. The fictional invitation referred to findings collected in [[DATE|2023]]. Owen asked for additional examples because the first draft appeared to overlook several edge cases.""",
"""[[PERSON|Fatima Ali]] wrote to [[PERSON|Grace Kim]] after a conference hosted by [[ORG|IBM]] in [[LOC|Toronto]]. Their message cited a discussion from [[DATE|April 2025]] about transparent evaluation. They agreed to record not just the final scores but also the mistakes found in each experiment.""",
"""The fictional partnership between [[ORG|Adobe]] and [[ORG|Spotify]] brought [[PERSON|Marta Silva]] to [[LOC|Barcelona]]. During the [[DATE|2024]] planning session, she described a new process for documenting user interviews. Representatives from both organizations asked for a shared set of definitions before collecting data.""",
"""[[PERSON|Yara Haddad]] and [[PERSON|Ben Thompson]] discussed a test plan with [[ORG|Toyota]] engineers in [[LOC|Osaka]]. The meeting took place in [[DATE|2025]] as part of a classroom case study. The participants disagreed about whether a faster algorithm would necessarily produce a more reliable prediction.""",
"""At an imaginary workshop in [[LOC|Rome]], [[PERSON|Milan Novak]] welcomed delegates from [[ORG|UNESCO]] and [[ORG|UNICEF]]. The event agenda, dated [[DATE|2024]], included digital literacy and accessibility. Milan asked participants to distinguish the goals of the organizations instead of treating them as interchangeable.""",
"""[[PERSON|Sofia Adams]] visited [[LOC|Madrid]] and [[LOC|Lisbon]] while researching a teaching exercise for [[ORG|Coursera]]. Her diary from [[DATE|June 2025]] described how location names appear in travel narratives. She noted that a sentence may contain several valid places without mentioning an organization more than once.""",
"""[[ORG|NVIDIA]] and [[ORG|Intel]] organized a fictional hardware discussion in [[LOC|Berlin]], where [[PERSON|Jack Morgan]] moderated the questions. The seminar was advertised for [[DATE|November 2025]]. Jack asked the speakers to avoid claiming performance improvements without measurements from the same testing conditions.""",
"""[[PERSON|Hannah Scott]] spoke with [[PERSON|Diego Ruiz]] after a training session at [[ORG|Salesforce]] in [[LOC|Chicago]]. Their notes from [[DATE|2025]] listed multiple sources of annotation disagreement. The two reviewers decided to resolve boundary cases before either model was allowed to see the evaluation set.""",
"""A public lecture in [[LOC|Edinburgh]] featured [[PERSON|Isla Campbell]] from [[ORG|BBC]] and [[PERSON|Ravi Kapoor]] from [[ORG|Reuters]]. The fictional lecture took place in [[DATE|2024]] and examined automated summaries. Both speakers warned that correct formatting does not ensure factual accuracy."""
])

group('extended', [
"""[[PERSON|Evelyn Ross]] works as a [[JOB_TITLE|data scientist]] at [[ORG|Google]] in [[LOC|London]]. In a fictional report dated [[DATE|2025]], she compared model predictions with a manually checked dataset. The title describes her occupation, not a person or organization name.""",
"""[[ORG|Apple]] asked [[PERSON|Ryan Foster]], a [[JOB_TITLE|product designer]] in [[LOC|Cupertino]], to review the [[PRODUCT|iPhone 16]] interface. The classroom exercise was set in [[DATE|2025]]. Ryan recorded usability observations and emphasized that the product name should be distinguished from the company producing it.""",
"""[[PERSON|Lena Ortiz]] joined [[ORG|Microsoft]] as a [[JOB_TITLE|software engineer]] in [[LOC|Seattle]]. The fictional personnel announcement appeared in [[DATE|2024]] and mentioned an internal accessibility project. Lena later prepared a guide explaining the difference between an employee's name and the job title assigned to that employee.""",
"""[[ORG|Samsung]] invited [[PERSON|Daniel Cho]], a [[JOB_TITLE|research analyst]], to test the [[PRODUCT|Galaxy S25]] during a workshop in [[LOC|Seoul]]. The exercise used mock observations collected in [[DATE|2025]]. Daniel was asked to record the device name exactly rather than paraphrase it as a generic smartphone.""",
"""[[PERSON|Amira Patel]], a [[JOB_TITLE|project manager]], coordinated a fictional launch meeting for [[ORG|Sony]] in [[LOC|Tokyo]]. The agenda listed the [[PRODUCT|PlayStation 5]] and a follow-up review in [[DATE|2025]]. Amira requested that product names and occupational titles be stored in separate fields in the final report.""",
"""At [[ORG|Adobe]] in [[LOC|San Jose]], [[PERSON|Miles Turner]] introduced himself as a [[JOB_TITLE|UX researcher]]. He presented a fictitious accessibility study from [[DATE|2024]] involving [[PRODUCT|Photoshop]]. The participants debated whether a generic phrase like design software should ever be labeled as a named product.""",
"""[[PERSON|Zara Ahmed]] served as a [[JOB_TITLE|security engineer]] on a classroom scenario involving [[ORG|Cisco]] in [[LOC|Austin]]. Her fictional analysis from [[DATE|2023]] discussed the [[PRODUCT|Webex]] application and its configuration. Zara asked reviewers to mark only information explicitly mentioned in the source text.""",
"""[[ORG|Amazon]] hosted a fictional operations meeting in [[LOC|Boston]], where [[PERSON|Caleb Young]] worked as a [[JOB_TITLE|logistics coordinator]]. The case study referenced the [[PRODUCT|Kindle Paperwhite]] in [[DATE|2025]]. Caleb pointed out that a retail brand, a product, and an occupation represent different information types.""",
"""[[PERSON|Mina Hassan]], a [[JOB_TITLE|marketing specialist]] at [[ORG|Spotify]], proposed a classroom campaign for [[LOC|Stockholm]]. The idea was first drafted in [[DATE|2024]] and mentioned a feature called [[PRODUCT|Spotify Wrapped]]. Mina wanted to extract the campaign date and product name without creating facts absent from the notes.""",
"""[[ORG|Tesla]] welcomed [[PERSON|Oliver Grant]], a [[JOB_TITLE|systems engineer]], to a fictional meeting in [[LOC|Berlin]]. The discussion referenced the [[PRODUCT|Model 3]] and a test cycle planned for [[DATE|2025]]. Oliver requested separate evaluation of product recognition, organization recognition, and job-title extraction."""
])

group('challenging', [
"""The report delivered to [[ORG|Google]] mentioned [[PERSON|Dr. Alice Monroe]] only in a footnote, while the primary meeting took place in [[LOC|New York]]. A second paragraph referred to [[DATE|2024]] without specifying an exact day. Reviewers were instructed to exclude the honorific and retain only the full person's name.""",
"""A fictional notice from [[ORG|IBM]] described an appointment in [[LOC|Paris]], not a permanent transfer. [[PERSON|Evan Lee]] checked the draft on [[DATE|August 4, 2025]] and noticed that the location was repeated in a later sentence. The extraction system should report both mentions if the same name appears twice.""",
"""[[PERSON|Nora Kim]] attended a workshop organized by [[ORG|Microsoft]] near [[LOC|Cambridge]]. The notes from [[DATE|2025]] said no product was announced, despite several requests for a device name. A model should not invent a product simply because the organizer is known for commercial software.""",
"""Although the team discussed possible hiring, no person was identified by name in the memo from [[ORG|Adobe]]. The fictional document was circulated in [[LOC|Berlin]] during [[DATE|2024]]. Reviewers should return an empty person list rather than guess which employee might have attended.""",
"""A student described the difference between visiting [[LOC|Washington]] and working for [[ORG|Washington University]] in a fictional class report. [[PERSON|Leo Harris]] checked the examples in [[DATE|2025]]. The two names overlap in wording but refer to different kinds of entities in their respective contexts.""",
"""[[PERSON|Maya Roberts]] said that [[ORG|Apple]] had not approved the proposed software demonstration in [[LOC|London]]. The fictional meeting was recorded in [[DATE|2025]]. Entity recognition should still detect the organization even though the sentence expresses a negative statement about its decision.""",
"""The meeting at [[ORG|Netflix]] was postponed, according to an email written by [[PERSON|Samuel White]] in [[LOC|Dublin]]. The original date was [[DATE|June 11, 2025]], but the message provided no replacement date. A system must not generate a new date that the text never states.""",
"""[[ORG|Oxford University]] asked [[PERSON|Sana Malik]] to review an application submitted from [[LOC|Oxford]]. Her fictional notes were dated [[DATE|2024]] and contained both a university name and a city with a shared word. The annotation should keep those two spans separate rather than merge them.""",
"""[[PERSON|Ava Lee]] was preparing a case study about [[LOC|Georgia]], a country, for a fictional event hosted by [[ORG|UNESCO]]. Her presentation referred to travel notes from [[DATE|2025]]. The place name could also describe a region elsewhere, so the intended sense must come from the context.""",
"""The fictional audit described a failed prototype without naming its developer or owner. It mentioned [[LOC|Berlin]] as the testing location and [[DATE|2023]] as the collection year, but no organization was identified. The correct annotation should leave the organization field empty despite the technical vocabulary."""
])

# Turkish supplementary examples matched semantically (first ten English test stories).
TR=[
"""[[PERSON|Sarah Johnson]], [[LOC|Londra]]'da [[ORG|Google]] mühendisleriyle buluşmak için şehre geldi. Ekip, [[DATE|12 Mart 2025]] tarihine planlanan gösterimi ve müşterilerden gelebilecek sıra dışı soruları görüştü. Sarah, toplantı bitmeden önce alınan kararların bir dosyaya yazılmasını istedi.""",
"""[[PERSON|Daniel Brooks]], [[LOC|Berlin]]'de düzenlenen bir konferansta [[ORG|Microsoft]] tarafından desteklenen küçük bir araştırma çalışmasını tanıttı. Sunumunda [[DATE|2024]] yılına ait bir değerlendirmeye ve birkaç yazılım hatasına yer verdi. Daha sonra katılımcıların test süreciyle ilgili sorularını yanıtladı.""",
"""[[PERSON|Emma Rivera]], [[LOC|Toronto]]'da [[ORG|IBM]] için bir eğitim atölyesi hazırladı. [[DATE|3 Haziran 2025]] tarihli programda eksik veriler ve kısa bir uygulama vardı. Katılımcılara makine öğrenmesi yöntemi seçmeden önce varsayımlarını yazmaları önerildi.""",
"""[[LOC|Seattle]]'daki tasarım ekibi, [[PERSON|Noah Carter]]'ı [[ORG|Adobe]] ofisine davet etti. [[DATE|2023]] yılındaki toplantıda tamamlanmamış bir prototip ve kullanılabilirlik notları incelendi. Noah, yeni çalışanların projeyi daha kolay anlayabilmesi için kısa bir özet hazırlanmasını önerdi.""",
"""[[PERSON|Priya Shah]], [[LOC|Seul]]'de [[ORG|Samsung]] temsilcileriyle bir öğrenci etkinliğini görüştü. [[DATE|2025]] yılında hazırlanan taslakta küçük ölçekli bir değerlendirme süreci ve katılımcı geri bildirimlerinin toplanması öneriliyordu. Priya, etkinlik sırasında hangi ölçümlerin toplanacağını açıklayan bir mesaj gönderdi.""",
"""[[LOC|Boston]]'daki üniversite araştırma grubu, [[ORG|Amazon]] ile düzenlenen seminere [[PERSON|Lucas Miller]]'ı davet etti. [[DATE|8 Nisan 2024]] tarihli oturum bilgi sistemlerinde güvenilirlik hakkındaydı. Lucas, kural tabanlı yöntemlerle etiketli veriyle eğitilmiş modelleri karşılaştırdı ve katılımcılardan varsayımlarını sorgulamalarını istedi.""",
"""[[ORG|Netflix]], [[LOC|Los Angeles]]'ta [[PERSON|Olivia Perez]]'in katıldığı kurgusal bir toplantı düzenledi. Çalışmada [[DATE|2022]] yılında kaydedilmiş abonelik geri bildirimleri ve çalışma ekibinin hazırladığı kısa örnekler incelendi. Olivia, yalnızca birkaç gözleme dayanarak yapılan genellemelerin yanıltıcı olabileceğini anlattı.""",
"""[[PERSON|Amina Yusuf]], [[LOC|Nairobi]]'de [[ORG|UNICEF]] ile bir teknoloji sınıfı planladı. Hazırlık notlarında olası başlangıç zamanı olarak [[DATE|Eylül 2025]] belirtiliyordu. Amina, eğitim malzemelerinin eski cihazları kullanan öğrenciler için de erişilebilir olması gerektiğini vurguladı.""",
"""[[ORG|Spotify]], [[LOC|Stockholm]]'de konuk konuşmacı [[PERSON|Liam Evans]]'ın da katıldığı bir tasarım çalışması düzenledi. Kurgusal proje ilk kez [[DATE|2024]] yılında önerilmişti ve amaç kullanıcı beklentilerini daha doğru anlamaktı. Liam, ekipten gözlemlenen kullanıcı davranışı ile varsayımları birbirinden ayırmasını istedi.""",
"""[[PERSON|Chloe Martin]], [[LOC|Tokyo]]'da [[ORG|Toyota]] tarafından düzenlenen bir gösterime katıldı. [[DATE|Şubat 2025]] programında sensör kalibrasyonu, gösterge paneli tasarımı ve gizlilik konuları vardı. Chloe, temel veriyi değiştirmeden farklı uyarı mesajlarını karşılaştıran bir deney önerdi."""
]

DEV_EN=[
"""[[PERSON|Rachel Green]] visited [[ORG|IBM]] in [[LOC|Boston]] to discuss a research workshop. Her notes from [[DATE|2024]] included suggestions for clearer technical writing. She proposed a short evaluation before extending the exercise to a larger group of participants.""",
"""[[ORG|Apple]] invited [[PERSON|Ben Miller]], a [[JOB_TITLE|software tester]], to review a mock demonstration in [[LOC|Seattle]]. The fictional case study was dated [[DATE|2025]] and concerned the [[PRODUCT|iPad Air]]. Ben recorded both positive observations and possible user misunderstandings.""",
"""[[PERSON|Carla Moreno]] read about the [[LOC|Amazon River]] before a classroom lesson in [[LOC|Lima]]. The map was printed in [[DATE|2022]] and did not describe a retail business. Carla told her classmates that context should decide the correct entity category.""",
"""[[PERSON|David Lane]] met [[PERSON|Mina Cole]] during a fictional conference organized by [[ORG|Adobe]] in [[LOC|Paris]]. They discussed evaluation criteria and reviewed an agenda created in [[DATE|2024]]. Neither speaker made any claim about product availability or sales results.""",
"""The fictional memo from [[ORG|Microsoft]] was published in [[DATE|2025]] after a workshop in [[LOC|Dublin]]. It described an engineering problem but named no employees. Reviewers should not guess a person merely because a company is mentioned."""
]
DEV_TR=[
"""[[PERSON|Rachel Green]], [[LOC|Boston]]'da [[ORG|IBM]] ile bir araştırma atölyesini görüştü. [[DATE|2024]] yılına ait notlarında teknik anlatımın daha açık olması gerektiğini belirtti. Daha büyük bir çalışma öncesinde küçük bir değerlendirme yapılmasını önerdi.""",
"""[[ORG|Apple]], [[PERSON|Ben Miller]]'ı [[LOC|Seattle]]'daki bir deneme etkinliğine davet etti. [[DATE|2025]] tarihli kurgusal çalışmada [[PRODUCT|iPad Air]] ele alındı. Ben, ürün hakkındaki olumlu yorumları ve kullanıcıların yanlış anlayabileceği noktaları ayrı ayrı kaydetti."""
]


# Annotate abbreviated second mentions; every named-person mention counts in the gold set.
SECOND_MENTIONS = {
 'standard': {'Sarah asked':'[[PERSON|Sarah]] asked', 'Daniel answered':'[[PERSON|Daniel]] answered',
 'Noah suggested':'[[PERSON|Noah]] suggested', 'Priya later':'[[PERSON|Priya]] later',
 'Lucas compared':'[[PERSON|Lucas]] compared', 'Olivia explained':'[[PERSON|Olivia]] explained',
 'Amina emphasized':'[[PERSON|Amina]] emphasized', 'Liam asked':'[[PERSON|Liam]] asked',
 'Chloe proposed':'[[PERSON|Chloe]] proposed'},
 'ambiguity': {'Nina later':'[[PERSON|Nina]] later', 'Leo reminded':'[[PERSON|Leo]] reminded',
 'Jordan volunteered':'[[PERSON|Jordan]] volunteered', 'Theo Lambert]] bought':'Theo Lambert]] bought'},
 'multi_entity': {'Owen asked':'[[PERSON|Owen]] asked', 'Milan asked':'[[PERSON|Milan]] asked',
 'Jack asked':'[[PERSON|Jack]] asked'},
 'extended': {'Ryan recorded':'[[PERSON|Ryan]] recorded', 'Lena later':'[[PERSON|Lena]] later',
 'Daniel was':'[[PERSON|Daniel]] was', 'Amira requested':'[[PERSON|Amira]] requested',
 'Zara asked':'[[PERSON|Zara]] asked', 'Caleb pointed':'[[PERSON|Caleb]] pointed',
 'Mina wanted':'[[PERSON|Mina]] wanted', 'Oliver requested':'[[PERSON|Oliver]] requested'},
 'challenging': {},
}
for group_name, mapping in SECOND_MENTIONS.items():
    for idx, paragraph in enumerate(GROUPS[group_name]):
        for old,new in mapping.items():
            if old in paragraph: paragraph=paragraph.replace(old,new)
        GROUPS[group_name][idx]=paragraph
TR_MENTIONS={'Sarah, toplantı':'[[PERSON|Sarah]], toplantı', 'Noah, yeni':'[[PERSON|Noah]], yeni',
 'Priya, etkinlik':'[[PERSON|Priya]], etkinlik', 'Lucas, kural':'[[PERSON|Lucas]], kural',
 'Olivia, yalnızca':'[[PERSON|Olivia]], yalnızca', 'Amina, eğitim':'[[PERSON|Amina]], eğitim',
 'Liam, ekipten':'[[PERSON|Liam]], ekipten', 'Chloe, temel':'[[PERSON|Chloe]], temel'}
for i, paragraph in enumerate(TR):
    for old,new in TR_MENTIONS.items():
        if old in paragraph: paragraph=paragraph.replace(old,new)
    TR[i]=paragraph
# Honorifics are outside the entity boundary.
GROUPS['challenging'][0] = GROUPS['challenging'][0].replace('[[PERSON|Dr. Alice Monroe]]', 'Dr. [[PERSON|Alice Monroe]]')


# One extra realistic sentence per test paragraph: longer context, no new named entities.
CONTEXT_SENTENCES = [
 'A second reviewer examined the written notes to make sure the final description reflected the original discussion rather than personal assumptions.',
 'Before publishing the summary, the team checked the source material and flagged statements that needed additional evidence or clarification.',
 'The participants also agreed that a clear evaluation would require both successful examples and a record of the mistakes.',
 'The original discussion was preserved in a shared document so that readers could trace each important detail to the source.',
 'The group planned a short follow-up session to compare methods using the same data and a consistent set of rules.',
 'After the meeting, the organizers drafted a neutral account that separated confirmed observations from uncertain interpretations.',
 'The final handout included enough background context to help someone who had not attended understand the situation correctly.',
 'Reviewers were encouraged to explain their decisions and to record ambiguous phrases before presenting the findings.',
 'The team kept the wording of the original material because seemingly small changes could affect the meaning of the record.',
 'At the end of the exercise, participants checked whether their conclusions relied on facts explicitly stated in the source.'
]
for group_name, marked_list in GROUPS.items():
    GROUPS[group_name]=[text+' '+CONTEXT_SENTENCES[i] for i,text in enumerate(marked_list)]
TR_CONTEXT = [
 'Son değerlendirmede ekip, hazırlanan özeti kaynak notlarla yeniden karşılaştırarak eksik ya da yanlış yorumlanan bilgileri işaretledi.',
 'Katılımcılar, rapora yalnızca kaynak metinde açıkça bulunan bilgilerin yazılması gerektiği konusunda fikir birliğine vardı.',
 'Hazırlanan kısa rapor, sonraki toplantıda aynı verilerle tutarlı bir karşılaştırma yapılabilmesi amacıyla saklandı.',
 'Ekip, küçük ifade farklılıklarının bilgi çıkarımı sonuçlarını değiştirebileceğini göz önünde bulundurarak metni olduğu gibi korudu.',
 'Toplantıdan sonra düzenlenen belgede, doğrulanmış gözlemlerle henüz açıklığa kavuşmamış yorumlar birbirinden ayrıldı.',
 'Çalışmanın sonunda herkes örnekleri tekrar inceleyerek kişi, kurum ve yer adlarının doğru işaretlendiğinden emin olmaya çalıştı.',
 'Bu yüzden sunumda yalnızca başarılı tahminler değil, hangi bilgilerin yanlış çıkarıldığı da ayrıca ele alınacaktı.',
 'Hazırlık ekibi, anlatımın açık olması için süreci adım adım belgeledi ve yanlış anlaşılabilecek ifadeleri yeniden kontrol etti.',
 'Sonuçları değerlendiren kişiler, aynı örneklerin farklı yaklaşımlarla karşılaştırılmasının daha anlamlı olacağına karar verdi.',
 'Deneyin güvenilirliği açısından başlangıçtaki metinler değiştirilmedi ve bütün çıkarımlar kaynakla tek tek karşılaştırıldı.'
]
TR=[text+' '+TR_CONTEXT[i] for i,text in enumerate(TR)]


def parse_marked(marked):
    parts=[]; entities=[]; cursor=0; out_len=0
    for m in PATTERN.finditer(marked):
        pre=marked[cursor:m.start()]
        parts.append(pre); out_len+=len(pre)
        label,val=m.group(1),m.group(2)
        start=out_len; parts.append(val); out_len+=len(val)
        entities.append({'text':val,'label':label,'start':start,'end':out_len})
        cursor=m.end()
    parts.append(marked[cursor:]); text=''.join(parts)
    assert '[[' not in text and ']]' not in text, text
    assert all(text[e['start']:e['end']]==e['text'] for e in entities)
    return text,entities

def record(marked, id, group, language, split, pair_id=None):
    text, all_ent=parse_marked(marked)
    core={'PERSON','ORG','LOC'}
    return {'id':id,'language':language,'split':split,'group':group,'pair_id':pair_id,
            'text':text,'entities':[e for e in all_ent if e['label'] in core],
            'extended_entities':[e for e in all_ent if e['label'] not in core]}

def main():
    en=[]
    for g, marked_list in GROUPS.items():
        for marked in marked_list:
            n=len(en)+1
            en.append(record(marked, f'EN{n:03d}', g, 'en', 'test', f'P{n:03d}' if n<=10 else None))
    tr=[record(m, f'TR{i:03d}', 'turkish_morphology', 'tr', 'supplementary', f'P{i:03d}') for i,m in enumerate(TR,1)]
    dev=[record(m,f'DEV_EN{i:03d}','pilot','en','dev') for i,m in enumerate(DEV_EN,1)] + [record(m,f'DEV_TR{i:03d}','pilot','tr','dev') for i,m in enumerate(DEV_TR,1)]
    assert len(en)==50 and len(tr)==10 and len(dev)==7
    for data, name in [(en,'english_test.jsonl'),(tr,'turkish_supplementary.jsonl'),(dev,'development.jsonl')]:
        with (ROOT/'data'/name).open('w',encoding='utf-8') as f:
            for r in data: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('Created:',len(en),'EN test /',len(tr),'TR supplementary /',len(dev),'dev')
    print('EN words min/mean/max',min(len(x['text'].split()) for x in en),round(sum(len(x['text'].split()) for x in en)/50,1),max(len(x['text'].split()) for x in en))
    print('Group counts:',dict(Counter(x['group'] for x in en)))

if __name__=='__main__':main()
