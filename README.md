# Nivora Local RAG

Nivora, PDF, Markdown, TXT ve Excel belgelerini cihaz üzerinde indeksleyen, yerel dil modeliyle kaynaklı cevaplar üreten offline-first bir RAG asistanıdır. Proje Microsoft Summer School kapsamında geliştirilmiştir.

## Öne çıkan özellikler

- PDF, Markdown, TXT ve XLSX belge yükleme
- Belge kategorileri, belge silme ve sohbet geçmişi
- Semantic/vector search + SQLite FTS5/BM25 hibrit retrieval
- Belge kapsamlı ve bölüm farkındalıklı arama
- Kaynak belgesi, chunk numarası ve citation gösterimi
- Alakasız sorular için güvenli fallback
- Excel toplamları ve filtreli metrikler için deterministic analiz yolları
- Yerel Qwen modelleri ve Microsoft Foundry Local
- React arayüzü + FastAPI backend
- Electron tabanlı Windows masaüstü uygulaması

## Mimari

```text
React / Electron UI
        |
        v
FastAPI local API (127.0.0.1:8000)
        |
        +--> SQLite + FTS5/BM25 keyword search
        +--> embedding cosine similarity search
        +--> hybrid rank fusion + relevance validation
        +--> Foundry Local chat model
```

Ana backend modülleri:

- `backend/src/local_rag/documents.py`: PDF, TXT, Markdown ve Excel okuma; chunk üretimi
- `backend/src/local_rag/store.py`: SQLite, FTS5/BM25 ve embedding saklama
- `backend/src/local_rag/service.py`: retrieval, belge kapsamı, section extraction, validation ve cevap orkestrasyonu
- `backend/src/local_rag/api.py`: FastAPI endpoint’leri
- `backend/src/local_rag/foundry.py`: embedding/chat model yaşam döngüsü
- `frontend/app/page.tsx`: Nivora kullanıcı arayüzü
- `frontend/electron/main.cjs`: Electron ana süreç ve backend sidecar yönetimi

## Gereksinimler

- Windows 10/11 önerilir
- Python 3.11+
- Node.js 22.13+
- İlk model ve veri indirmesi için internet
- Model ve belgeler indirildikten sonra çalışma internet olmadan sürdürülebilir

## Backend kurulumu

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

API’yi başlatmak için:

```powershell
cd backend
.venv\Scripts\local-rag-api.exe
```

API adresi: `http://127.0.0.1:8000`

Yararlı endpoint’ler:

- `GET /api/health`
- `GET /api/status`
- `GET /api/documents`
- `POST /api/documents`
- `DELETE /api/documents/{source}`
- `POST /api/ask`
- `GET /api/docs`

## Frontend geliştirme

```powershell
cd frontend
npm install
npm run dev
```

Geliştirme arayüzü varsayılan olarak `http://localhost:3000` adresinde açılır. Frontend, yerel FastAPI backend’e bağlanır.

## CLI kullanımı

```powershell
cd backend
.venv\Scripts\Activate.ps1

local-rag download-hf --topic rag --limit 20
local-rag ingest documents
local-rag status
local-rag search "What is corrective RAG?"
local-rag ask "What is corrective RAG?" --top-k 2
local-rag --chat-model qwen2.5-1.5b chat --top-k 2
```

Hugging Face indirme işlemi internet gerektirir. İndirilen Markdown belgeleri yerel `documents/huggingface/` altında tutulur.

## Model seçenekleri

- `qwen2.5-1.5b`: Daha iyi bağlam ve cevap kalitesi; daha yüksek RAM ve gecikme
- `qwen2.5-0.5b`: Daha düşük bellek kullanımı ve daha hızlı cevap; daha sınırlı reasoning
- Embedding modeli: `qwen3-embedding-0.6b`

API ve Nivora arayüzündeki varsayılan chat modeli `qwen2.5-1.5b` olarak ayarlanmıştır. CLI’de hız için `qwen2.5-0.5b` seçilebilir.

## Retrieval yaklaşımı

Nivora tek bir arama yöntemine dayanmaz:

1. Soru embedding’e çevrilir ve cosine similarity ile semantic sonuçlar bulunur.
2. Aynı soru SQLite FTS5/BM25 ile keyword olarak aranır.
3. İki sonuç listesi rank tabanlı hibrit füzyonla birleştirilir.
4. Özel ad içeren sorularda belge kapsamı ve proje/section başlıkları dikkate alınır.
5. İlgisiz evidence varsa model çağrılmadan güvenli fallback döndürülür.
6. Cevap citation’ları kaynak sayısına göre doğrulanır.

Deterministic işlemler, dil modelinin tahminine bırakılmaz. Excel toplamları, filtreli metrikler, glossary tanımları ve bazı profil bölümleri doğrudan kod tabanlı analizle ele alınır.

## Masaüstü uygulaması

Electron uygulaması backend’i yerel bir sidecar process olarak başlatır. Paketleme için:

```powershell
cd frontend
npm run desktop:backend
npm run desktop:build
npm run desktop:package
```

Installer çıktısı `frontend/release/` altında oluşturulur. Uygulama, backend’i `127.0.0.1` üzerinde çalıştırır; belgeler ve modeller cihazda kalır.

## Testler

Backend testleri:

```powershell
cd backend
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

Frontend testleri:

```powershell
cd frontend
npm test
npm run lint
```

Test kapsamı; ingest idempotency, PDF/Excel okuma, FTS araması, hibrit fusion, relevance validation, citation kontrolü, belge silme ve API davranışlarını içerir.

## Veri ve gizlilik

- Belgeler varsayılan olarak yerel `backend/documents/` altında tutulur.
- Bilgi tabanı `backend/data/knowledge.db` dosyasındadır.
- API yalnızca localhost üzerinde çalışır.
- İlk model indirmesinden sonra cevap üretimi için internet gerekmez.
- Hassas belgeleri Git’e eklemeyin; `documents/uploads/` ve yerel veritabanı dosyaları proje kaynak kodundan ayrı tutulmalıdır.

## Bilinen sınırlamalar

- 0.5B model karmaşık karşılaştırma ve uzun liste sorularında 1.5B modele göre daha fazla hata yapabilir.
- İlk model çağrısı lazy loading nedeniyle sonraki çağrılardan daha yavaştır.
- Çok büyük koleksiyonlarda SQLite üzerinde Python cosine search yerine özel bir vector index gerekebilir.
- PDF metin düzeni bozuksa section sınırları ve tablo yapısı etkilenebilir.

## Proje yapısı

```text
Local RAG/
├── backend/
│   ├── src/local_rag/
│   ├── tests/
│   ├── documents/
│   ├── data/
│   └── pyproject.toml
├── frontend/
│   ├── app/
│   ├── electron/
│   ├── public/
│   └── package.json
└── README.md
```
