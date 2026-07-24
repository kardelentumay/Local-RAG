# Local RAG Assistant - Phase 2

Belge koleksiyonundan kaynak göstererek cevap üreten, tamamen yerel çalışan bir RAG uygulaması. Belgeler parçalara ayrılır, Microsoft Foundry Local ile vektörleştirilir, SQLite'ta saklanır ve en ilgili parçalar yerel sohbet modeline bağlam olarak verilir.

## Gereksinimler

- Python 3.11+
- Windows 10/11 (WinML paketi önerilir), macOS veya Linux
- İlk model indirmesi için internet; sonraki çalıştırmalar çevrimdışı olabilir

## Kurulum

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Windows dışındaki sistemlerde etkinleştirme komutu `source .venv/bin/activate` biçimindedir.

## Kullanım

Belgeleri `documents/` klasörüne koyun. Desteklenen türler: `.txt`, `.md`, `.pdf`.

```powershell
local-rag download-hf --topic rag --limit 20
local-rag ingest documents
local-rag status
local-rag search "Kayıt şartları nelerdir?"
local-rag ask "Kayıt şartları nelerdir?"
local-rag chat
```

`download-hf`, varsayılan olarak `GXMZU/llm-rag-agent-papers` veri setinden seçilen bölümdeki makaleleri `documents/huggingface/<konu>/` içine ayrı Markdown dosyaları olarak indirir. Konular `rag`, `llm` ve `agent`; limit aralığı 1–100'dür. İndirme sırasında internet gerekir, sonrasında belgeler yerel kalır.

İlk `ingest` çağrısı embedding modelini, ilk `ask` çağrısı sohbet modelini indirip yükleyebilir. Varsayılanlar:

- Embedding: `qwen3-embedding-0.6b`
- Chat: `qwen2.5-0.5b` (hız odaklı varsayılan; daha yüksek kalite için `qwen2.5-1.5b` seçilebilir)
- Veritabanı: `data/knowledge.db`

Farklı ayarlar komut seçenekleriyle verilebilir:

```powershell
local-rag --db data/notes.db --embedding-model qwen3-embedding-0.6b ingest documents --chunk-size 700 --overlap 100
local-rag --chat-model qwen2.5-1.5b ask "Soru" --top-k 2
```

`ingest`, aynı dosya değişmemişse onu yeniden işlemez; değişmiş dosyanın eski parçalarını atomik olarak yeniler. Cevap bağlamda yoksa sistem tahmin yürütmemesi için yönlendirilir. Çıktıda kullanılan kaynak ve benzerlik puanları ayrıca gösterilir.

Uzun makale koleksiyonlarını daha hızlı indekslemek için batch boyutu artırılabilir:

```powershell
local-rag ingest documents\huggingface\rag --batch-size 64
```

## Mimari

1. `documents.py`: TXT/Markdown/PDF okuma ve örtüşmeli parçalara ayırma.
2. `foundry.py`: Foundry Local model yaşam döngüsü, embedding ve chat çağrıları.
3. `store.py`: SQLite şeması, FTS5/BM25 anahtar kelime araması ve cosine vector retrieval.
4. `service.py`: Reciprocal Rank Fusion tabanlı hybrid retrieval ve RAG orkestrasyonu.
5. `cli.py`: `ingest`, `search`, `ask`, `chat`, `status` arayüzü.

Koleksiyon küçük tutulduğu için vektörler SQLite'tan belleğe alınıp cosine similarity Python'da hesaplanır. Semantic ve BM25 sıralamaları Reciprocal Rank Fusion ile birleştirilir; aynı makaleden en fazla iki parça seçilir. Varsayılan chunk ayarı 700 karakter ve 100 karakter örtüşmedir. Büyük koleksiyonlarda özel bir vektör indeksi gerekir.

## Test

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

Testler Foundry modeli indirmez; deterministik sahte embedding/chat adaptörleri kullanır.
