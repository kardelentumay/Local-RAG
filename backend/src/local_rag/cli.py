from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .foundry import FoundryChat, FoundryEmbeddings, FoundryRuntime
from .huggingface import DEFAULT_DATASET, download_articles
from .service import RAGService
from .store import SQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-rag", description="Foundry Local tabanlı çevrimdışı belge asistanı")
    parser.add_argument("--db", type=Path, default=Path("data/knowledge.db"))
    parser.add_argument("--embedding-model", default="qwen3-embedding-0.6b")
    parser.add_argument("--chat-model", default="qwen2.5-0.5b")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Belgeleri veritabanına ekle")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--chunk-size", type=int, default=700)
    ingest.add_argument("--overlap", type=int, default=100)
    ingest.add_argument("--batch-size", type=int, default=16)
    search = commands.add_parser("search", help="İlgili belge parçalarını getir")
    search.add_argument("question")
    search.add_argument("--top-k", type=int, default=3)
    ask = commands.add_parser("ask", help="Tek bir soruyu cevapla")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=2)
    chat = commands.add_parser("chat", help="Etkileşimli sohbet başlat")
    chat.add_argument("--top-k", type=int, default=2)
    download = commands.add_parser("download-hf", help="Hugging Face'ten makale indir")
    download.add_argument("--topic", choices=("rag", "llm", "agent"), default="rag")
    download.add_argument("--limit", type=int, default=20)
    download.add_argument("--dataset", default=DEFAULT_DATASET)
    download.add_argument("--output", type=Path)
    commands.add_parser("status", help="Veritabanı istatistiklerini göster")
    return parser


def print_results(results) -> None:
    if not results:
        print("Sonuç bulunamadı.")
        return
    for index, item in enumerate(results, start=1):
        preview = item.content.replace("\n", " ")[:220]
        print(f"[{index}] {item.source} / parça {item.position + 1} / skor {item.score:.3f}\n    {preview}")


def print_sources(results) -> None:
    if not results:
        return
    for index, item in enumerate(results, start=1):
        print(f"[{index}] {item.source} / parça {item.position + 1} / skor {item.score:.3f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = SQLiteStore(args.db)
    if args.command == "status":
        documents, chunks = store.stats()
        print(f"Veritabanı: {args.db}\nBelgeler: {documents}\nParçalar: {chunks}")
        return 0
    if args.command == "download-hf":
        output = args.output or Path("documents") / "huggingface" / args.topic
        try:
            report = download_articles(args.topic, args.limit, output, args.dataset)
            print(
                f"Veri seti: {report.dataset}\nKonu: {report.topic}\n"
                f"İndirilen makale: {report.downloaded}\nKlasör: {report.output_dir}"
            )
            return 0
        except (RuntimeError, ValueError) as exc:
            print(f"Hata: {exc}", file=sys.stderr)
            return 1

    runtime = None
    try:
        runtime = FoundryRuntime()
        embeddings = FoundryEmbeddings(runtime, args.embedding_model)
        chat = FoundryChat(runtime, args.chat_model) if args.command in {"ask", "chat"} else None
        service = RAGService(store, embeddings, chat)
        if args.command == "ingest":
            report = service.ingest(args.path, args.chunk_size, args.overlap, args.batch_size)
            print(f"İşlenen belge: {report.processed}\nAtlanan belge: {report.skipped}\nEklenen parça: {report.chunks}")
        elif args.command == "search":
            print_results(service.search(args.question, args.top_k))
        elif args.command == "ask":
            answer = service.answer(args.question, args.top_k)
            print(answer.text)
            if answer.sources:
                print("\nKaynaklar:")
                print_sources(answer.sources)
        else:
            print("Çıkmak için /quit yazın.")
            while True:
                question = input("\nSoru> ").strip()
                if question.lower() in {"/quit", "/exit", "çık", "cik"}:
                    break
                if not question:
                    continue
                answer = service.answer(question, args.top_k)
                print(f"\n{answer.text}")
                if answer.sources:
                    print("\nKaynaklar:")
                    print_sources(answer.sources)
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
