from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_DATASET = "GXMZU/llm-rag-agent-papers"
DATASET_API = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class DownloadReport:
    dataset: str
    topic: str
    downloaded: int
    output_dir: Path


def download_articles(
    topic: str,
    limit: int,
    output_dir: Path,
    dataset: str = DEFAULT_DATASET,
    fetch_json: Callable[[str], dict] | None = None,
) -> DownloadReport:
    normalized_topic = topic.strip().lower()
    if normalized_topic not in {"rag", "llm", "agent"}:
        raise ValueError("topic yalnızca rag, llm veya agent olabilir")
    if limit < 1 or limit > 100:
        raise ValueError("limit 1 ile 100 arasında olmalıdır")

    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": normalized_topic,
            "offset": 0,
            "length": limit,
        }
    )
    payload = (fetch_json or _fetch_json)(f"{DATASET_API}?{query}")
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError("Hugging Face veri setinden kayıt alınamadı")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, wrapper in enumerate(rows, start=1):
        row = wrapper.get("row", wrapper)
        title = _clean_value(row.get("title")) or f"{normalized_topic.upper()} makalesi {index}"
        content = _clean_value(row.get("content")) or _clean_value(row.get("abstract"))
        if not content:
            continue
        paper_id = _clean_value(row.get("paper_id"))
        link = _clean_value(row.get("link"))
        year = _clean_value(row.get("year"))
        filename = f"{index:03d}-{_slugify(title)}.md"
        markdown = _to_markdown(title, year, paper_id, link, dataset, normalized_topic, content)
        (output_dir / filename).write_text(markdown, encoding="utf-8")
        written += 1

    manifest = {
        "dataset": dataset,
        "topic": normalized_topic,
        "requested": limit,
        "downloaded": written,
        "source": f"https://huggingface.co/datasets/{dataset}",
    }
    (output_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return DownloadReport(dataset, normalized_topic, written, output_dir)


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "local-rag-assistant/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except Exception as exc:
        raise RuntimeError(f"Hugging Face bağlantısı başarısız: {exc}") from exc


def _clean_value(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _slugify(value: str) -> str:
    ascii_safe = value.lower().encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_safe).strip("-")
    return (slug or "article")[:80]


def _to_markdown(
    title: str,
    year: str,
    paper_id: str,
    link: str,
    dataset: str,
    topic: str,
    content: str,
) -> str:
    metadata = [
        f"# {title}",
        "",
        f"- Yıl: {year or 'Belirtilmemiş'}",
        f"- Makale kimliği: {paper_id or 'Belirtilmemiş'}",
        f"- Makale bağlantısı: {link or 'Belirtilmemiş'}",
        f"- Hugging Face veri seti: https://huggingface.co/datasets/{dataset}",
        f"- Konu: {topic}",
        "",
        "## İçerik",
        "",
        content,
        "",
    ]
    return "\n".join(metadata)
