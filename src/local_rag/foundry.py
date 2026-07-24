from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChatProvider(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class FoundryRuntime:
    def __init__(self, app_name: str = "local-rag-assistant"):
        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager
        except ImportError as exc:
            raise RuntimeError(
                "Foundry Local SDK bulunamadı. Windows'ta 'pip install foundry-local-sdk-winml' komutunu çalıştırın."
            ) from exc
        try:
            FoundryLocalManager.initialize(Configuration(app_name=app_name))
        except Exception as exc:
            if "already" not in str(exc).lower() and "initializ" not in str(exc).lower():
                raise
        self.manager = FoundryLocalManager.instance
        self._models: dict[str, object] = {}

    def load_model(self, alias: str):
        if alias not in self._models:
            model = self.manager.catalog.get_model(alias)
            if model is None:
                available = ", ".join(sorted(item.alias for item in self.manager.catalog.list_models()))
                raise RuntimeError(
                    f"Foundry Local model kataloğunda '{alias}' bulunamadı. "
                    f"Kullanılabilir modeller: {available}"
                )
            model.download()
            model.load()
            self._models[alias] = model
        return self._models[alias]

    def close(self) -> None:
        for model in reversed(list(self._models.values())):
            model.unload()
        self._models.clear()


class FoundryEmbeddings:
    def __init__(self, runtime: FoundryRuntime, model_alias: str):
        self.client = runtime.load_model(model_alias).get_embedding_client()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        response = self.client.generate_embeddings(values)
        return [list(item.embedding) for item in response.data]


class FoundryChat:
    def __init__(self, runtime: FoundryRuntime, model_alias: str, max_tokens: int = 160):
        self.client = runtime.load_model(model_alias).get_chat_client()
        self.client.settings.max_tokens = max_tokens
        self.client.settings.temperature = 0.2
        self.client.settings.top_p = 0.85
        self.client.settings.frequency_penalty = 0.8
        self.client.settings.presence_penalty = 1.2
        self.client.settings.random_seed = 42

    def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self.client.complete_chat(messages)
            return response.choices[0].message.content or ""
        except Exception as exc:
            detail = str(exc)
            if "cancel" in detail.lower():
                raise RuntimeError(
                    "Foundry Local cevap üretimini iptal etti. 'local-rag chat --top-k 2' ile tekrar deneyin "
                    "ve işlem sürerken terminali kapatmayın veya Ctrl+C kullanmayın."
                ) from exc
            raise RuntimeError(f"Foundry Local cevap üretemedi: {detail}") from exc
