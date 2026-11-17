"""Cliente de IA (OpenRouter, API compatível com OpenAI).

A chave NUNCA fica no código. Ordem de busca:
1. variável de ambiente OPENROUTER_API_KEY (scripts de linha de comando);
2. st.secrets["OPENROUTER_API_KEY"] (Streamlit local ou Streamlit Cloud).
Sem chave, o sistema cai no modo demonstração (heurística local).
"""

from __future__ import annotations

import json
import os
import re
import time

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"


def _from_streamlit_secrets(name: str) -> str | None:
    try:
        import streamlit as st

        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:  # sem secrets.toml, fora do Streamlit etc.
        return None


def get_setting(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _from_streamlit_secrets(name) or default


def get_api_key() -> str | None:
    key = get_setting("OPENROUTER_API_KEY")
    return key if key and key.startswith("sk-") else None


def get_model() -> str:
    return get_setting("DCL_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def is_available() -> bool:
    return get_api_key() is not None


class LLMError(RuntimeError):
    pass


def _client():
    from openai import OpenAI

    key = get_api_key()
    if not key:
        raise LLMError("OPENROUTER_API_KEY não configurada")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=60)


def _extract_json(text: str) -> dict:
    """Aceita JSON puro ou JSON cercado de ```json ... ``` / texto extra."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_json(system: str, user: str, validate, retries: int = 3) -> dict:
    """Chama o modelo pedindo JSON, valida e re-tenta devolvendo o erro ao modelo.

    `validate` recebe o dict e devolve a versão normalizada ou levanta ValueError.
    """
    client = _client()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=get_model(),
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            return validate(_extract_json(content))
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            # Feedback explícito: o modelo corrige o próprio erro na próxima tentativa.
            messages = messages[:2] + [
                {"role": "assistant", "content": content if "content" in locals() else ""},
                {"role": "user", "content": f"Sua resposta foi rejeitada: {exc}. Responda de novo, só com o JSON válido."},
            ]
        except Exception as exc:  # rede, limite de taxa, modelo indisponível
            last_error = exc
            time.sleep(2 ** attempt)
    raise LLMError(f"falha após {retries} tentativas: {last_error}")


def chat_text(system: str, user: str, max_tokens: int = 600) -> str:
    client = _client()
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
