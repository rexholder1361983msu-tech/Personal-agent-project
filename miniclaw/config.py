"""项目配置。环境变量前缀 MINICLAW_。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_steps: int = 8
    db_path: str = "miniclaw.db"
    default_tenant: str = "default"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AgentConfig":
        """从环境变量构造；传入 dict 便于测试。"""
        source = os.environ if env is None else env

        def get(key: str, default: str) -> str:
            return source.get(f"MINICLAW_{key}", default)

        return cls(
            api_base=get("API_BASE", "https://api.openai.com/v1"),
            api_key=get("API_KEY", ""),
            model=get("MODEL", "gpt-4o-mini"),
            temperature=float(get("TEMPERATURE", "0.0")),
            max_steps=int(get("MAX_STEPS", "8")),
            db_path=get("DB", "miniclaw.db"),
            default_tenant=get("DEFAULT_TENANT", "default"),
        )
