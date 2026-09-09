from __future__ import annotations

import pytest


@pytest.fixture()
def db_path(tmp_path):
    """每个测试独立的 SQLite 文件路径。"""
    return tmp_path / "sessions.db"
