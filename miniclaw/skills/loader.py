"""SKILL.md 技能加载。

front matter 只需要 name/description 两个键，手写解析避免引入 YAML 依赖：

    ---
    name: summarizer
    description: Summarize the given text.
    ---
    指令正文……
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SkillFormatError(ValueError):
    pass


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    source: Path | None = None


def parse_skill_md(text: str, source: Path | None = None) -> Skill:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFormatError(f"{source}: missing front matter delimiter '---'")
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        raise SkillFormatError(f"{source}: unterminated front matter")

    meta: dict[str, str] = {}
    for line in lines[1:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillFormatError(f"{source}: invalid front matter line {stripped!r}")
        key, _, value = stripped.partition(":")
        meta[key.strip()] = value.strip()

    name = meta.get("name")
    description = meta.get("description")
    if not name:
        raise SkillFormatError(f"{source}: front matter requires 'name'")
    if not description:
        raise SkillFormatError(f"{source}: front matter requires 'description'")
    instructions = "\n".join(lines[end + 1 :]).strip()
    return Skill(name=name, description=description, instructions=instructions, source=source)


def load_skill(path: Path) -> Skill:
    return parse_skill_md(path.read_text(encoding="utf-8"), source=path)


def load_skills(directory: Path) -> list[Skill]:
    """加载目录及其一级子目录下的全部 SKILL.md；技能名重复视为配置错误。"""
    if not directory.is_dir():
        raise FileNotFoundError(f"skill directory not found: {directory}")
    paths = sorted(directory.glob("SKILL.md")) + sorted(directory.glob("*/SKILL.md"))
    skills = [load_skill(path) for path in paths]
    names = [skill.name for skill in skills]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"duplicate skill names: {sorted(duplicates)}")
    return sorted(skills, key=lambda s: s.name)
