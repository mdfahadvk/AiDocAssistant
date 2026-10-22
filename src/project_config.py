"""Project-to-document configuration used while indexing documents."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_PROJECT = "General"


def _mapping_path() -> Path:
    return Path(os.getenv("PROJECT_MAPPING_FILE", "project_mapping.json")).expanduser()


def load_project_mapping() -> dict[str, list[str]]:
    path = _mapping_path()
    if not path.is_file():
        return {}
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(project): [str(name) for name in names if str(name).strip()]
        for project, names in value.items()
        if str(project).strip() and isinstance(names, list)
    }


def project_for_document(source_label: str, mapping: dict[str, list[str]] | None = None) -> str:
    """Resolve a document basename to its configured project."""
    filename = Path(str(source_label).replace("\\", "/")).name.casefold()
    for project, documents in (mapping if mapping is not None else load_project_mapping()).items():
        if any(Path(str(document).replace("\\", "/")).name.casefold() == filename for document in documents):
            return project
    return DEFAULT_PROJECT


def canonical_project_name(project: str, mapping: dict[str, list[str]] | None = None) -> str:
    """Trim a project name and preserve an existing display-name casing."""
    project = str(project).strip()
    if not project:
        return ""
    known_mapping = mapping if mapping is not None else load_project_mapping()
    return next(
        (existing for existing in known_mapping if existing.casefold() == project.casefold()),
        project,
    )


def assign_document_to_project(source_label: str, project: str) -> dict[str, list[str]]:
    """Persist one filename's project assignment and return the updated mapping."""
    project = str(project).strip()
    if not project:
        raise ValueError("Project name cannot be empty.")
    filename = Path(str(source_label).replace("\\", "/")).name
    mapping = load_project_mapping()
    project = canonical_project_name(project, mapping)
    filename_key = filename.casefold()
    for existing_project, documents in mapping.items():
        mapping[existing_project] = [
            document
            for document in documents
            if Path(str(document).replace("\\", "/")).name.casefold() != filename_key
        ]
    mapping = {name: documents for name, documents in mapping.items() if documents}
    mapping.setdefault(project, []).append(filename)
    path = _mapping_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return mapping
