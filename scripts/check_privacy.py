#!/usr/bin/env python3
"""Falla si un fichero versionado parece contener datos privados del autor."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PATTERNS = {
    "correo electrónico": re.compile(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    ),
    "ruta de usuario de macOS": re.compile(r"/Users/[^/\s`\"']+"),
    "ruta de usuario de Linux": re.compile(r"/home/[^/\s`\"']+"),
    "ruta de usuario de Windows": re.compile(
        r"(?i)[A-Z]:\\Users\\[^\\\s`\"']+"
    ),
    "ruta privada del sistema": re.compile(
        r"/(?:opt/data|Volumes|private/var)/[^\s`\"']*"
    ),
    "dirección IPv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "dirección MAC": re.compile(r"(?i)\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b"),
    "cabecera de clave privada": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
}

ALLOWED_VALUES = {
    "127.0.0.1",
    "0.0.0.0",
}

ALLOWED_COMMIT_EMAIL = re.compile(
    r"(?i)^(?:\d+\+)?[A-Z0-9-]+@users\.noreply\.github\.com$"
)
GITHUB_GENERATED_COMMITTER_EMAIL = "noreply@github.com"


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(item.decode()) for item in raw.split(b"\0") if item]


def commit_metadata_findings() -> list[tuple[str, str]]:
    raw = subprocess.check_output(
        ["git", "log", "--all", "--format=%H%x00%ae%x00%ce"], text=True
    )
    findings: list[tuple[str, str]] = []
    for record in raw.splitlines():
        commit, author_email, committer_email = record.split("\0")
        for role, email in (
            ("autor", author_email),
            ("committer", committer_email),
        ):
            allowed = ALLOWED_COMMIT_EMAIL.fullmatch(email) or (
                role == "committer" and email == GITHUB_GENERATED_COMMITTER_EMAIL
            )
            if not allowed:
                findings.append(
                    (f"correo personal del {role}", f"commit {commit[:12]}")
                )
    return findings


def main() -> int:
    findings: list[tuple[str, str]] = commit_metadata_findings()
    this_script = Path(__file__).resolve()

    for path in tracked_files():
        if not path.is_file():
            continue
        # Las expresiones de detección aparecen literalmente en este fichero.
        if path.resolve() == this_script:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    if match.group(0) in ALLOWED_VALUES:
                        continue
                    findings.append((label, f"{path}:{line_number}"))

    if not findings:
        print("Privacy check: OK")
        return 0

    print("Privacy check: se han encontrado posibles datos privados:")
    for label, location in findings:
        # No mostramos el valor para evitar copiarlo a los logs de CI.
        print(f"- {label}: {location}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
