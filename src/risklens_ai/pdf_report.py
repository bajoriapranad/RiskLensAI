"""Minimal PDF report writer using only the Python standard library."""

from __future__ import annotations

from pathlib import Path
import textwrap


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_text_pdf(text: str, output_path: str | Path) -> Path:
    """Write a simple multi-page PDF containing memo text."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=92) or [""])

    pages = [lines[i : i + 48] for i in range(0, len(lines), 48)] or [[]]
    objects: list[bytes] = []

    def add_object(payload: str | bytes) -> int:
        objects.append(payload.encode("latin-1") if isinstance(payload, str) else payload)
        return len(objects)

    font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []

    for page_lines in pages:
        commands = ["BT", "/F1 10 Tf", "50 760 Td", "14 TL"]
        for line in page_lines:
            commands.append(f"({_escape_pdf_text(line)}) Tj")
            commands.append("T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        content_id = add_object(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_ids.append(
            add_object(
                f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            )
        )

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    patched: list[bytes] = []
    for obj in objects:
        patched.append(obj.replace(b"/Parent 0 0 R", f"/Parent {pages_id} 0 R".encode("ascii")))

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(patched, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_start = len(output)
    output.extend(f"xref\n0 {len(patched) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(patched) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(output)
    return path

