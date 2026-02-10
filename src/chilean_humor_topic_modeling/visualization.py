from __future__ import annotations

from pathlib import Path


def save_plotly_figure(
    figure,
    output_dir: Path,
    stem: str,
    width: int = 1400,
    height: int = 800,
) -> tuple[Path, Path]:
    """
    Save Plotly figure as both HTML (interactive) and PNG (GitHub-friendly).

    Returns:
        (html_path, png_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{stem}.html"
    png_path = output_dir / f"{stem}.png"

    figure.write_html(str(html_path), include_plotlyjs="cdn")
    figure.write_image(str(png_path), format="png", width=width, height=height, scale=2)
    return html_path, png_path
