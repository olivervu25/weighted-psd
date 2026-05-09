"""
combine_plots.py
================
Tile PNG files into a single figure arranged in a grid.

Usage:
    python scripts/combine_plots.py img1.png img2.png ... --out out.png --ncols N
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def combine(input_paths, out_path, ncols, titles=None, title_size=42):
    images = [Image.open(p) for p in input_paths]

    # Normalise to a common height so columns line up
    max_h = max(im.height for im in images)
    resized = []
    for im in images:
        if im.height != max_h:
            scale = max_h / im.height
            im = im.resize((round(im.width * scale), max_h), Image.LANCZOS)
        resized.append(im)

    nrows = (len(resized) + ncols - 1) // ncols
    col_w = max(im.width for im in resized)

    title_band = int(title_size * 1.6) if titles else 0

    canvas_w = col_w * ncols
    canvas_h = (max_h + title_band) * nrows
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    draw = ImageDraw.Draw(canvas) if titles else None
    font = _load_font(title_size) if titles else None

    for idx, im in enumerate(resized):
        row, col = divmod(idx, ncols)
        x = col * col_w
        y = row * (max_h + title_band)

        if titles and idx < len(titles) and titles[idx]:
            text = titles[idx]
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x + (col_w - tw) // 2
            ty = y + (title_band - th) // 2
            draw.text((tx, ty), text, fill=(0, 0, 0, 255), font=font)

        canvas.paste(im, (x, y + title_band))

    canvas = canvas.convert("RGB")
    canvas.save(out_path)
    print(f"Saved → {out_path}  ({canvas_w}×{canvas_h}, {nrows}×{ncols} grid)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Input PNG files (in order)")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--ncols", type=int, default=2, help="Number of columns")
    p.add_argument("--titles", nargs="+", default=None,
                   help="Per-panel titles, in same order as inputs")
    p.add_argument("--title_size", type=int, default=42,
                   help="Title font size in px (default 42)")
    args = p.parse_args()

    missing = [f for f in args.inputs if not Path(f).exists()]
    if missing:
        for f in missing:
            print(f"ERROR: file not found: {f}")
        raise SystemExit(1)

    if args.titles and len(args.titles) != len(args.inputs):
        raise SystemExit(
            f"ERROR: got {len(args.titles)} titles for {len(args.inputs)} inputs"
        )

    combine(args.inputs, args.out, args.ncols,
            titles=args.titles, title_size=args.title_size)


if __name__ == "__main__":
    main()