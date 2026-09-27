"""Turn the screenshots from ``capture_demo.py`` into the README media.

    python scripts/make_media.py --frames media/frames

Writes media/demo.mp4 (every step, captioned, ~4 s each; needs an ``ffmpeg`` executable, e.g. the one in
the ``imageio-ffmpeg`` wheel), media/demo.gif (the key steps, smaller) and media/flow.png (a grid).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 900
CAPTIONS = {
    "algebra_curriculum": "Algebra textbook PDF: topics from its bookmarks, prerequisites from order",
    "algebra_diagnostic_question": "Adaptive diagnostic: a difficulty staircase per topic",
    "algebra_placement": "Placement: mastered / not yet per topic, and where to start",
    "algebra_tutor_explain": "Explain mode: every statement cites the book's section and page",
    "algebra_tutor_socratic": "Socratic mode (topic 2.1): leading questions, then the full cited explanation",
    "algebra_practice_feedback": "Practice: generated questions, checked before use; instant feedback with the source",
    "algebra_practice_rubric": "Free-text answers graded on a rubric from the material, shown to the learner",
    "algebra_review": "SM-2 spaced repetition: questions come back when they are due",
    "algebra_progress": "Progress: mastery per topic, diagnostic result, upcoming reviews",
    "controls_curriculum": "Second subject, no code changes: control-systems notes (Markdown)",
    "controls_diagnostic_question": "The same diagnostic on a different subject",
    "controls_placement": "Placement; missing prerequisites shorten the test",
    "controls_tutor_explain": "Cited explanation from the control-systems notes",
    "controls_progress": "Progress on the second subject",
}
GIF_STEPS = ["algebra_curriculum", "algebra_placement", "algebra_tutor_explain", "algebra_tutor_socratic",
             "algebra_practice_rubric", "algebra_progress", "controls_curriculum", "controls_tutor_explain"]


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def framed(path: Path, caption: str, step: int, total: int) -> Image.Image:
    """Screenshot fitted into W x H under a caption bar."""
    im = Image.open(path).convert("RGB")
    bar = 64
    scale = min(W / im.width, (H - bar) / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    out = Image.new("RGB", (W, H), (246, 247, 250))
    out.paste(im, ((W - im.width) // 2, bar))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W, bar], fill=(40, 44, 90))
    d.text((24, 18), f"{step}/{total}  {caption}", font=font(24), fill=(255, 255, 255))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "media" / "frames"))
    ap.add_argument("--out", default=str(ROOT / "media"))
    args = ap.parse_args()
    frames = sorted(Path(args.frames).glob("*.png"))
    out = Path(args.out)
    key = lambda p: p.stem.split("_", 1)[1]                      # noqa: E731
    slides = [framed(p, CAPTIONS.get(key(p), key(p)), i, len(frames)) for i, p in enumerate(frames, 1)]

    # grid of the key screens
    grid_keys = ["algebra_curriculum", "algebra_placement", "algebra_tutor_explain", "algebra_tutor_socratic",
                 "algebra_practice_rubric", "algebra_progress", "controls_curriculum", "controls_tutor_explain",
                 "controls_progress"]
    tiles = [s.resize((W // 2, H // 2), Image.LANCZOS) for p, s in zip(frames, slides) if key(p) in grid_keys]
    cols = 3
    grid = Image.new("RGB", (cols * W // 2, -(-len(tiles) // cols) * H // 2), "white")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * W // 2, (i // cols) * H // 2))
    grid.save(out / "flow.png", optimize=True)

    # GIF: key steps, 3.5 s each, smaller
    gif = [s.resize((960, 675), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128)
           for p, s in zip(frames, slides) if key(p) in GIF_STEPS]
    gif[0].save(out / "demo.gif", save_all=True, append_images=gif[1:], duration=3500, loop=0, optimize=True)

    # MP4: every step, 4 s each
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("no ffmpeg found: skipping demo.mp4")
            return
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(slides):
            s.save(Path(tmp) / f"{i:03d}.png")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "1/4", "-i", str(Path(tmp) / "%03d.png"),
                        "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "23", "-movflags", "+faststart",
                        str(out / "demo.mp4")], check=True)
    print("wrote", out / "demo.mp4", out / "demo.gif", out / "flow.png")


if __name__ == "__main__":
    main()
