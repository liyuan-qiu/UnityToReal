from pathlib import Path
import re
from PIL import Image, ImageDraw

svg_path = Path(r"c:\3DReconstruction\samplePhoto\samplePhoto2\4x4_1000-0.svg")
out_path = svg_path.with_suffix(".jpg")
text = svg_path.read_text(encoding="utf-8")

m = re.search(r'viewBox="([^"]+)"', text)
vb = [float(x) for x in m.group(1).split()]
vw, vh = vb[2], vb[3]
size = 1000
scale_x = size / vw
scale_y = size / vh

img = Image.new("RGB", (size, size), (0, 0, 0))
draw = ImageDraw.Draw(img)

for m in re.finditer(r"<rect([^>]*)>", text):
    attrs = m.group(1)

    def get(name: str, default: str = "0") -> float:
        mm = re.search(rf'{name}="([^"]+)"', attrs)
        return float(mm.group(1)) if mm else float(default)

    x, y, w, h = get("x"), get("y"), get("width"), get("height")
    fill = re.search(r'fill="([^"]+)"', attrs)
    color = (255, 255, 255) if fill and fill.group(1) == "white" else (0, 0, 0)
    x0 = int(round(x * scale_x))
    y0 = int(round(y * scale_y))
    x1 = int(round((x + w) * scale_x))
    y1 = int(round((y + h) * scale_y))
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color)

img.save(out_path, "JPEG", quality=95)
print(f"Wrote {out_path} ({img.size[0]}x{img.size[1]})")
