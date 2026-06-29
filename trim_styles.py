#!/usr/bin/env python3
"""Trim server.py STYLES to the single default model (Realistic Vision V5.1)."""
import re, ast
f = "/home/sp9/stream-diffusion-local/StreamDiffusion/web/server.py"
s = open(f).read()
if s.count('"model":') == 1:
    print("already single-model"); raise SystemExit(0)
new = '''STYLES = {
    "photorealistic": {
        "label": "Photorealistic",
        "model": "SG161222/Realistic_Vision_V5.1_noVAE",
        "suffix": "RAW photo, photorealistic, 35mm photograph, natural skin texture, "
                  "soft natural lighting, highly detailed, sharp focus",
        "negative": "anime, illustration, cartoon, drawing, painting, cgi, 3d render, "
                    "low quality, blurry, deformed",
    },
}

'''
s2 = re.sub(r"STYLES = \{.*?\n\}\n\n", new, s, count=1, flags=re.S)
assert s2 != s, "STYLES block not matched"
assert s2.count('"model":') == 1, "expected exactly 1 model after trim"
ast.parse(s2)
open(f, "w").write(s2)
print("STYLES trimmed to 1 model (Realistic Vision); server.py AST OK")
