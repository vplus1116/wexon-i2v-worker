#!/usr/bin/env python3
"""WEXON T2 image-to-video GPU worker (RunPod Serverless, LTX-Video).

Тяжёлые импорты (torch/diffusers) — ЛЕНИВО внутри обработчика (try/except),
чтобы ошибки возвращались в ответе задачи, а не роняли воркер немым крашем.

input:
{
  "image_url": "https://...png",       # обязательный
  "prompt": "...", "negative_prompt": "...",
  "width": 480, "height": 832,         # кратно 32
  "num_frames": 97, "steps": 40, "fps": 24, "seed": 0
}
output: {"video_url": "<cloudinary>", "frames": n, "size":[w,h], "took": sec}
"""
import os, re, tempfile, uuid, time, traceback

# HF-кэш: network volume если есть, иначе локально
if os.path.isdir("/runpod-volume"):
    os.environ.setdefault("HF_HOME", "/runpod-volume/hf")
else:
    os.environ["HF_HOME"] = "/app/hf"
try:
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
except Exception:
    pass

import runpod

MODEL = os.environ.get("LTX_MODEL", "Lightricks/LTX-Video")

# --- Cloudinary (лёгкий импорт) ---
CLOUD_OK = False
try:
    import cloudinary, cloudinary.uploader

    def _init_cloudinary():
        cu = (os.environ.get("CLOUDINARY_URL") or "").strip()
        if cu:
            m = re.match(r"cloudinary://([^:]+):([^@]+)@(.+)", cu)
            if m:
                cloudinary.config(api_key=m.group(1), api_secret=m.group(2),
                                  cloud_name=m.group(3), secure=True)
                return True
        cn = os.environ.get("CLOUDINARY_CLOUD_NAME")
        ak = os.environ.get("CLOUDINARY_API_KEY")
        sec = os.environ.get("CLOUDINARY_API_SECRET")
        if cn and ak and sec:
            cloudinary.config(cloud_name=cn, api_key=ak, api_secret=sec, secure=True)
            return True
        return False

    CLOUD_OK = _init_cloudinary()
except Exception:
    CLOUD_OK = False

_pipe = None
_diff = {}


def _load_deps():
    """Ленивая загрузка тяжёлых модулей. Бросает с понятным сообщением."""
    if "torch" not in _diff:
        import torch
        from diffusers import LTXImageToVideoPipeline
        from diffusers.utils import export_to_video, load_image
        _diff["torch"] = torch
        _diff["LTXImageToVideoPipeline"] = LTXImageToVideoPipeline
        _diff["export_to_video"] = export_to_video
        _diff["load_image"] = load_image
    return _diff


def get_pipe():
    global _pipe
    if _pipe is None:
        d = _load_deps()
        p = d["LTXImageToVideoPipeline"].from_pretrained(MODEL, torch_dtype=d["torch"].bfloat16)
        p.to("cuda")
        try:
            p.vae.enable_tiling()
        except Exception:
            pass
        _pipe = p
    return _pipe


def _round32(v):
    v = int(v)
    return max(32, (v // 32) * 32)


def _round_frames(n):
    n = int(n)
    return ((n - 1) // 8) * 8 + 1 if n > 1 else 9


def handler(job):
    inp = job.get("input", {}) or {}
    url = inp.get("image_url")
    if not url:
        return {"error": "no image_url"}
    if not CLOUD_OK:
        return {"error": "cloudinary not configured: задай CLOUDINARY_URL в env эндпоинта"}

    prompt = inp.get("prompt") or (
        "subtle cinematic motion, gentle slow camera push-in, soft parallax, "
        "natural movement, neon light shimmer, photorealistic, high detail")
    neg = inp.get("negative_prompt") or (
        "worst quality, inconsistent motion, jittery, flicker, distorted, warped, blurry, deformed")
    width = _round32(inp.get("width", 480))
    height = _round32(inp.get("height", 832))
    num_frames = _round_frames(inp.get("num_frames", 97))
    steps = int(inp.get("steps", 40))
    fps = int(inp.get("fps", 24))
    seed = int(inp.get("seed", 0))

    t0 = time.time()
    try:
        d = _load_deps()
        image = d["load_image"](url)
        pipe = get_pipe()
        gen = d["torch"].Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=image, prompt=prompt, negative_prompt=neg,
            width=width, height=height, num_frames=num_frames,
            num_inference_steps=steps, generator=gen,
        ).frames[0]
        work = tempfile.mkdtemp(prefix="t2_")
        out = os.path.join(work, "clip.mp4")
        d["export_to_video"](frames, out, fps=fps)
    except Exception as e:
        return {"error": "i2v failed", "detail": str(e)[-700:],
                "trace": traceback.format_exc()[-1500:]}

    try:
        up = cloudinary.uploader.upload_large(
            out, resource_type="video", folder="wexon_t2",
            public_id="t2clip_%s" % uuid.uuid4().hex[:10])
    except Exception as e:
        return {"error": "upload failed", "detail": str(e)[-400:]}

    return {"video_url": up.get("secure_url"), "frames": num_frames,
            "size": [width, height], "took": round(time.time() - t0, 1)}


runpod.serverless.start({"handler": handler})
