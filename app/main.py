"""
LocateAnything service.

A small FastAPI server that loads NVIDIA's LocateAnything-3B visual-grounding
model (via transformers, baked into the image) and exposes a URL-based API.
Clients send an IMAGE URL — not base64 — and this service fetches the bytes,
runs grounding, and returns the raw model text plus parsed bounding boxes and
points (in both normalized [0,1000] and absolute pixel coordinates).

    POST /locate  { image_url, prompt? | (task? + query?), mode?, max_tokens?, temperature? }
               -> { text, boxes, points, image, model, mode, prompt, timing_ms }
    GET  /health

Unlike glm-ocr-service (FastAPI in front of Ollama), this model ships custom
trust_remote_code code and a non-standard parallel-box generate(), so it runs
directly in transformers, in-process. Inference is serialized behind a lock.
"""

from __future__ import annotations

import io
import os
import re
import threading
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

MODEL_ID = os.getenv("LOCATE_MODEL", "nvidia/LocateAnything-3B")
DEVICE = os.getenv("LOCATE_DEVICE", "cuda")
DEFAULT_MODE = os.getenv("GENERATION_MODE", "hybrid")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "8192"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "30"))
ATTN_IMPL = os.getenv("ATTN_IMPLEMENTATION") or None

# Prompt templates straight from the model card. Pick one via `task` and fill it
# with `query`; or send a fully-formed `prompt` to bypass these entirely.
TEMPLATES = {
    "detect": "Locate all instances matching: {q}",
    "ground": "Locate a single instance: {q}",
    "ground_multi": "Locate all instances: {q}",
    "text": "Please locate the text: {q}",
    "scene_text": "Detect all text in box format",
    "gui_box": "Locate the region: {q}",
    "gui_point": "Point to: {q}",
}

# Loaded once on startup (lifespan); inference is serialized because a single
# in-process GPU model is not safe to call from multiple threads at once.
_state: dict = {"model": None, "tokenizer": None, "processor": None, "ready": False, "error": None}
_infer_lock = threading.Lock()


def _load_model() -> None:
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if ATTN_IMPL:
        kwargs["attn_implementation"] = ATTN_IMPL

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, **kwargs).to(DEVICE).eval()

    _state.update(model=model, tokenizer=tokenizer, processor=processor)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load synchronously during startup — uvicorn won't accept connections until
    # this returns, so by the time /health is reachable we're ready or errored.
    try:
        _load_model()
        _state["ready"] = True
        print(f"[locate] model ready: {MODEL_ID} on {DEVICE}")
    except Exception as e:  # noqa: BLE001 — surface via /health instead of crashing
        _state["error"] = str(e)
        print(f"[locate] model load FAILED: {e}")
    yield


app = FastAPI(title="LocateAnything Service", version="1.0.0", lifespan=lifespan)


class LocateRequest(BaseModel):
    image_url: str
    prompt: str | None = None       # full instruction; takes precedence over task/query
    task: str | None = None         # one of TEMPLATES (default: detect)
    query: str | None = None        # categories/phrase to fill the template
    mode: str | None = None         # fast | slow | hybrid
    max_tokens: int | None = None
    temperature: float = 0.0
    do_sample: bool | None = None    # default: sample only when temperature > 0


@app.get("/health")
def health():
    if _state["ready"]:
        return {"status": "ok", "model": MODEL_ID, "device": DEVICE}
    return JSONResponse(
        status_code=503,
        content={
            "status": "loading" if _state["error"] is None else "error",
            "model": MODEL_ID,
            "error": _state["error"],
        },
    )


def _build_prompt(req: LocateRequest) -> str | None:
    if req.prompt:
        return req.prompt
    task = (req.task or "detect").lower()
    if req.query is None and task != "scene_text":
        return None
    tmpl = TEMPLATES.get(task, TEMPLATES["detect"])
    return tmpl.format(q=(req.query or "").strip())


def _fetch_image(url: str) -> tuple[bytes | None, JSONResponse | None]:
    try:
        with httpx.stream("GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as resp:
            if resp.status_code != 200:
                return None, JSONResponse(
                    status_code=400,
                    content={"error": f"image fetch returned HTTP {resp.status_code}"},
                )
            data = bytearray()
            for chunk in resp.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_IMAGE_BYTES:
                    return None, JSONResponse(
                        status_code=413,
                        content={"error": f"image exceeds {MAX_IMAGE_BYTES} bytes"},
                    )
            return bytes(data), None
    except Exception as e:  # noqa: BLE001
        return None, JSONResponse(
            status_code=400, content={"error": f"failed to fetch image: {e}"}
        )


# <box> wraps either 4 ints (a box: x1,y1,x2,y2) or 2 ints (a point: x,y), each
# in its own <int> tag, normalized to [0, 1000]. See the model card.
_BOX_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL)
_INT_RE = re.compile(r"<(\d+)>")


def _scale(v: int, dim: int) -> float:
    return round(v / 1000 * dim, 1)


def _parse_output(text: str, w: int, h: int) -> tuple[list, list]:
    boxes, points, last = [], [], 0
    for m in _BOX_RE.finditer(text):
        nums = [int(n) for n in _INT_RE.findall(m.group(1))]
        # best-effort label: the trailing line of text the model emitted before this box
        pre = text[last:m.start()].strip()
        label = pre.splitlines()[-1].strip(" :,.;-\t") if pre else ""
        last = m.end()
        if len(nums) == 4:
            x1, y1, x2, y2 = nums
            boxes.append({
                "label": label or None,
                "box_norm": [x1, y1, x2, y2],
                "box": [_scale(x1, w), _scale(y1, h), _scale(x2, w), _scale(y2, h)],
            })
        elif len(nums) == 2:
            x, y = nums
            points.append({
                "label": label or None,
                "point_norm": [x, y],
                "point": [_scale(x, w), _scale(y, h)],
            })
    return boxes, points


def _apply_chat_template(processor, messages):
    fn = getattr(processor, "py_apply_chat_template", None) or processor.apply_chat_template
    return fn(messages, tokenize=False, add_generation_prompt=True)


def _process_vision_info(processor, messages):
    fn = getattr(processor, "process_vision_info", None)
    if fn is not None:
        return fn(messages)
    from qwen_vl_utils import process_vision_info  # fallback
    return process_vision_info(messages)


def _decode(response, tokenizer, input_ids) -> str:
    # The custom generate() may return a string, a list of strings, or token ids.
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, (list, tuple)) and response and isinstance(response[0], str):
        return response[0].strip()
    try:
        seq = response[0]
        in_len = input_ids.shape[1]
        if hasattr(seq, "shape") and seq.shape[0] >= in_len:  # full sequence -> drop the prompt
            seq = seq[in_len:]
        return tokenizer.decode(seq, skip_special_tokens=True).strip()
    except Exception:  # noqa: BLE001
        return str(response)


@app.post("/locate")
def locate(req: LocateRequest):
    import torch

    if not _state["ready"]:
        return JSONResponse(
            status_code=503, content={"error": "model not ready", "detail": _state["error"]}
        )
    if not req.image_url:
        return JSONResponse(status_code=422, content={"error": "image_url is required"})

    prompt = _build_prompt(req)
    if not prompt:
        return JSONResponse(
            status_code=422,
            content={"error": "provide `prompt`, or `query` (with optional `task`)"},
        )

    t0 = time.monotonic()
    image_bytes, err = _fetch_image(req.image_url)
    if err is not None:
        return err
    download_ms = int((time.monotonic() - t0) * 1000)

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": f"invalid image: {e}"})
    w, h = image.size

    model, tokenizer, processor = _state["model"], _state["tokenizer"], _state["processor"]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    mode = req.mode or DEFAULT_MODE
    max_new = req.max_tokens or DEFAULT_MAX_NEW_TOKENS
    do_sample = req.do_sample if req.do_sample is not None else (req.temperature or 0) > 0

    t1 = time.monotonic()
    try:
        with _infer_lock:
            text = _apply_chat_template(processor, messages)
            images, videos = _process_vision_info(processor, messages)
            inputs = processor(
                text=[text], images=images, videos=videos, return_tensors="pt"
            ).to(DEVICE)

            gen = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", inputs.get("image_grid_thw", None)),
                tokenizer=tokenizer,
                max_new_tokens=max_new,
                generation_mode=mode,
                do_sample=bool(do_sample),
            )
            if "pixel_values" in inputs:
                gen["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            if do_sample:
                gen["temperature"] = req.temperature or 0.7

            with torch.inference_mode():
                response = model.generate(**gen)
            answer = _decode(response, tokenizer, inputs["input_ids"])
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=502, content={"error": f"LocateAnything inference failed: {e}"}
        )
    inference_ms = int((time.monotonic() - t1) * 1000)

    boxes, points = _parse_output(answer, w, h)
    return {
        "text": answer,
        "boxes": boxes,
        "points": points,
        "image": {"width": w, "height": h},
        "model": MODEL_ID,
        "mode": mode,
        "prompt": prompt,
        "timing_ms": {"download_ms": download_ms, "inference_ms": inference_ms},
    }
