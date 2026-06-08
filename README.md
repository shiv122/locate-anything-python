# LocateAnything Service

A small **FastAPI** service that loads NVIDIA's
[**LocateAnything-3B**](https://huggingface.co/nvidia/LocateAnything-3B)
visual-grounding model (via transformers, baked into the image). Clients send an
**image URL** — not base64 — and the service fetches the bytes, runs grounding,
and returns the raw model text plus **parsed bounding boxes / points**.

Built to pair with the detector backend the same way `glm-ocr-service` does:
detection uploads each frame to DigitalOcean Spaces and sends that public URL here.

> Unlike GLM-OCR (which is Ollama behind FastAPI), this model ships custom
> `trust_remote_code` code and a non-standard parallel-box `generate()`, so it
> runs **directly in transformers**, in-process. Inference is serialized behind a
> lock (one request on the GPU at a time).

## API

### `POST /locate`
Send a fully-formed `prompt`, **or** a `task` + `query` (the service fills the
template from the model card):

```json
{
  "image_url": "https://df-detection.blr1.digitaloceanspaces.com/frames/frame_x.jpg",
  "prompt": "Locate all instances matching: person, ball",
  "mode": "hybrid",        // optional: fast | slow | hybrid
  "max_tokens": 8192,       // optional
  "temperature": 0.0        // optional (0 => greedy)
}
```
or
```json
{ "image_url": "https://.../frame.jpg", "task": "detect", "query": "person, ball" }
```

Response:
```json
{
  "text": "person<box><120><80><240><360></box> ball<box><500><510><560><570></box>",
  "boxes": [
    { "label": "person", "box_norm": [120,80,240,360], "box": [153.6,57.6,307.2,259.2] },
    { "label": "ball",   "box_norm": [500,510,560,570], "box": [640.0,367.2,716.8,410.4] }
  ],
  "points": [],
  "image": { "width": 1280, "height": 720 },
  "model": "nvidia/LocateAnything-3B",
  "mode": "hybrid",
  "prompt": "Locate all instances matching: person, ball",
  "timing_ms": { "download_ms": 120, "inference_ms": 850 }
}
```
- `box_norm` / `point_norm` are the raw model coordinates, normalized to `[0,1000]`.
- `box` / `point` are absolute pixels (`norm / 1000 * image_dim`).
- `label` is best-effort (text emitted before each `<box>`); trust `text` if unsure.

Errors return `{ "error": "..." }` with a 4xx/5xx status.

### Tasks (`task` field)
| task | template |
|------|----------|
| `detect` (default) | `Locate all instances matching: {query}` |
| `ground` | `Locate a single instance: {query}` |
| `ground_multi` | `Locate all instances: {query}` |
| `text` | `Please locate the text: {query}` |
| `scene_text` | `Detect all text in box format` |
| `gui_box` | `Locate the region: {query}` |
| `gui_point` | `Point to: {query}` (returns points, not boxes) |

### `GET /health`
```json
{ "status": "ok", "model": "nvidia/LocateAnything-3B", "device": "cuda" }
```
Returns 503 with `{ "status": "loading" | "error", ... }` until weights load.

## Run

Requires a GPU host (`nvidia-container-toolkit`). The image bakes in the weights
(~6 GB), so the first build is slow but cold starts only pay model-load time.

```bash
# add --build-arg HF_TOKEN=hf_xxx if the repo turns out to be gated
docker build -t locate-anything-service .
docker run --gpus all -p 8080:8080 locate-anything-service

# smoke test (model takes ~1 min to load on first boot)
curl localhost:8080/health
curl -X POST localhost:8080/locate \
  -H 'content-type: application/json' \
  -d '{"image_url":"https://.../frame.jpg","task":"detect","query":"person, ball"}'
```

## Wiring into the detector (done after you host + test this)

Point the detector backend at this service (mirrors `GLM_OCR_HOST`):
```
LOCATE_HOST=http://<this-service-host>:8080
LOCATE_MODEL=nvidia/LocateAnything-3B
```
The detector will POST `{image_url, prompt | task+query}` to `/locate`.

## Config (env)

| Var | Default | Notes |
|-----|---------|-------|
| `PORT` | `8080` | FastAPI listen port |
| `LOCATE_MODEL` | `nvidia/LocateAnything-3B` | HF repo id or local path (baked in) |
| `LOCATE_DEVICE` | `cuda` | torch device |
| `GENERATION_MODE` | `hybrid` | `fast` \| `slow` \| `hybrid` |
| `MAX_NEW_TOKENS` | `8192` | generation cap |
| `ATTN_IMPLEMENTATION` | _(model default)_ | e.g. `sdpa`, `eager`, `flash_attention_2` |
| `DOWNLOAD_TIMEOUT_SECONDS` | `30` | image fetch timeout |
| `MAX_IMAGE_BYTES` | `26214400` | 25 MB fetch cap |
| `HF_TOKEN` | — | **build-time only**, for gated repos |

## Notes / things to confirm on first run

This model's inference path is custom (`trust_remote_code`), so a few details
are coded defensively and worth confirming once it's actually running on a GPU:

- **Chat template**: the card calls `processor.py_apply_chat_template(...)`; the
  code tries that, then falls back to `apply_chat_template`.
- **`generate()` return**: handled whether it returns a string, list of strings,
  or token ids (prompt prefix stripped).
- **`flash-attn` / MagiAttention**: optional. If the custom code requires a
  specific attention backend, set `ATTN_IMPLEMENTATION` (e.g. `sdpa`/`eager`).
# locate-anything-python
