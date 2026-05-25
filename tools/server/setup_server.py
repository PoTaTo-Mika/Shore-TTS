from __future__ import annotations

import io, os, sys, tempfile
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["PROJECT_ROOT"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

import torch, torchaudio, uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from tools.inference import infer, load_model

# --------------- config ---------------

CHECKPOINT = os.environ.get("SHORE_CHECKPOINT", "checkpoints/pretrain-200M")
DEVICE = os.environ.get("SHORE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HOST = os.environ.get("SHORE_HOST", "0.0.0.0")
PORT = int(os.environ.get("SHORE_PORT", "1145"))

# --------------- lifespan ---------------

model, sample_rate = None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, sample_rate
    model, sample_rate = load_model(CHECKPOINT, torch.device(DEVICE))
    print(f"Model ready: sr={sample_rate} device={DEVICE}")
    yield


app = FastAPI(title="Shore-TTS", lifespan=lifespan)

# --------------- routes ---------------


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "sample_rate": sample_rate}


@app.post("/v1/tts")
async def tts(
    text: str = Form(),
    ref_audio: UploadFile = File(),
    ref_text: str = Form(),
    steps: int = Form(32),
    cfg_strength: float = Form(1.0),
    speed: float = Form(1.0),
    seed: int | None = Form(None),
    fix_duration: float | None = Form(None),
    max_text_length: int = Form(200),
    sway_sampling_coef: float | None = Form(None),
):
    import traceback, logging
    import torch as _torch

    suffix = Path(ref_audio.filename).suffix or ".wav"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(await ref_audio.read())
            tmp = f.name
        wav, sr = infer(
            model, text, ref_audio=tmp, ref_text=ref_text,
            steps=steps, cfg_strength=cfg_strength, speed=speed,
            seed=seed, fix_duration=fix_duration,
            max_text_length=max_text_length,
            sway_sampling_coef=sway_sampling_coef,
            device=_torch.device(DEVICE),
        )
    except Exception:
        logging.getLogger("shore_tts.server").error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    finally:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)

    buf = io.BytesIO()
    torchaudio.save(buf, wav, sr, format="wav")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Sample-Rate": str(sr), "X-Duration-Sec": f"{wav.shape[-1] / sr:.2f}"})

# --------------- main ---------------

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
