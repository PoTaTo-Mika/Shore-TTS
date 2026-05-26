from __future__ import annotations

import base64, io, os, sys, tempfile
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["PROJECT_ROOT"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

import torch, torchaudio, uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from tools.inference import infer, load_model, parellel_infer

BATCH_MAX_SIZE = 8

class BatchItem(BaseModel):
    text: str
    ref_text: str
    ref_audio: str  # base64-encoded wav

class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=BATCH_MAX_SIZE)
    steps: int = 32
    cfg_strength: float = 1.0
    speed: float = 1.0
    seed: int | None = None
    fix_duration: float | None = None
    sway_sampling_coef: float | None = None

model, sample_rate = None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, sample_rate
    model, sample_rate = load_model(CHECKPOINT, torch.device(DEVICE))
    print(f"Model ready: sr={sample_rate} device={DEVICE}")
    yield

app = FastAPI(title="Shore-TTS", lifespan=lifespan)

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


@app.post("/v1/tts/batch")
async def tts_batch(req: BatchRequest):
    import traceback, logging
    import torch as _torch

    tmpfiles: list[Path] = []
    try:
        texts, ref_audios, ref_texts = [], [], []
        for item in req.items:
            raw = base64.b64decode(item.ref_audio)
            suffix = ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(raw)
                tmpfiles.append(Path(f.name))
                ref_audios.append(str(tmpfiles[-1]))
            texts.append(item.text)
            ref_texts.append(item.ref_text)

        wavs, sr = parellel_infer(
            model,
            batch_inputs=[texts, ref_audios, ref_texts],
            steps=req.steps,
            cfg_strength=req.cfg_strength,
            speed=req.speed,
            seed=req.seed,
            fix_duration=req.fix_duration,
            sway_sampling_coef=req.sway_sampling_coef,
            device=_torch.device(DEVICE),
        )
    except Exception:
        logging.getLogger("shore_tts.server").error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    finally:
        for fp in tmpfiles:
            fp.unlink(missing_ok=True)

    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, wav in enumerate(wavs):
            wav_buf = io.BytesIO()
            torchaudio.save(wav_buf, wav, sr, format="wav")
            zf.writestr(f"{i}.wav", wav_buf.getvalue())
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={
                        "X-Sample-Rate": str(sr),
                        "X-Batch-Count": str(len(wavs)),
                    })

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Shore-TTS server")
    p.add_argument("--checkpoint", default=os.environ.get("SHORE_CHECKPOINT", "checkpoints/pretrain-200M"))
    p.add_argument("--device", default=os.environ.get("SHORE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--host", default=os.environ.get("SHORE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("SHORE_PORT", "1145")))
    args = p.parse_args()

    CHECKPOINT = args.checkpoint
    DEVICE = args.device
    HOST = args.host
    PORT = args.port

    uvicorn.run(app, host=HOST, port=PORT)
