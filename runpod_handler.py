import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import runpod
import soundfile as sf
from huggingface_hub import snapshot_download
from loguru import logger
from pydantic import ValidationError
from pydub import AudioSegment

from fish_speech.utils.schema import ServeTTSRequest
from tools.server.model_manager import ModelManager

HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
DEFAULT_MODEL_ID = "fishaudio/s2-pro"
DEFAULT_DECODER_CONFIG = "modded_dac_vq"
MAX_INLINE_BASE64_CHARS = int(
    os.environ.get("RUNPOD_MAX_INLINE_BASE64_CHARS", "12000000")
)

MODEL_MANAGER = None
INITIALIZATION_ERROR = None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_cached_snapshot_path(model_id: str) -> Path | None:
    if "/" not in model_id:
        return None

    org, name = model_id.split("/", 1)
    model_root = Path(HF_CACHE_ROOT) / f"models--{org}--{name}"
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if refs_main.is_file():
        snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / snapshot_hash
        if candidate.is_dir():
            return candidate

    if snapshots_dir.is_dir():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return snapshots[0]

    return None


def resolve_model_dir() -> Path:
    override = os.environ.get("FISH_MODEL_DIR")
    if override:
        model_dir = Path(override).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"FISH_MODEL_DIR does not exist: {model_dir}")
        return model_dir

    model_id = os.environ.get("FISH_MODEL_ID", DEFAULT_MODEL_ID)
    cached_path = resolve_cached_snapshot_path(model_id)
    if cached_path is not None:
        logger.info(f"Using Runpod cached model at {cached_path}")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return cached_path

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    logger.info(
        f"Runpod model cache not found, downloading {model_id} from Hugging Face"
    )
    snapshot_path = snapshot_download(repo_id=model_id, token=hf_token)
    return Path(snapshot_path)


def get_decoder_checkpoint_path(model_dir: Path) -> Path:
    override = os.environ.get("DECODER_CHECKPOINT_PATH")
    if override:
        decoder_path = Path(override).expanduser().resolve()
    else:
        decoder_path = model_dir / "codec.pth"

    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_path}")

    return decoder_path


def get_llama_checkpoint_path(model_dir: Path) -> Path:
    override = os.environ.get("LLAMA_CHECKPOINT_PATH")
    if override:
        checkpoint_path = Path(override).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"LLAMA_CHECKPOINT_PATH does not exist: {checkpoint_path}"
            )
        return checkpoint_path
    return model_dir


def create_model_manager() -> ModelManager:
    model_dir = resolve_model_dir()
    llama_checkpoint_path = get_llama_checkpoint_path(model_dir)
    decoder_checkpoint_path = get_decoder_checkpoint_path(model_dir)
    decoder_config_name = os.environ.get("DECODER_CONFIG_NAME", DEFAULT_DECODER_CONFIG)
    backend = os.environ.get("BACKEND", "cuda")
    default_device = "cpu" if backend == "cpu" else "cuda"

    Path("references").mkdir(parents=True, exist_ok=True)

    logger.info(
        "Initializing Fish Speech Runpod worker with "
        f"llama={llama_checkpoint_path} decoder={decoder_checkpoint_path}"
    )
    return ModelManager(
        mode="tts",
        device=os.environ.get("FISH_DEVICE", default_device),
        half=env_flag("HALF", False),
        compile=env_flag("COMPILE", False),
        llama_checkpoint_path=str(llama_checkpoint_path),
        decoder_checkpoint_path=str(decoder_checkpoint_path),
        decoder_config_name=decoder_config_name,
        warm_up=env_flag("FISH_RUNPOD_WARMUP", True),
    )


def audio_to_int16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


def encode_audio(audio: np.ndarray, sample_rate: int, audio_format: str) -> bytes:
    if audio_format == "pcm":
        return audio_to_int16(audio).tobytes()

    if audio_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()

    segment = AudioSegment(
        data=audio_to_int16(audio).tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    buffer = io.BytesIO()

    if audio_format == "mp3":
        segment.export(
            buffer, format="mp3", bitrate=os.environ.get("FISH_MP3_BITRATE", "192k")
        )
        return buffer.getvalue()

    if audio_format == "opus":
        segment.export(
            buffer,
            format="ogg",
            codec="libopus",
            bitrate=os.environ.get("FISH_OPUS_BITRATE", "96k"),
        )
        return buffer.getvalue()

    raise ValueError(f"Unsupported output format: {audio_format}")


def build_request(job_input: dict) -> ServeTTSRequest:
    if job_input.get("streaming"):
        raise ValueError(
            "Runpod serverless worker only supports non-streaming requests"
        )

    text = job_input.get("text") or job_input.get("prompt")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must include a non-empty 'text' field")

    request_data = {
        "text": text,
        "format": job_input.get(
            "format",
            os.environ.get("FISH_DEFAULT_RESPONSE_FORMAT", "wav"),
        ),
        "latency": job_input.get("latency", "normal"),
        "references": job_input.get("references", []),
        "reference_id": job_input.get("reference_id"),
        "seed": job_input.get("seed"),
        "use_memory_cache": job_input.get(
            "use_memory_cache",
            os.environ.get("FISH_USE_MEMORY_CACHE", "on"),
        ),
        "normalize": job_input.get("normalize", True),
        "streaming": False,
        "max_new_tokens": job_input.get("max_new_tokens", 1024),
        "chunk_length": job_input.get("chunk_length", 200),
        "top_p": job_input.get("top_p", 0.8),
        "repetition_penalty": job_input.get("repetition_penalty", 1.1),
        "temperature": job_input.get("temperature", 0.8),
    }
    return ServeTTSRequest(**request_data)


def run_inference(request: ServeTTSRequest) -> tuple[int, np.ndarray]:
    if MODEL_MANAGER is None:
        raise RuntimeError("Fish Speech model manager is not initialized")

    final_result = None
    for result in MODEL_MANAGER.tts_inference_engine.inference(request):
        if result.code == "error":
            raise RuntimeError(str(result.error))
        if result.code == "final":
            final_result = result.audio

    if final_result is None or not isinstance(final_result, tuple):
        raise RuntimeError("No audio was generated")

    sample_rate, audio = final_result
    return sample_rate, audio


def handler(job):
    if INITIALIZATION_ERROR is not None:
        raise RuntimeError(f"Worker initialization failed: {INITIALIZATION_ERROR}")

    job_input = job.get("input") or {}
    start_time = time.time()

    runpod.serverless.progress_update(job, "validating input")
    try:
        request = build_request(job_input)
    except (ValidationError, ValueError) as exc:
        return {"error": str(exc)}

    if MODEL_MANAGER is None:
        raise RuntimeError("Worker did not finish initializing")

    max_text_length = int(os.environ.get("MAX_TEXT_LENGTH", "0"))
    if max_text_length > 0 and len(request.text) > max_text_length:
        return {"error": f"Text exceeds MAX_TEXT_LENGTH={max_text_length}"}

    runpod.serverless.progress_update(job, "running inference")
    sample_rate, audio = run_inference(request)

    runpod.serverless.progress_update(job, "encoding audio")
    audio_bytes = encode_audio(audio, sample_rate, request.format)
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")

    if len(audio_base64) > MAX_INLINE_BASE64_CHARS:
        return {
            "error": (
                "Generated audio is too large for an inline Runpod response. "
                "Try shorter text or request format='mp3' or format='opus'."
            ),
            "format": request.format,
            "sample_rate": sample_rate,
            "audio_base64_chars": len(audio_base64),
        }

    duration_seconds = float(len(audio) / sample_rate)
    elapsed = time.time() - start_time

    return {
        "format": request.format,
        "sample_rate": sample_rate,
        "audio_base64": audio_base64,
        "audio_bytes": len(audio_bytes),
        "audio_base64_chars": len(audio_base64),
        "duration_seconds": duration_seconds,
        "execution_seconds": elapsed,
        "model_id": os.environ.get("FISH_MODEL_ID", DEFAULT_MODEL_ID),
    }


try:
    MODEL_MANAGER = create_model_manager()
except Exception as exc:
    INITIALIZATION_ERROR = exc
    logger.exception("Failed to initialize Fish Speech Runpod worker")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
