"""FastAPI server for Kani TTS with streaming support, Web UI and Voice Gallery"""

import io
import time
import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from scipy.io.wavfile import write as wav_write
import torch

# kani-tts imports
from audio import LLMAudioPlayer, StreamingAudioWriter
from generation import TTSGenerator
from config import CHUNK_SIZE, LOOKBACK_FRAMES, TEMPERATURE, TOP_P, MAX_TOKENS

# Zero-shot cloning imports (from kani_tts package)
try:
    from kani_tts.speaker.speaker_embedder import SpeakerEmbedder
except ImportError:
    # Fallback/alternative import if structure differs
    SpeakerEmbedder = None

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

app = FastAPI(title="Kani TTS AI Studio", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
VOICES_DIR = "/app/voices"
if not os.path.exists(VOICES_DIR):
    os.makedirs(VOICES_DIR)

# Global models
generator = None
player = None
embedder = None

class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    top_p: Optional[float] = TOP_P
    chunk_size: Optional[int] = CHUNK_SIZE
    lookback_frames: Optional[int] = LOOKBACK_FRAMES
    voice_name: Optional[str] = None # Name of the saved voice to use

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global generator, player, embedder
    print("🚀 Initializing TTS models...")
    try:
        generator = TTSGenerator()
        player = LLMAudioPlayer(generator.tokenizer)
        if SpeakerEmbedder:
            print("🎙️ Initializing Speaker Embedder...")
            embedder = SpeakerEmbedder()
        print("✅ TTS models initialized successfully!")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

@app.get("/health")
async def health_check():
    """Check if server is ready"""
    return {
        "status": "healthy" if generator is not None else "initializing",
        "tts_initialized": generator is not None and player is not None,
        "cloning_available": embedder is not None
    }

@app.get("/voices")
async def list_voices():
    """List all saved voice profiles"""
    voices = []
    for f in os.listdir(VOICES_DIR):
        if f.endswith(".npy"):
            voices.append(f[:-4])
    return sorted(voices)

@app.post("/voices")
async def upload_voice(file: UploadFile = File(...), name: str = Form(...)):
    """Compute and save a speaker embedding from an audio file"""
    if not embedder:
        raise HTTPException(status_code=501, detail="Speaker Embedder not available")
    
    if not name or not name.isalnum():
        raise HTTPException(status_code=400, detail="Invalid name. Use alphanumeric characters only.")

    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📡 Computing embedding for: {name}")
        embedding = embedder.embed_audio_file(temp_path)
        
        # Save as numpy array
        save_path = os.path.join(VOICES_DIR, f"{name}.npy")
        np.save(save_path, embedding)
        
        return {"status": "success", "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.delete("/voices/{name}")
async def delete_voice(name: str):
    """Delete a saved voice profile"""
    file_path = os.path.join(VOICES_DIR, f"{name}.npy")
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Voice not found")

def get_speaker_emb(voice_name: Optional[str]):
    """Load speaker embedding if requested"""
    if not voice_name:
        return None
    file_path = os.path.join(VOICES_DIR, f"{voice_name}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    return None

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """Generate complete audio file (non-streaming)"""
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    try:
        speaker_emb = get_speaker_emb(request.voice_name)
        
        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames
        )
        audio_writer.start()

        generator.generate(
            request.text,
            audio_writer,
            max_tokens=request.max_tokens,
            speaker_emb=speaker_emb
        )

        audio_writer.finalize()

        if not audio_writer.audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        full_audio = np.concatenate(audio_writer.audio_chunks)
        wav_buffer = io.BytesIO()
        wav_write(wav_buffer, 22050, full_audio)
        wav_buffer.seek(0)

        return Response(
            content=wav_buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech_{request.voice_name or 'default'}.wav"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-tts")
async def stream_speech(request: TTSRequest):
    """Stream audio chunks as they're generated"""
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    import queue
    import threading
    import struct

    speaker_emb = get_speaker_emb(request.voice_name)

    async def audio_chunk_generator():
        chunk_queue = queue.Queue()
        class ChunkList(list):
            def append(self, chunk):
                super().append(chunk)
                chunk_queue.put(("chunk", chunk))

        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames
        )
        audio_writer.audio_chunks = ChunkList()

        def generate():
            try:
                audio_writer.start()
                generator.generate(
                    request.text, 
                    audio_writer, 
                    max_tokens=request.max_tokens,
                    speaker_emb=speaker_emb
                )
                audio_writer.finalize()
                chunk_queue.put(("done", None))
            except Exception as e:
                chunk_queue.put(("error", str(e)))

        gen_thread = threading.Thread(target=generate)
        gen_thread.start()

        try:
            while True:
                msg_type, data = chunk_queue.get(timeout=60)
                if msg_type == "chunk":
                    pcm_data = (data * 32767).astype(np.int16)
                    chunk_bytes = pcm_data.tobytes()
                    length_prefix = struct.pack('<I', len(chunk_bytes))
                    yield length_prefix + chunk_bytes
                elif msg_type == "done":
                    yield struct.pack('<I', 0)
                    break
                elif msg_type == "error":
                    yield struct.pack('<I', 0xFFFFFFFF)
                    break
        finally:
            gen_thread.join()

    return StreamingResponse(
        audio_chunk_generator(),
        media_type="application/octet-stream"
    )

@app.get("/")
async def root():
    """Serve the Web UI"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Kani TTS Server is running. index.html not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
