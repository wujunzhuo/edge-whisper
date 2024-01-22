import os
import tempfile
import threading
import traceback
import aiofiles
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pyAudioAnalysis.audioBasicIO import read_audio_file
from pydub.silence import split_on_silence
from pydub import AudioSegment
from whisper import load_model, transcribe


WHISPER_MODEL = "base"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model(WHISPER_MODEL)

lock = threading.Lock()


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, file.filename)

            async with aiofiles.open(path, 'wb') as out_file:
                while content := await file.read(1024*30):
                    await out_file.write(content)

            rate, audio = read_audio_file(path)

            aud = AudioSegment(
                audio.tobytes(), frame_rate=rate,
                sample_width=audio.dtype.itemsize,
                channels=1)
            audio_chunks = split_on_silence(
                aud,
                min_silence_len=100,
                silence_thresh=-45,
                keep_silence=20)
            if not audio_chunks:
                return {"transcription": ""}

            with lock:
                result = transcribe(model, path, task="translate")

        print(result)

        return {
            "transcription": result["text"],
            "temperature": result["segments"][0]["temperature"],
            "no_speech_prob": result["segments"][0]["no_speech_prob"],
            "language": result["language"],
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
