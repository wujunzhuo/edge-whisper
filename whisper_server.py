import asyncio
import json
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


WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
WHISPERCPP_BIN = os.getenv('WHISPERCPP_BIN', './whispercpp_main')
WHISPERCPP_MODEL = os.getenv('WHISPERCPP_MODEL', './ggml-base-q5_1.bin')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:8000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


if WHISPER_MODEL:
    whisper_model = load_model(WHISPER_MODEL)

lock = threading.Lock()


async def run_model(input_path: str, result_path: str) -> dict[str]:
    if WHISPER_MODEL:
        result = transcribe(whisper_model, input_path, task='translate')
        return {
            'transcription': result['text'],
            'language': result['language'],
        }

    proc = await asyncio.create_subprocess_shell(
        f'{WHISPERCPP_BIN} -m {WHISPERCPP_MODEL} -l auto \
            -oj -of {result_path} {input_path}',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()

    print(f'whisper.cpp:\n{stdout.decode()}')

    ret_code = proc.returncode or 0
    if ret_code != 0:
        raise RuntimeError(f'whisper.cpp [{ret_code}]:\n{stderr.decode()}')

    async with aiofiles.open(result_path + '.json', 'r') as f:
        content = await f.read()
        result = json.loads(content)

    return {
        'transcription': result["transcription"][0]['text'],
        'language': result["result"]['language'],
    }


@app.post('/transcribe')
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, file.filename)

            async with aiofiles.open(input_path, 'wb') as f:
                while content := await file.read(1024*30):
                    await f.write(content)

            rate, audio = read_audio_file(input_path)

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
                return {'transcription': ''}

            with lock:
                result_path = os.path.join(tmpdir, 'result')
                result = await run_model(input_path, result_path)

        print(result)
        return result

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'message': str(e)})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
