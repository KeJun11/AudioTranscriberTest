from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from utils import AudioTranscriber
import os
import logging
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Audio Transcriber API")

# CORS MIDDLEWARE
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/transcribe/")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model_size: str = "base",
    chunk_minutes: int = 10,
    language: str = "en",
    compute_type: str = "int8",
    device: str = "cpu"
):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{audio_file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        # Create transcriber instance
        transcriber = AudioTranscriber(
            chunk_length_minutes=chunk_minutes,
            model_size=model_size,
            language=language,
            compute_type=compute_type,
            device=device,
        )

        # Process audio file
        output_file, num_chunks = transcriber.process_audio_file(temp_path)

        # Clean up temp file
        os.remove(temp_path)

        # Read transcript
        with open(output_file, "r", encoding="utf-8") as f:
            transcript = f.read()

        return {
            "transcript": transcript,
            "output_file": output_file,
            "status": f"Processed {num_chunks} chunks successfully!"
        }

    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download/output/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)