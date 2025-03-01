# utils.py
from pydub import AudioSegment
import os
from faster_whisper import WhisperModel
from tqdm import tqdm
import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioTranscriber:
    """A class for transcribing audio files using Faster Whisper."""

    SUPPORTED_FORMATS = (".mp3", ".m4a", ".wav", ".flac")

    def __init__(
        self,
        chunk_length_minutes: int = 30,
        model_size: str = "medium",
        language: str = "en",
        compute_type: str = "float16",
        device: str = "cuda",
    ):
        """
        Initialize the AudioTranscriber.

        Args:
            chunk_length_minutes: Length of each audio chunk in minutes
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large', 'large-v2')
            language: Force transcription language (e.g., 'en' for English)
            compute_type: Compute type for model execution ('float16', 'float32', 'int8')
            device: Device to use for inference ('cuda', 'cpu')
        """
        self.chunk_length_ms = chunk_length_minutes * 60 * 1000
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type
        self.device = device

        # Create necessary directories
        Path("chunks").mkdir(exist_ok=True)
        Path("transcripts").mkdir(exist_ok=True)
        Path("output").mkdir(exist_ok=True)

        # Initialize model
        self._model = None

    @property
    def model(self):
        """Lazy-load the model only when needed."""
        if self._model is None:
            logger.info(f"Loading Faster Whisper model: {self.model_size}")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def process_audio_file(self, input_file: str) -> Tuple[str, int]:
        """
        Process an audio file: split, transcribe, and merge results.

        Args:
            input_file: Path to the input audio file

        Returns:
            Tuple containing output file path and number of chunks processed
        """
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Audio file not found: {input_file}")

        if not input_file.lower().endswith(self.SUPPORTED_FORMATS):
            raise ValueError(
                f"Unsupported file format. Supported formats: {self.SUPPORTED_FORMATS}"
            )

        # Clear previous chunks and transcripts
        self._clear_directory("chunks")
        self._clear_directory("transcripts")

        # Split audio into chunks
        num_chunks = self.split_audio(input_file)

        # Transcribe all chunks
        transcripts = self.transcribe_chunks()

        # Merge transcripts
        output_file = self.merge_transcripts(transcripts)

        return output_file, num_chunks

    def split_audio(self, input_file: str) -> int:
        """
        Split the input audio file into chunks.

        Args:
            input_file: Path to the input audio file

        Returns:
            Number of chunks created
        """
        try:
            logger.info(f"Loading audio file: {input_file}")
            audio = AudioSegment.from_file(input_file)

            logger.info(f"Splitting audio file into {self.chunk_length_ms}ms chunks")
            chunks = [
                audio[i : i + self.chunk_length_ms]
                for i in range(0, len(audio), self.chunk_length_ms)
            ]

            for i, chunk in enumerate(chunks):
                output_path = os.path.join("chunks", f"chunk_{i+1:03d}.mp3")
                chunk.export(output_path, format="mp3")

            logger.info(f"Created {len(chunks)} audio chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error splitting audio: {str(e)}")
            raise

    def transcribe_chunks(self, progress: Optional[gr.Progress] = None) -> List[str]:
        """
        Transcribe all audio chunks in the chunks directory.

        Args:
            progress: Optional Gradio progress tracker

        Returns:
            List of transcription texts
        """
        try:
            transcripts = []

            # Get all audio files and sort them numerically
            chunk_files = [
                f
                for f in os.listdir("chunks")
                if f.lower().endswith(self.SUPPORTED_FORMATS)
            ]
            chunk_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

            if not chunk_files:
                logger.warning("No audio chunks found to transcribe")
                return []

            # Create progress iterator
            if progress:
                chunk_iter = progress.tqdm(chunk_files, desc="Transcribing chunks")
            else:
                chunk_iter = tqdm(chunk_files, desc="Transcribing chunks")

            for chunk_file in chunk_iter:
                input_path = os.path.join("chunks", chunk_file)
                output_path = os.path.join(
                    "transcripts", f"{os.path.splitext(chunk_file)[0]}.txt"
                )

                # Transcribe with Faster Whisper
                segments, _ = self.model.transcribe(
                    input_path,
                    language=self.language,
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True,  # Filter out audio without speech
                    vad_parameters=dict(
                        min_silence_duration_ms=500
                    ),  # Adjust as needed
                )

                # Collect all segment texts
                segment_texts = [segment.text for segment in segments]
                transcript = " ".join(segment_texts).strip()

                # Save transcript to file
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcript)

                transcripts.append(transcript)
                logger.info(f"Transcribed: {chunk_file}")

            return transcripts

        except Exception as e:
            logger.error(f"Error transcribing chunks: {str(e)}")
            raise

    def merge_transcripts(self, transcripts: List[str]) -> str:
        """
        Merge all transcripts into a single markdown file.

        Args:
            transcripts: List of transcript texts

        Returns:
            Path to the output markdown file
        """
        try:
            output_file = os.path.join("output", "full_transcript.md")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# Full Transcript\n\n")
                for i, transcript in enumerate(transcripts, 1):
                    f.write(f"## Part {i}\n\n{transcript}\n\n")

            logger.info(f"Created merged transcript: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error merging transcripts: {str(e)}")
            raise

    def _clear_directory(self, directory: str) -> None:
        """
        Clear all files in the specified directory.

        Args:
            directory: Directory to clear
        """
        dir_path = Path(directory)
        if dir_path.exists():
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info(f"Cleared directory: {directory}")
