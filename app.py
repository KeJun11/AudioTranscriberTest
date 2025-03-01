# app.py
import gradio as gr
import os
from utils import AudioTranscriber
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_file, model_size, chunk_minutes, language, compute_type, device
):
    """Process audio file and return transcription"""
    try:
        # Create transcriber with selected options
        transcriber = AudioTranscriber(
            chunk_length_minutes=chunk_minutes,
            model_size=model_size,
            language=language,
            compute_type=compute_type,
            device=device,
        )

        # Process the audio file
        output_file, num_chunks = transcriber.process_audio_file(audio_file)

        # Read the transcription
        with open(output_file, "r", encoding="utf-8") as f:
            transcript = f.read()

        return transcript, output_file, f"Processed {num_chunks} chunks successfully!"

    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return None, None, f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Audio Transcriber") as demo:
    gr.Markdown("# Audio Transcription with Faster Whisper")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.File(label="Upload Audio File", file_types=["audio"])

            with gr.Group():
                model_size = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                    value="medium",
                    label="Model Size",
                )
                chunk_minutes = gr.Slider(
                    minimum=1,
                    maximum=60,
                    value=30,
                    step=1,
                    label="Chunk Length (minutes)",
                )
                language = gr.Dropdown(
                    choices=["auto", "en", "fr", "de", "es", "it", "ja", "zh", "ru"],
                    value="en",
                    label="Language",
                )
                compute_type = gr.Dropdown(
                    choices=["float16", "float32", "int8"],
                    value="float16",
                    label="Compute Type",
                )
                device = gr.Dropdown(
                    choices=["cuda", "cpu"], value="cuda", label="Device"
                )

            transcribe_btn = gr.Button("Transcribe Audio", variant="primary")

        with gr.Column(scale=2):
            text_output = gr.Markdown(label="Transcription")
            file_output = gr.File(label="Download Transcript")
            status_output = gr.Textbox(label="Status")

    transcribe_btn.click(
        transcribe_audio,
        inputs=[audio_input, model_size, chunk_minutes, language, compute_type, device],
        outputs=[text_output, file_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
