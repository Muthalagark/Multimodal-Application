!pip install gradio
!pip install deep_translator
!pip install groq
!pip install transformers
!pip install gradio diffusers
!pip install gradio diffusers
!pip install gradio transformers datasets torch
!pip install gradio
!pip install deep_translator
!pip install groq
!pip install transformers
!pip install gradio diffusers
!pip install gradio diffusers
!pip install gradio transformers datasets torch
!pip install torch transformers gradio
!pip install diffusers transformers scipy ftfy accelerate
!pip install torch transformers gradio
!pip install huggingface_hub
!pip install accelerate

import accelerate
import gradio as gr
from transformers import pipeline
from datasets import load_dataset
import torch
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import warnings
from diffusers import StableDiffusionPipeline
from huggingface_hub import login

warnings.filterwarnings("ignore")

class TamilSpeechTranslator:
    def __init__(self):
        print("Loading models...")

        # Initialize Whisper for Tamil speech recognition
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"task": "transcribe", "language": "tamil"}
        )

        # Initialize translation model
        self.translator_name = "facebook/nllb-200-distilled-600M"
        print(f"Loading translator model: {self.translator_name}")
        self.translator = pipeline(
            "translation",
            model=self.translator_name,
            device=0 if torch.cuda.is_available() else -1
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Models loaded successfully. Using device: {self.device}")

        # Initialize Stable Diffusion for text-to-image generation
        self.image_generator = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=True # Replace with your Hugging Face token if needed
        ).to(self.device)

        # Initialize text generation model
        self.text_generator = pipeline(
            "text-generation",
            model="EleutherAI/gpt-neo-1.3B",
            device=0 if torch.cuda.is_available() else -1
        )


    def transcribe_audio(self, audio):
        try:
            # Transcribe audio to Tamil text
            result = self.transcriber(
                audio,
                generate_kwargs={"task": "transcribe", "language": "tamil"}
            )
            return result["text"]
        except Exception as e:
            return f"Transcription error: {str(e)}"

    def translate_text(self, text):
        try:
            # Translate Tamil to English using NLLB model
            translation = self.translator(
                text,
                src_lang="tam_Taml",
                tgt_lang="eng_Latn"
            )[0]["translation_text"]
            return translation
        except Exception as e:
            return f"Translation error: {str(e)}"

    def generate_image(self, prompt):
        try:
            image = self.image_generator(prompt).images[0]
            return image
        except Exception as e:
            return f"Image generation error: {str(e)}"

    def generate_creative_text(self, prompt):
        try:
            generated_text = self.text_generator(prompt, max_length=100)[0]["generated_text"]
            return generated_text
        except Exception as e:
            return f"Creative text generation error: {str(e)}"

    def process_audio(self, audio):
        if audio is None:
            return "Please upload an audio file.", "No translation available.", None, None

        # First transcribe
        tamil_text = self.transcribe_audio(audio)

        # Then translate
        english_text = self.translate_text(tamil_text)

        # Generate image from the translation
        image = self.generate_image(english_text)

        # Generate creative text based on the translation
        creative_text = self.generate_creative_text(english_text)

        return tamil_text, english_text, image, creative_text

def create_interface():
    # Initialize the translator
    translator = TamilSpeechTranslator()

    return gr.Interface(
        fn=translator.process_audio,
        inputs=gr.Audio(type="filepath"),
        outputs=[
            gr.Textbox(label="Tamil Transcription"),
            gr.Textbox(label="English Translation"),
            gr.Image(label="Generated Image"),
            gr.Textbox(label="Creative Text")
        ],
        title="Tamil Speech to English Text & Image & Creative Text Translator",
        description="""Upload a Tamil audio file to:
        1. Transcribe it to Tamil text
        2. Translate the Tamil text to English
        3. Generate an image based on the English translation
        4. Generate creative text based on the English translation

        Note: Best results with clear audio and standard Tamil."""
    )

def main():
    interface = create_interface()
    interface.launch()

if __name__ == "__main__":
    main()
