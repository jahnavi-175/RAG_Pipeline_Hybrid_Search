"""
Utility: list which Gemini models your API key currently has access to,
and which methods (generateContent, embedContent) each supports.

Model names get renamed/retired over time, so if config.py's model names
ever 404 again, run this first to see current valid options instead of guessing.

Run:
    python list_models.py
"""
import config
import google.generativeai as genai

genai.configure(api_key=config.GEMINI_API_KEY)

print("Models available to your API key:\n")
for model in genai.list_models():
    print(f"- {model.name}")
    print(f"    supports: {model.supported_generation_methods}")