import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-flash-lite-latest"  # si da error de modelo, mira el nombre vigente en AI Studio

resp = client.models.generate_content(model=MODEL, contents="Di hola en una frase.")
print(resp.text)