import base64
import io
import ollama
from PIL import Image
from config import OLLAMA_HOST, OLLAMA_MODEL

_client = ollama.Client(host=OLLAMA_HOST)

