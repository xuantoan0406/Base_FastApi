from PIL import Image
import requests
from langchain.schema import embeddings
from transformers import Blip2Processor, Blip2Model, Blip2ForConditionalGeneration, AutoTokenizer
import torch
from core.config import settings
from langchain_qdrant import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient, models


#
class ImageCaption:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            settings.W_MODEL_BLIP, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        )
        self.processor = Blip2Processor.from_pretrained(settings.W_MODEL_BLIP)

    def captions(self, path_image):
        image = Image.open(path_image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text


