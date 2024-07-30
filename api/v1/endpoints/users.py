from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model, Blip2ForConditionalGeneration, AutoTokenizer
import torch
from langchain_qdrant import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient, models


class ConnectQdrant:
    def __init__(self, host, port, collection_name):
        self.client = QdrantClient(host="localhost", port=6333)
        self.client.create_collection(
            collection_name="Test",  # Name of the collection
            vectors_config=models.VectorParams(
                size=128,  # Size of the vectors to be stored
                distance=models.Distance.COSINE  # Distance metric for vector similarity
            )
        )
        print("success")

    def disconnect(self):
        self.client.delete_collection("my_collection")
        print("success")

    def insert(self, data):
        self.client.insert_documents(
            collection_name="Test1",
            documents=[
                {"id": data["id"], "vector": data["vector"]}
            ]
        )


z = ConnectQdrant("localhost", 8000, "sda")


#
class ImageCaption:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        )
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    def captions(self, path_image):
        image = Image.open(path_image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text

# a = ImageCaption()
# a.textEmbedding("Analyses for accuracy driven model design")
# caption = a.captions("assets/images/00001-3171169063.png")
# print(caption)
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
# )
# url = "/home/toan-dx/recommend-system/00001-3171169063.png"
# image = Image.open("12602-925960129.jpg")
# print(image)
# prompt = "Detailed description"
# inputs = processor(images=image, return_tensors="pt").to(device)
#
# outputs = model.generate(**inputs)
# generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
# print(generated_text)
#
# tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
# inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")
# text_features = model.get_text_features(**inputs)
# from transformers import BlipForConditionalGeneration, AutoProcessor
# device="cuda"
# model = BlipForConditionalGeneration.from_pretrained("dblasko/blip-dalle3-img2prompt").to(device)
# processor = AutoProcessor.from_pretrained("dblasko/blip-dalle3-img2prompt")
#
# # from datasets import load_dataset
# #
# # dataset = load_dataset("laion/dalle-3-dataset", split=f'train[0%:1%]') # for fast download time in the toy example
# # example = dataset[img_index][0]
# # image = example["image"]
# # caption = example["caption"]
# image=Image.open("assets/images/00000-614204843.png")
# inputs = processor(images=image, return_tensors="pt").to(device)
# pixel_values = inputs.pixel_values
#
# generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#
# print(f"Generated caption: {generated_caption}")
# from qdrant_client import QdrantClient, models
#
# # Initialize Qdrant client
# client = QdrantClient(host="localhost", port=6333)
#
# # Create a collection
# client.create_collection(
#     collection_name="my_collection",  # Name of the collection
#     vectors_config=models.VectorParams(
#         size=100,  # Size of the vectors to be stored
#         distance=models.Distance.COSINE  # Distance metric for vector similarity
#     )
# )
#
# print("Collection created successfully.")
