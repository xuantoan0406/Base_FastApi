from transformers import AutoImageProcessor, Swinv2ForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
# dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = Image.open("assets/images/12602-925960129.jpg")
image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

inputs = image_processor(image, return_tensors="pt")
print(inputs)
with torch.no_grad():
    logits = model(**inputs).logits
print(logits)
# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])