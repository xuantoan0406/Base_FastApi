from fastapi import APIRouter
from api.v1.endpoints.imageGenerate import image_generate, hello_wold

generateImage = APIRouter()

generateImage.post("/get-image")(image_generate)
generateImage.get("/")(hello_wold)
