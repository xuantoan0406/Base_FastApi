from fastapi import FastAPI
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

from api.v1.endpoints import routers
from core.config import settings

app = FastAPI(
    title="GenBot",
    description="generate images",
    # docs_url=True
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routers.generateImage)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
