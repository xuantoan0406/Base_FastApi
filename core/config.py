from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    HOST: str = Field(..., env="HOST")
    PORT: str = Field(..., env="PORT")
    W_MODEL_BLIP: str = Field(..., env="W_MODEL_BLIP")
    W_MODEL_FEATURE: str = Field(..., env="W_MODEL_FEATURE")

settings = Settings()
