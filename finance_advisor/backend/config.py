from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    # ✅ Groq Config
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama3-70b-8192", env="GROQ_MODEL")

    # ✅ Embeddings (local)
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # ✅ Optional app settings
    debug: bool = Field(True, env="DEBUG")
    allowed_origins: str = Field("*", env="ALLOWED_ORIGINS")

    class Config:
        env_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..", ".env")
        )
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()