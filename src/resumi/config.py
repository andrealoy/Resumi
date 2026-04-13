from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Resumi", alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    openai_api_key: str = Field(default="replace-me", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5.4", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    openai_embedding_dimensions: int = Field(
        default=512, alias="OPENAI_EMBEDDING_DIMENSIONS"
    )
    openai_embedding_batch_size: int = Field(
        default=32, alias="OPENAI_EMBEDDING_BATCH_SIZE"
    )

    docs_root: str = Field(default="docs", alias="DOCS_ROOT")
    db_path: str = Field(default=".data/documents.db", alias="DB_PATH")
    faiss_index_dir: str = Field(default=".data/faiss", alias="FAISS_INDEX_DIR")
    rag_chunk_size: int = Field(default=160, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=40, alias="RAG_CHUNK_OVERLAP")
    rag_search_limit: int = Field(default=8, alias="RAG_SEARCH_LIMIT")
    rag_candidate_pool: int = Field(default=24, alias="RAG_CANDIDATE_POOL")

    gmail_client_secrets_file: str = Field(
        default="credentials/gmail-client-secret.json",
        alias="GMAIL_CLIENT_SECRETS_FILE",
    )
    gmail_token_file: str = Field(
        default="credentials/gmail-token.json", alias="GMAIL_TOKEN_FILE"
    )
    gmail_query: str = Field(default="in:anywhere", alias="GMAIL_QUERY")
    gmail_user_id: str = Field(default="me", alias="GMAIL_USER_ID")
    gmail_max_results: int = Field(default=100, alias="GMAIL_MAX_RESULTS")

    google_calendar_id: str = Field(default="primary", alias="GOOGLE_CALENDAR_ID")
    google_service_account_file: str = Field(
        default="credentials/google-service-account.json",
        alias="GOOGLE_SERVICE_ACCOUNT_FILE",
    )

    gradio_path: str = Field(default="/gradio", alias="GRADIO_PATH")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
