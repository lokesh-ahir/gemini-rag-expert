"""
AI Module Configuration
Central configuration for the entire AI module.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration"""
    url: str = os.getenv("AI_MODULE_DB_URL", "postgresql://localhost/ai_module")
    pool_size: int = 10
    pool_timeout: int = 30


@dataclass
class QdrantConfig:
    """Qdrant vector store configuration"""
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "circuit_components")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    vector_size: int = 3072


@dataclass
class ModelConfig:
    """LLM model configuration"""
    gemini_flash_id: str = os.getenv("GEMINI_FLASH_MODEL", "gemini-1.5-flash")
    gemini_neon4_id: str = os.getenv("GEMINI_NEON4_MODEL", "gemini-neon-4-pro")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class AIModuleConfig:
    """Complete AI module configuration"""
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    models: ModelConfig = ModelConfig()
    
    # Chat settings
    messages_per_page: int = 25
    max_chat_history: int = 1000
    
    # Circuit settings
    max_components_per_circuit: int = 50
    max_connections_per_circuit: int = 100
    
    @classmethod
    def from_env(cls) -> "AIModuleConfig":
        """Load configuration from environment variables"""
        return cls()
    
    def validate(self) -> None:
        """Validate required configuration"""
        required = []
        
        if not self.models.google_api_key:
            required.append("GOOGLE_API_KEY")
        
        if not self.database.url:
            required.append("AI_MODULE_DB_URL")
        
        if required:
            raise ValueError(f"Missing required environment variables: {', '.join(required)}")


# Global config instance
_config: Optional[AIModuleConfig] = None


def get_config() -> AIModuleConfig:
    """Get the global configuration"""
    global _config
    if _config is None:
        _config = AIModuleConfig.from_env()
    return _config


def set_config(config: AIModuleConfig) -> None:
    """Set the global configuration"""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration"""
    global _config
    _config = None
