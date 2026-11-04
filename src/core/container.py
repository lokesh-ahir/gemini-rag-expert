from typing import Dict, Type, Any, Optional
from functools import lru_cache
import importlib


class ServiceContainer:
    """Dependency injection container for modular services"""

    _instance = None
    _services: Dict[str, Any] = {}
    _configs: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def configure(cls, config: Dict[str, Dict[str, Any]]):
        """Configure services from configuration"""
        cls._configs = config

    @classmethod
    @lru_cache(maxsize=128)
    def get_service(cls, service_name: str) -> Any:
        """Get a service instance (cached)"""
        if service_name not in cls._services:
            if service_name not in cls._configs:
                raise ValueError(f"Service {service_name} not configured")

            config = cls._configs[service_name]
            if not config.get("enabled", True):
                raise ValueError(f"Service {service_name} is disabled")

            module_path = config["module_path"]
            class_name = config["class_name"]
            module_config = config.get("config", {})

            module = importlib.import_module(module_path)
            service_class = getattr(module, class_name)
            cls._services[service_name] = service_class(**module_config)

        return cls._services[service_name]

    @classmethod
    def reset(cls):
        """Reset all cached services"""
        cls._services.clear()
        cls.get_service.cache_clear()

    @classmethod
    def register(cls, service_name: str, instance: Any):
        """Manually register a service instance"""
        cls._services[service_name] = instance
