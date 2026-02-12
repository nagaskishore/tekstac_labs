"""
Simple Configuration Management
Loads API keys from .env file or OS environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Default .env file location
ENV_FILE_PATH = Path(__file__).parent / ".env"

def load_config(use_env_file=True):
    """
    Load configuration from .env file or OS environment
    
    Args:
        use_env_file: If True, loads from .env file. If False, uses only OS environment
    """
    if use_env_file and ENV_FILE_PATH.exists():
        load_dotenv(ENV_FILE_PATH)
        print(f"? Loaded config from: {ENV_FILE_PATH}")
    else:
        print("? Using OS environment variables")

def get_api_key(key_name, required=True):
    """Get API key from environment with optional validation"""
    value = os.getenv(key_name)
    if required and not value:
        raise ValueError(f"Required API key '{key_name}' not found in environment")
    return value

# Load configuration (call once when module is imported)
load_config()

# API Keys
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
OPENAI_BASE_URL = get_api_key("OPENAI_BASE_URL", required=False) or "https://api.openai.com/v1"
TAVILY_API_KEY = get_api_key("TAVILY_API_KEY", required=False)
AMADEUS_CLIENT_ID = get_api_key("AMADEUS_CLIENT_ID", required=False)
AMADEUS_CLIENT_SECRET = get_api_key("AMADEUS_CLIENT_SECRET", required=False)
LANGSMITH_API_KEY = get_api_key("LANGSMITH_API_KEY", required=False)
OPEN_WEATHER_API_KEY = get_api_key("OPEN_WEATHER_API_KEY", required=False)

# Model Configuration
MODELS = {
    "default": "gpt-4o-mini",
    "fast": "gpt-5-nano", 
    "smart": "gpt-5-mini",
    "mini": "gpt-4.1-mini",
    "embedding": "text-embedding-3-small"
}

MODEL_PARAMS = {
        "gpt-4o-mini": {"temperature": 0.7, "max_tokens": 1000},
        "gpt-4.1-mini": {"temperature": 0.7, "max_tokens": 1000},
        "gpt-5-nano": {"temperature": 1.0, "max_tokens": 1000},  
        "gpt-5-mini": {"temperature": 1.0, "max_tokens": 1000}, 
        "text-embedding-3-small": {}  
    }


def get_model(model_type="default"):
    """Get model name by type"""
    return MODELS.get(model_type, MODELS["default"])

def get_model_params(model_name):
    """Get parameters for a specific model"""
    return MODEL_PARAMS.get(model_name, {"temperature": 0.7, "max_tokens": 2048})

# OpenAI Client Configuration
def get_openai_config():
    """Get OpenAI client configuration"""
    return {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "timeout": 100
    }

if __name__ == "__main__":
    print("TravelMate Configuration")
    print(f"OpenAI Base URL: {OPENAI_BASE_URL}")
    print(f"Default Model: {get_model()}")
    print(f"Available Models: {list(MODELS.values())}")
    print(f"API Keys loaded: {bool(OPENAI_API_KEY)}")