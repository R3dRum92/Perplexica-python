import toml
import os

config_file_name = "config.toml"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

config_file_path = os.path.join(parent_dir, config_file_name)


class Config:
    def __init__(self, config_data):
        self.GENERAL = config_data.get("GENERAL", {})
        self.API_KEYS = config_data.get("API_KEYS", {})
        self.API_ENDPOINTS = config_data.get("API_ENDPOINTS", {})

    @property
    def PORT(self):
        return self.GENERAL.get("PORT", 8080)

    @property
    def SIMILARITY_MEASURE(self):
        return self.GENERAL.get("SIMILARITY MEASURE", "cosine")

    @property
    def KEEP_ALIVE(self):
        return self.GENERAL.get("KEEP_ALIVE", "5m")

    @property
    def OPENAI_API_KEY(self):
        return self.API_KEYS.get("OPENAI", "")

    @property
    def GROQ_API_KEY(self):
        return self.API_KEYS.get("GROQ", "")

    @property
    def ANTHROPIC_API_KEY(self):
        return self.API_KEYS.get("ANTHROPIC", "")

    @property
    def GEMINI_API_KEY(self):
        return self.API_KEYS.get("GEMINI", "")

    @property
    def SEARXNG_API_ENDPOINT(self):
        return self.API_ENDPOINTS.get("SEARXNG", "http://localhost:32768")

    @property
    def OLLAMA_API_ENDPOINT(self):
        return self.API_ENDPOINTS.get("OLLAMA", "")


def load_config():
    config_data = toml.load(config_file_path)
    return Config(config_data=config_data)


def update_config(updated_config):
    current_config = toml.load(config_file_path)

    def deep_update(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and key in target:
                deep_update(target[key], value)
            else:
                target[key] = value

    deep_update(current_config, updated_config)

    with open(config_file_path, "w") as f:
        toml.dump(current_config, f)


config = load_config()


def get_port():
    return config.PORT


def get_searxng_API_endpoint():
    return config.SEARXNG_API_ENDPOINT


def get_anthropic_api_key():
    return config.ANTHROPIC_API_KEY


def get_gemini_api_key():
    return config.GEMINI_API_KEY


def get_groq_api_key():
    return config.GROQ_API_KEY


def get_ollama_api_endpoint():
    return config.OLLAMA_API_ENDPOINT


def get_keep_alive():
    return config.KEEP_ALIVE


def get_openai_api_key():
    return config.OPENAI_API_KEY


def get_similarity_measure():
    return config.SIMILARITY_MEASURE
