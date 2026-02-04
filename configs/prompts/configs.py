import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_configs():
    p = Path(os.getenv("CONFIG_YAML_PATH", "configs/prompts/prompts.yaml"))
    version = os.getenv("CONFIG_VERSION", "1.0.0")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    
    current_version_configs = next((item for item in data["versions"] if item["version"] == version), None)
    
    if current_version_configs is not None:
        if current_version_configs.get("is_active") and current_version_configs.get("environment") == os.getenv("APP_ENV"):
            return current_version_configs
        else:
            raise ValueError(f"Prompt version {version} is not active or does not match the environment in {p}")
    
    else: 
        raise ValueError(f"Prompt version {version} not found in {p}")

configs = load_configs()