from config.loader import AppConfig
from core import constants
import json
import os
import logging


logger = logging.getLogger(__name__)

def load_manifest(config: AppConfig):
    """
    Loads the manifest file from the specified path.
    Returns the parsed JSON content.
    """
    manifest_path = constants.CACHE_INGESTION_MANIFEST_PATH
    if not os.path.exists(manifest_path):
        logger.info(f"Manifest file not found at {manifest_path}.")
        return None
    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
    return manifest_data

def save_manifest(config: AppConfig, data: dict):
    """
    Saves the provided data to the manifest file at the specified path.
    """
    manifest_path = constants.CACHE_INGESTION_MANIFEST_PATH
    with open(manifest_path, 'w') as f:
        json.dump(data, f, indent=2)
