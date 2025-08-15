"""
This module contains constant values used throughout the "Howie" application.
It serves as a single source of truth for conventional paths, filenames, and default values.
"""

# --- Directory Paths (relative to the project root) ---
CONFIG_DIR = "config"
PROMPTS_DIR = "prompts"
DATA_DIR = "data"
CACHE_DIR = ".cache"


# --- Filenames ---
CONFIG_FILE_NAME = "config.toml"
PROMPTS_FILE_NAME = "prompts.toml"
DOCSTORE_FILE_NAME = "docstore.json"
VIDEO_SUMMARY_CACHE_FILE_NAME = "steves-pour-over-method.mp4.summary.json"
INGESTION_MANIFEST_FILE_NAME = "ingestion_manifest.json"

# --- Full Paths (constructed for convenience) ---
# Note: These assume the application is run from the project root.
import os

# Construct full paths to config and prompt files
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)
PROMPTS_FILE_PATH = os.path.join(PROMPTS_DIR, PROMPTS_FILE_NAME)

# Construct full paths for cache files
CACHE_DOCSTORE_PATH = os.path.join(CACHE_DIR, DOCSTORE_FILE_NAME)
CACHE_VIDEO_SUMMARY_PATH = os.path.join(CACHE_DIR, VIDEO_SUMMARY_CACHE_FILE_NAME)
CACHE_INGESTION_MANIFEST_PATH = os.path.join(CACHE_DIR, INGESTION_MANIFEST_FILE_NAME)
