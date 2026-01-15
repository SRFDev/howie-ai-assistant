import logging
import sys
from llama_index.core import Settings

from config.loader import AppConfig
from config.logger_config import setup_logging
from core.vertex_ai_service import VertexAIService
from core.gcs_service import GCSService 
from backend.models import initialize_global_models


# Setup logger
setup_logging(logger_name="howie", log_level="INFO")
logger = logging.getLogger(__name__)

def main():
    logger.info("--- WAKING UP HOWIE (Deploying Vertex AI Endpoint) ---")
    
    try:
        config = AppConfig()
        initialize_global_models(config)
        storage = GCSService(config)
        vertex_service = VertexAIService(config, Settings.embed_model, storage)
        vertex_service.wake()

        logger.info("Successfully woke Howie up.")

    except Exception as e:
        logger.error(f"Failed to wake up Howie: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()