__version__ = '0.9.4'

import os

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

try:
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', espeakng_loader.get_library_path())
    os.environ.setdefault('ESPEAK_DATA_PATH', espeakng_loader.get_data_path())
    if not hasattr(EspeakWrapper, 'set_data_path'):
        @classmethod
        def set_data_path(cls, data_path):
            os.environ['ESPEAK_DATA_PATH'] = str(data_path)

        EspeakWrapper.set_data_path = set_data_path
except Exception:
    pass

from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add custom handler with clean format including module and line number
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <cyan>{module:>16}:{line}</cyan> | <level>{level: >8}</level> | <level>{message}</level>",
    colorize=True,
    level="INFO" # "DEBUG" to enable logger.debug("message") and up prints 
                 # "ERROR" to enable only logger.error("message") prints
                 # etc
)

# Disable before release or as needed
logger.disable("kokoro")

from .model import KModel
from .pipeline import KPipeline
