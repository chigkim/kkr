'''Compatibility for misaki 0.9.x with phonemizer 3.3 and espeakng-loader.'''

from __future__ import annotations

import os


def configure_espeak() -> None:
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', espeakng_loader.get_library_path())
    os.environ.setdefault('ESPEAK_DATA_PATH', espeakng_loader.get_data_path())

    if not hasattr(EspeakWrapper, 'set_data_path'):
        @classmethod
        def set_data_path(cls, data_path):
            os.environ['ESPEAK_DATA_PATH'] = str(data_path)

        EspeakWrapper.set_data_path = set_data_path
