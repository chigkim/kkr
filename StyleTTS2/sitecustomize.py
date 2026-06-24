'''Runtime compatibility for misaki + phonemizer + espeakng-loader.'''

from __future__ import annotations

import os


def _configure_espeak() -> None:
    try:
        import espeakng_loader
    except Exception:
        return

    os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', espeakng_loader.get_library_path())
    os.environ.setdefault('ESPEAK_DATA_PATH', espeakng_loader.get_data_path())

    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except Exception:
        return

    if not hasattr(EspeakWrapper, 'set_data_path'):
        @classmethod
        def set_data_path(cls, data_path):
            os.environ['ESPEAK_DATA_PATH'] = str(data_path)

        EspeakWrapper.set_data_path = set_data_path


_configure_espeak()
