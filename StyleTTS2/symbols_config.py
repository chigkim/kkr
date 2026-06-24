import json
import os
import os.path as osp


def resolve_path(path, config_path=None):
    if not path:
        return None
    if osp.isabs(path):
        return path
    if config_path:
        return osp.abspath(osp.join(osp.dirname(config_path), path))
    return osp.abspath(path)


def apply_symbols_config(config, config_path=None):
    text_params = config.get('text_params', {}) or {}
    symbols_path = text_params.get('symbols_path') or config.get('symbols_path')
    if not symbols_path:
        return None

    symbols_path = resolve_path(symbols_path, config_path)
    with open(symbols_path, encoding='utf-8') as f:
        data = json.load(f)
    symbols = data['symbols'] if isinstance(data, dict) else data
    vocab_size = len(symbols)

    os.environ['KOKORO_SYMBOLS_PATH'] = symbols_path
    config.setdefault('model_params', {})['n_token'] = vocab_size
    return symbols_path, vocab_size
