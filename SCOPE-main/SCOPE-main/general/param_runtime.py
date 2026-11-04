import threading
from functools import lru_cache
from .parameters import load_params  # function exists
import pdb

_cfg_path = None
_version  = 0
_lock = threading.Lock()

def set_cfg_path(path: str):
    global _cfg_path
    _cfg_path = path

def bump_version():
    """Saved parameters have changed, bump version to invalidate cache"""
    global _version,P
    with _lock:
        _version += 1
        params.cache_clear()  # clear cache to reload params on next access

@lru_cache(maxsize=1)
def params(version: int = None):
    """get parameters object, reloading if version changed"""
    return load_params(_cfg_path)  # only called when cache miss

class _ParamsProxy:
    def __getattr__(self, name):
        return getattr(params(), name)   

P = _ParamsProxy()  