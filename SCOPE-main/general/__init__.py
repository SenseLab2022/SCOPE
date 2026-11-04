
# general/__init__.py
from importlib import import_module

parameters = import_module('.parameters', __name__)
__all__ = ['parameters']