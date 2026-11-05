"""Services package initializer.

Expose frequently-used service modules for convenient imports like
`from services import prompt_service` in tests and command-line tools.
"""

from . import prompt_service  # re-export for convenience

__all__ = ["prompt_service"]
