"""Name -> factory lookup, so variants are added without editing the core.

Both group members add algorithm variants concurrently. If adding one meant editing
``train.py``, every variant would be a merge conflict in the same file. Variants
register themselves here instead, and configs select them by name.
"""

from collections.abc import Callable


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            if name in self._items:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._items[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {self.names()}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


ALGOS = Registry("algo")
