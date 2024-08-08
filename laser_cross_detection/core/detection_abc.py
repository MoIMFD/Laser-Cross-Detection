from abc import ABC, abstractmethod
from typing import Any


class DetectionMethodABC(ABC):

    @abstractmethod
    def __call__(self, image, *args: Any, **kwds: Any) -> Any:
        pass
