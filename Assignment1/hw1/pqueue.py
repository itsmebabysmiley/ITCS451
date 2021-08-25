"""This module contains a SimplePriorityQueue class.

From https://docs.python.org/3/library/heapq.html
"""

from heapq import heappush, heappop
from dataclasses import dataclass, field
from typing import Any, Tuple, Union, List


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    key: Any=field(compare=False)
    item: Any=field(compare=False)
    removed: bool = False

    def remove(self):
        self.removed = True


class SimplePriorityQueue:

    def __init__(self) -> None:
        self.pq: List[PrioritizedItem] = []
        self.entry_finder = {}

    def is_empty(self) -> bool:
        if len(self.pq) > 0:
            for x in self.pq:
                if not x.removed:
                    return False
        return True

    def is_in(self, key: Any) -> bool:
        return key in self.entry_finder

    def get(self, key: Any) -> Union[Tuple[float, Any], None]:
        if key in self.entry_finder:
            x = self.entry_finder[key]
            return x.priority, x.item
        else:
            return None

    def add(self, key: Any, item: Any, priority: float):
        """Add a new item or update the priority of an existing item"""
        if key in self.entry_finder:
            self.remove(key)
        entry = PrioritizedItem(priority, key, item)
        self.entry_finder[key] = entry
        heappush(self.pq, entry)

    def remove(self, key):
        """Mark an existing item as REMOVED.  Raise KeyError if not found."""
        entry = self.entry_finder.pop(key)
        entry.remove()

    def pop(self) -> Any:
        """Remove and return the lowest priority item. Raise KeyError if empty."""
        while self.pq:
            x = heappop(self.pq)
            item = x.item
            key = x.key
            if not x.removed:
                del self.entry_finder[key]
                return item
        raise KeyError('pop from an empty priority queue')