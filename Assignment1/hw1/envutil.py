"""This module contains helper functions to generate enviornment."""

# %%

from typing import List, Tuple
import numpy as np


def gen_maze(size: int, add_mud: bool = False) -> np.ndarray:
    """Generate a maze that has random paths to the goal.

    The maze is represented by an array of size with the following properties:
    - A wall is surrounding the maze.
    - Start position is at x = 1 and y = 1 (top-left index).
    - Goal position is at x = size-2 and y = size-2 (bottom-right index)
    - The starting orientation of the agent is random

    Symbol mapping:
    -  0: ' ', empty (passable)
    -  1: '#', wall (not passable)
    -  2: '^', agent is facing up (north)
    -  3: '>', agent is facing right (east)
    -  4: 'v', agent is facing down (south)
    -  5: '^', agent is facing left (west)
    -  6: 'G', goal
    -  7: '~', mud (passable, but cost more)
    """
    assert size > 5, '`size` must be greater than 5.'
    grid = np.ones(shape=(size, size), dtype=np.int32)
    start = (1, 1)
    goal = (size - 2, size - 2)
    success_count = 0
    while (grid==0).sum() < 0.6 * (size - 2)**2 or success_count < 1:
        gaps = set()
        gaps = gaps.union([(0, i) for i in range(size)])
        gaps = gaps.union([(size-1, i) for i in range(size)])
        gaps = gaps.union([(i, 0) for i in range(size)])
        gaps = gaps.union([(i, size-1) for i in range(size)])
        p = np.random.random()
        p = [p , 1-p]
        cur = start
        paths = []
        while cur != goal:
            gaps.add(cur)
            paths.append(cur)
            x, y = cur
            # if np.random.random() < 0.01:
            #     neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            # else:
            neighbors = [(x+1, y), (x, y+1)]
            neighbors = [n for n in neighbors if n not in gaps]
            if len(neighbors) == 0 or len(paths) > size * 4:
                break  # fail
            elif len(neighbors) == 1:
                step = [0]
            else:
                step = np.random.choice(np.arange(len(neighbors)), 1, p=p)
            cur = neighbors[step[0]]
        else:
            paths.append(cur)
            for c in paths:
                grid[c] = 0
            success_count += 1

    if add_mud:
        passables = list(zip(*np.where(grid == 0)))
        muds = np.random.choice(
            np.arange(len(passables)),
            size=len(passables) // 3,
            replace=False)
        for m in muds:
            grid[passables[m]] = 7

    grid[start] = np.random.choice([2, 3, 4, 5])
    grid[goal] = 6
    grid.flags.writeable = False
    return grid.T


def render_maze(grid: np.ndarray, maps: List[str] = None) -> str:
    if maps is None:
        maps = [' ', '#', '^', '>', 'v', '<', 'G', '~', 'x', 'X', 'o', 'O']
    s = [' '.join([maps[x] for x in row]) for row in grid]
    return '\n'.join(s)


def find_agent(grid: np.ndarray) -> Tuple[int, int]:
    """Return x, y location of the agent. If not exist, return None, None."""
    y, x = np.where((grid > 1) & (grid < 6))
    if len(x) > 0 and len(y) > 0:
        return x[0], y[0]
    else:
        return None, None
# %%
