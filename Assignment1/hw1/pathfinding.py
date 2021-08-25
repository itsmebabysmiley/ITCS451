"""This module contains classes and function to solve a pathfinding problem.

Author: Nopparat Pengsuk
    -
    -
Student ID: 6288103
    -
    -
"""

# %%

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Callable, Union
from hw1.envutil import render_maze, find_agent

import numpy as np

@dataclass(frozen=True, unsafe_hash=False)
class MazeState:

    # TODO 1: Add other state information here.
    grid: np.ndarray
    # If you need anything more than `grid`, please add here

    # TODO 2 Create a list of all possible actions.
    # Please replace it with your own actions
    # Note that an agent can only rotate and move forward.
    actions: Tuple[str] = ('North', 'South', 'East', 'West')

    def __eq__(self, o: object) -> bool:
        if isinstance(o, MazeState):
            return np.all(self.grid == o.grid)
        return False

    def __hash__(self) -> int:
        return render_maze(self.grid).__hash__()

    # TODO 3: Create a transition function
    @classmethod
    def transition(cls, state: MazeState, action: str) -> MazeState:
        """Return a new state after performing `action`.

        If the action is not possible, it should return None.

        The mud disappears as soon as the agent walk onto it.

        Note
        ---------------------
        Keep in mind that you should not modify the previous state
        If you need to clone a numpy's array, you can do so like this:
        >>> y = np.array(x)
        >>> y.flags.writeable = False
        This will create an array y as a copy of array x and then make
        array y immutable (cannot be changed, for safty).
        """
        x,y = find_agent(state.grid)
        # print(y,x)
        # current = np.where(np.isin(state.grid, [2,3,4,5]))
        # y, x = current[0], current[1]
        # print(y,x)
        temp_grid = np.array(state.grid)
        #check action is possible
        if action == 'North' and state.grid[y-1 ,x] != 1:
            temp_grid[y,x] = 0
            temp_grid[y-1,x] = 2
        elif action == 'South' and state.grid[y+1, x] != 1:
            temp_grid[y,x] = 0
            temp_grid[y+1,x] = 4
        elif action == 'East' and state.grid[y, x+1] != 1:
            temp_grid[y,x] = 0
            temp_grid[y,x+1] = 3
        elif action == 'West' and state.grid[y, x-1] != 1:
            temp_grid[y,x] = 0
            temp_grid[y,x-1] = 5
        #wall
        else:
            return None

        temp_grid.flags.writeable = False
        new_state = MazeState(temp_grid)
        # print(new_state.grid)
        return new_state

    # TODO 4: Create a cost function
    @classmethod
    def cost(cls, state: MazeState, action: str) -> float:
        """Return the cost of `action` for a given `state`.

        If the action is not possible, the cost should be infinite.

        Note
        ------------------
        You may come up with your own cost for each action, but keep in mind
        that the cost must be positive and any walking into
        a mod position should cost more than walking into an empty position.
        """
        x,y = find_agent(state.grid)
        # current = np.where(np.isin(state.grid, [2,3,4,5]))
        # y, x = current[0], current[1]
        if action == 'North':
            cell = state.grid[y-1 ,x]
        elif action == 'South':
            cell = state.grid[y+1, x]
        elif action == 'East':
            cell = state.grid[y, x+1]
        elif action == 'West':
            cell = state.grid[y, x-1]

        if cell == 1:
            return float('inf')
        else:
            return 10 if cell == 7 else 1

    # TODO 5: Create a goal test function
    @classmethod
    def is_goal(cls, state: MazeState) -> bool:
        """Return True if `state` is the goal."""
        x,y = find_agent(state.grid)
        if x == 8 and y == 8:
            return True
        else:
            return False

    # TODO 6: Create a heuristic function
    @classmethod
    def heuristic(cls, state: MazeState) -> float:
        """Return a heuristic value for the state.

        Note
        ---------------
        You may come up with your own heuristic function.
        """
        #Manhattan distance
        x,y = find_agent(state.grid)
        goal = (8,8)
        h = abs(x-8)+abs(y-8)
        return h
# %%

@dataclass
class TreeNode:
    path_cost: float
    state: MazeState
    action: str
    depth: int
    parent: TreeNode = None


def dfs_priority(node: TreeNode) -> float:
    return -1.0 * node.depth


def bfs_priority(node: TreeNode) -> float:
    return 1.0 * node.depth


# TODO: 7 Create a priority function for the greedy search
def greedy_priority(node: TreeNode) -> float:
    return 0.0


# TODO: 8 Create a priority function for the A* search
def a_star_priority(node: TreeNode) -> float:
    return 0.0


# TODO: 9 Implement the graph search algorithm.
def graph_search(
        init_state: MazeState,
        priority_func: Callable[[TreeNode], float]) -> Tuple[List[str], float]:
    """Perform graph search on the initial state and return a list of actions.

    If the solution cannot be found, return None and infinite cost.
    """
    return None, float('inf')
