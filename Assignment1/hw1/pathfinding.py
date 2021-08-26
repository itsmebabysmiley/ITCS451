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
from queue import PriorityQueue
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
        agent = find_agent(state.grid)
        goal = tuple([ i-2 for i in list(state.grid.shape)])

        return agent == goal

    # TODO 6: Create a heuristic function
    @classmethod
    def heuristic(cls, state: MazeState) -> float:
        """Return a heuristic value for the state.

        Note
        ---------------
        You may come up with your own heuristic function.
        """

        #Manhattan distance
        (x1,y1) = find_agent(state.grid)
        (x2,y2) = tuple([ i-2 for i in list(state.grid.shape)])
        h = abs(x1-x2)+abs(y1-y2)
        return h
# %%

@dataclass
class TreeNode:
    path_cost: float 
    g_score = float("inf")
    f_score = float("inf")
    state: MazeState
    action: str
    depth: int
    parent: TreeNode = None
    position: tuple[int] = (-1,-1)

    # def __eq__(self, other):
    #     return self.depth == other.depth
    # def __lt__(self, other):
    #     return self.path_cost < other.path_cost

def dfs_priority(node: TreeNode) -> float:
    return -1.0 * node.depth


def bfs_priority(node: TreeNode) -> float:
    return 1.0 * node.depth


# TODO: 7 Create a priority function for the greedy search
def greedy_priority(node: TreeNode) -> float:
    return MazeState.heuristic(node.state)


# TODO: 8 Create a priority function for the A* search
def a_star_priority(node: TreeNode) -> float:
    h = MazeState.heuristic(node.state)
    return node.path_cost + h


# TODO: 9 Implement the graph search algorithm.
def graph_search(
        init_state: MazeState,
        priority_func: Callable[[TreeNode], float]) -> Tuple[List[str], float]:
    """Perform graph search on the initial state and return a list of actions.

    If the solution cannot be found, return None and infinite cost.
    """

    open_set = []
    close_set = []
    depth = 0
    node = TreeNode(0.0,init_state,init_state.actions[0],depth,None)
    open_set.append(node)
    grid = init_state.grid
    came_from = [] #close set
    track_agent = []
    total_cost = 0.0
    grid = init_state.grid
    g_score = {}
    f_score = {}
    for i in range(0,len(grid[0])):
        for j in range(0,len(grid[0])):
            g_score[(i,j)] = float("inf")
            f_score[(i,j)] = float("inf")
    (x,y) = find_agent(init_state.grid)
    g_score[(y,x)] = 0
    f_score[(y,x)] = MazeState.heuristic(init_state)
    while len(open_set) > 0:     

        depth += 1 #for wat?
        open_set.sort(key=lambda TreeNode: TreeNode.f_score, reverse=False)
        current = open_set[0]
        (x,y)= find_agent(current.state.grid)
        cur_loc = (y,x)
        print(f'agent: {current.position} g(n):{current.path_cost},{g_score[cur_loc]} f(n): {current.f_score}')
        close_set.append(cur_loc)
        came_from.append(current.action)
        # print(f'cur_state: {cur_loc} f(n): {f_score[cur_loc]} (g(n): {g_score[cur_loc]})')
        #current is goal return list of actions.
        if MazeState.is_goal(current.state):
            return came_from[1:], total_cost
        open_set.pop(0)
        #find neighbors
        neighbors = []
        for action in current.state.actions:
            next_state = MazeState.transition(current.state,action)
            cost = MazeState.cost(current.state,action)+current.path_cost
            #wall
            if next_state is None:
                pass
            else:
                neighbor = TreeNode(cost, next_state, action, depth, current)
                (x,y)= find_agent(neighbor.state.grid)
                nei_loc = (y,x)
                # print(f'agent: {nei_loc} action: {neighbor.action}')
                if nei_loc in close_set:
                    continue
                g_score[nei_loc] = cost
                neighbor.position = nei_loc
                neighbor.f_score = a_star_priority(neighbor)
                neighbors.append(neighbor)
        neighbors.sort(key=lambda TreeNode: TreeNode.f_score)
        if len(neighbors) > 0:
            open_set.append(neighbors[0])
        # for neighbor in neighbors: 
        # #     # (x,y) = find_agent(neighbor.state.grid)
        # #     tentative_gScore = g_score[cur_loc] + neighbor.path_cost
        # #     if tentative_gScore < g_score[nei_loc]:
        # #         print('he')
        # #         came_from.append(current.action)
        # #         g_score[nei_loc] = tentative_gScore
        # #         f_score[nei_loc] = a_star_priority(neighbor)
        # #         if neighbor not in open_set:
        # #             open_set.append(neighbor)

        # print(np.array(g_score))
        # print(np.array(f_score))
    print(came_from)
    return None, float('inf')












