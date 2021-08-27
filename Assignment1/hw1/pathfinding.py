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
from hw1.pqueue import PrioritizedItem, SimplePriorityQueue
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
    state: MazeState
    action: str
    depth: int
    parent: TreeNode = None

    position: tuple[int] = (np.nan,np.nan)
    g_score = float("inf")
    f_score = float("inf")
    h_score = float("inf")

    # def print_tree(self):
    #     if self.parent != None:
    #         TreeNode.print_tree(self.parent)
    #     return self.action


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



def reconstruct_paths(node: TreeNode):
    x = []
    total_cost = node.path_cost
    while node.parent != None:
        x.append(node.action)
        node = node.parent
    x.reverse()
    return x,total_cost


# TODO: 9 Implement the graph search algorithm.
def graph_search(
        init_state: MazeState,
        priority_func: Callable[[TreeNode], float]) -> Tuple[List[str], float]:
    """Perform graph search on the initial state and return a list of actions.

    If the solution cannot be found, return None and infinite cost.
    """
    ##for visualization##
    vis_grid = np.array(init_state.grid)
    ##for visualization##

    open_set = SimplePriorityQueue()
    open_list = []
    close_set = []
    g_score = {}
    f_score = {}
    count = 0
    size = np.shape(init_state.grid)
    for i in range(0, size[0]):
        for j in range(0, size[0]):
            g_score[(i,j)] = float('inf')
            f_score[(i,j)] = float('inf')
    (x,y) = find_agent(init_state.grid)
    g_score[(y,x)] = 0
    hScore = MazeState.heuristic(init_state)
    f_score[(y,x)] = hScore
    node = TreeNode(0,init_state,"North",0, None)
    node.g_score = 0.0
    node.f_score = hScore
    node.position = (y,x)

    open_set.add((y,x),node,hScore)
    
    while not open_set.is_empty():
        count += 1
        current = open_set.pop(); #node
        
        ##for visualization##
        # print(f'[{count}] agent:{current.position} f(n): {current.f_score} g(n):{current.g_score}')

        # vis_grid[current.position] = 9 if vis_grid[current.position] == 7 else 8
        # print(render_maze(vis_grid))
        ##for visualization##
        
        if MazeState.is_goal(current.state):
            actions, total_cost = reconstruct_paths(current)
            return actions, total_cost

        close_set.append(current.position)
        #Find neighbors
        actions = ['North', 'South', 'East', 'West']
        neighbors = []
        for action in actions:
            state = MazeState.transition(current.state, action)
            
            #wall
            if state is None:
                pass
            else:
                (x,y) = find_agent(state.grid)
                neighbor_loc = (y,x)
                #Explore the neighbor
                if neighbor_loc not in close_set:
                    tempG = current.g_score + MazeState.cost(current.state, action) #distance(current,neighbor)

                    if not open_set.is_in(neighbor_loc):
                        
                        path_cost = MazeState.cost(current.state, action)+current.path_cost
                        neighbor = TreeNode(path_cost,state,action,count,current)
                        neighbor.position = neighbor_loc
                        
                        if priority_func.__name__ == 'a_star_priority':
                            neighbor.g_score = g_score[neighbor_loc] = path_cost
                            neighbor.f_score = f_score[neighbor_loc] = a_star_priority(neighbor)
                            open_set.add(neighbor.position, neighbor, neighbor.f_score)
                        #greedy search
                        else:
                            hScore = MazeState.heuristic(neighbor.state)
                            open_set.add(neighbor.position, neighbor, hScore)
                            # print(f'neighbor:{neighbor.position} path_cost: {neighbor.path_cost} h: {neighbor.f_score}')

                    #If in openset and has better G(n), then update G(n)    
                    elif tempG <= g_score[neighbor_loc]:
                            neighbor.g_score = tempG
                            neighbor.f_score = a_star_priority(neighbor)
                            # print(f'update neighbor: {neighbor.position} f(n): {neighbor.f_score}')
                else:
                    pass 
                # for k,v in g_score.items():
                #     if v != float('inf'):
                #         print(f'{k}: {v}')

    return None, float('inf')















