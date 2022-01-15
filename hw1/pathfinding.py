"""This module contains classes and function to solve a pathfinding problem.

Author: Nopparat Pengsuk
Student ID: 6288103
pseudocode : https://en.wikipedia.org/wiki/A*_search_algorithm
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
    actions: Tuple[str] = ('Move North', 'Move South', 'Move East', 'Move West')

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
        temp_grid = np.array(state.grid)
        #check action is possible
        if action == 'Move North' and state.grid[y-1 ,x] != 1:
            temp_grid[y,x] = 0
            temp_grid[y-1,x] = 2
        elif action == 'Move South' and state.grid[y+1, x] != 1:
            temp_grid[y,x] = 0
            temp_grid[y+1,x] = 4
        elif action == 'Move East' and state.grid[y, x+1] != 1:
            temp_grid[y,x] = 0
            temp_grid[y,x+1] = 3
        elif action == 'Move West' and state.grid[y, x-1] != 1:
            temp_grid[y,x] = 0
            temp_grid[y,x-1] = 5
        #wall
        else:
            return None

        temp_grid.flags.writeable = False
        new_state = MazeState(temp_grid)
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
        if action == 'Move North':
            cell = state.grid[y-1 ,x]
        elif action == 'Move South':
            cell = state.grid[y+1, x]
        elif action == 'Move East':
            cell = state.grid[y, x+1]
        elif action == 'Move West':
            cell = state.grid[y, x-1]

        #wall
        if cell == 1:
            return float('inf')
        #cost from current to next state.
        #mud cost 10. normal cost 1
        else:
            return 10 if cell == 7 else 1


    # TODO 5: Create a goal test function
    @classmethod
    def is_goal(cls, state: MazeState) -> bool:
        """Return True if `state` is the goal."""
        x,y = find_agent(state.grid)
        agent = (y,x) 
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
        (y1,x1) = find_agent(state.grid)
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
    #keep track location of the node
    position: Tuple[int] = (np.nan,np.nan)

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
    return node.path_cost + MazeState.heuristic(node.state)


# Show every step the agent made. 
def visualize_step(node: TreeNode, grid: np.array, g_score: Tuple):

    print(f'agent:{node.position} g(n):{g_score} depth={node.depth}')
    grid[node.position] = 9 if grid[node.position] == 7 else 8
    print(render_maze(grid))
    print('~~'*np.shape(grid)[0])


def reconstruct_path(node: TreeNode):
    x = []
    total_cost = node.path_cost
    while node.parent != None:
        x.append(node.action)
        node = node.parent
    x.reverse()
    return x,float(total_cost)


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
    came_from = []
    close_set = []
    size = np.shape(init_state.grid)
    g_score = {(i,j): float('inf') for i in range(size[0]) for j in range(size[0])}
    h_score = {(i,j): float('inf') for i in range(size[0]) for j in range(size[0])}
    f_score = {(i,j): float('inf') for i in range(size[0]) for j in range(size[0])}
    count = 0   #count is count no matter what. hehe

    (x,y) = find_agent(init_state.grid)
    g_score[(y,x)] = 0
    hScore = MazeState.heuristic(init_state)
    f_score[(y,x)] = hScore #start node use h score
    init_node = TreeNode(0,init_state,"None",0, None)
    init_node.position = (y,x)

    open_set.add((y,x),init_node,hScore)
    
    while not open_set.is_empty():
        count += 1
        current = open_set.pop(); #node O(1)
        

        '''for visualization'''
        # print(f'[{count}]',end=' ')
        # visualize_step(grid = vis_grid, node = current, g_score = g_score[current.position])
        '''for visualization'''


        #is goal?
        if MazeState.is_goal(current.state):
            actions, total_cost = reconstruct_path(current)
            return actions, total_cost
        
        #already explore :3
        close_set.append(current.position)

        #Explore the neighbors
        actions = ['Move North', 'Move South', 'Move East', 'Move West']
        for action in actions:
            state = MazeState.transition(current.state, action)
            
            #wall
            if state is None:
                pass
            else:
                (x,y) = find_agent(state.grid)
                neighbor_loc = (y,x)
                
                if priority_func.__name__ == 'a_star_priority':

                    if neighbor_loc not in close_set:

                        tempG = g_score[current.position] + MazeState.cost(current.state, action) #g(cur)+distance(current,neighbor)
                        
                        #better path update the g score and f score
                        if tempG < g_score[neighbor_loc]:
                            came_from.append(current)

                            neighbor_node = TreeNode(float('inf'),state,action,current.depth+1,current)
                            neighbor_node.position = neighbor_loc
                            neighbor_node.path_cost = g_score[neighbor_loc] = tempG
                            f_score[neighbor_loc] = a_star_priority(neighbor_node)
                            
                            if not open_set.is_in(neighbor_loc):                            
                                open_set.add(neighbor_node.position, neighbor_node, f_score[neighbor_loc])
                    else:
                        #smile to the camara!
                        pass
                #greedy search
                else:
                    #find h score and use as a priority.
                    path_cost = MazeState.cost(current.state, action) + current.path_cost
                    neighbor_node = TreeNode(path_cost, state, action,current.depth+1, current, position = neighbor_loc)
                    g_score[neighbor_loc] = path_cost
                    h_score[neighbor_loc] = MazeState.heuristic(neighbor_node.state)
                    open_set.add(neighbor_loc,neighbor_node,h_score[neighbor_loc])        

    #no solution is it possible with this maze? -.-
    return None, float('inf')















