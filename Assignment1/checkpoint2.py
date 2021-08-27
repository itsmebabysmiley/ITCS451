from typing import List
import numpy as np

from hw1.envutil import gen_maze, render_maze, find_agent
from hw1.pathfinding import (
    MazeState,
    graph_search,
    bfs_priority, dfs_priority, greedy_priority, a_star_priority)


# CHANGE SETTINGS TO TEST YOUR CODE HERE
SIZE = 10
ADD_MUD = True
PRIORITY1 = greedy_priority
PRIORITY2 = a_star_priority


def trace_plan(init_state: MazeState, plan: List[str]):
    grid = np.array(init_state.grid)
    x, y = find_agent(grid)
    cur_state = init_state
    for action in plan:
        new_state = MazeState.transition(cur_state, action)
        if new_state is None:
            print(find_agent(cur_state.grid),action)
            raise ValueError('Impossible action in the plan!')
        new_x, new_y = find_agent(new_state.grid)
        if new_x == x and new_y == y:
            # rotate?
            pass
        else:
            t = 8
            if grid[y, x] == 7:
                t = 9
            grid[y, x] = t
            x, y = new_x, new_y
        cur_state = new_state
    print(render_maze(grid))

my_grid = np.array(
    [
        [1,1,1,1,1,1,1,1,1,1],
        [1,2,7,0,1,1,1,1,1,1],
        [1,7,1,0,0,0,1,1,1,1],
        [1,0,7,1,1,0,0,0,0,1],
        [1,0,0,1,1,1,1,0,0,1],
        [1,0,0,0,1,1,1,7,0,1],
        [1,1,7,0,0,7,0,0,0,1],
        [1,1,7,1,1,1,1,7,0,1],
        [1,1,0,0,7,0,0,0,6,1],
        [1,1,1,1,1,1,1,1,1,1],
    ])
no_wall = np.array(
    [
        [1,1,1,1,1,1,1,1,1,1],
        [1,2,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,6,1],
        [1,1,1,1,1,1,1,1,1,1],
    ])
init_maze = gen_maze(SIZE, add_mud=ADD_MUD)
init_state = MazeState(init_maze)
print('Initial State:')
print(render_maze(init_state.grid))
print('=' * 50)
print('Algorithm:', PRIORITY1.__name__)
plan1, cost1 = graph_search(init_state, PRIORITY1)
print('Plan:', plan1)
print('Cost:', cost1)
if plan1 is not None:
    print('Steps:', len(plan1))
    trace_plan(init_state, plan1)
print('=' * 50)

print('Initial State:')
print(render_maze(init_state.grid))
print('=' * 50)
print('Algorithm:', PRIORITY2.__name__)
plan1, cost1 = graph_search(init_state, PRIORITY2)
print('Plan:', plan1)
print('Cost:', cost1)
if plan1 is not None:
    print('Steps:', len(plan1))
    trace_plan(init_state, plan1)
print('=' * 50)

