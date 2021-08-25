import numpy as np

from hw1.envutil import gen_maze, render_maze
from hw1.pathfinding import MazeState, TreeNode
from hw1.pqueue import SimplePriorityQueue


maze1 = gen_maze(10, add_mud=True)
state1 = MazeState(maze1)
print(render_maze(maze1))

# try to perform action:
for action in state1.actions:
    print('-' * 50)
    print('ACTION: ', action)
    state2 = MazeState.transition(state1, action)
    cost = MazeState.cost(state1, action)
    if state2 is None:
        print('Impossible!')
    else:
        print(render_maze(state2.grid))
    print('Cost: ', cost)

print('=' * 50)
# try random actions for a while:
cur_state = state1
print(render_maze(cur_state.grid))
for __ in range(10):
    action = np.random.choice(cur_state.actions)
    print('-' * 50)
    print('ACTION: ', action)
    next_state = MazeState.transition(cur_state, action)
    cost = MazeState.cost(cur_state, action)
    if next_state is None:
        print('Impossible!')
    else:
        print(render_maze(next_state.grid))
        cur_state = next_state
    print('Cost: ', cost)


# check goal
print('=' * 50)
y = np.array(maze1)
y[1, 1] = 0
y[-2, -2] = 3
goal_state = MazeState(y)

print('Start state is goal?', MazeState.is_goal(state1))

print(render_maze(goal_state.grid))
print('Goal state is goal?', MazeState.is_goal(goal_state))

# check heuristics
print('=' * 50)
print('Heristics:', MazeState.heuristic(state1))
print('Heristics:', MazeState.heuristic(goal_state))