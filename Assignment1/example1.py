import numpy as np

from hw1.envutil import gen_maze, render_maze, find_agent
from hw1.pathfinding import MazeState, TreeNode
from hw1.pqueue import SimplePriorityQueue

# Creating a maze state
maze1 = gen_maze(40)
state1 = MazeState(maze1)
print(render_maze(maze1))
print('Agent location: ', find_agent(state1.grid))

# Cloning a maze state
maze2 = gen_maze(40)
maze3 = np.array(maze2)
state2 = MazeState(maze2)
state3 = MazeState(maze3)

print('state1 == state2: ', state1 == state2)
print('state1 == state3: ', state1 == state3)
print('state2 == state3: ', state2 == state3)

# Creating a tree node
node1 = TreeNode(
    path_cost=0,
    state=state1,
    action=None,
    depth=0,
    parent=None)
# More example
node2 = TreeNode(42, state2, 'XXX', 1, parent=state1)
node3 = TreeNode(0, state3, None, 0, parent=None)

print('node1 == node2: ', node1 == node2)
print('node1 == node3: ', node1 == node3)
print('node2 == node3: ', node2 == node3)

# Using the priority queue
pq = SimplePriorityQueue()
print('empty:', pq.is_empty())  # check empty

pq.add(
    key=node1.state,
    item=node1,
    priority=10)   # adding item

# Check existence
print('node1 in pq: ', pq.is_in(node1.state))
print('node2 in pq: ', pq.is_in(node2.state))

# More example
pq.add(node2.state, node2, 2)
print('node2 in pq: ', pq.is_in(node2.state))
print('node3 in pq: ', pq.is_in(node3.state))
print('empty: ', pq.is_empty())

# Get lowest priority item
pop_node = pq.pop()
print('pop node2', pop_node == node2)

# Adding the same `key` will replace an exisitng one
pq.add(node2.state, node2, 2)
pq.add(node3.state, node3, 1)  # node2's state == node3's state
pop_node = pq.pop()
print('pop node3', pop_node == node3)
print('empty: ', pq.is_empty())
pop_node = pq.pop()
print('pop node1', pop_node == node1)
print('empty: ', pq.is_empty())
