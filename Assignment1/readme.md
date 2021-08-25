# Assignment 1

## Background

We are going to create a pathfinding agent -- an agent that finds a path with the least cost. The environment is a grid world like this (10x10):

```text
# # # # # # # # # #
# >   ~       ~   #
#   #       # # ~ #
#   # # #       ~ #
#   # # # # # ~   #
#   ~ ~ ~ # # # ~ #
#     #   # # #   #
#   ~ #   # # # ~ #
#       ~       G #
# # # # # # # # # #

```

In this example, the agent ('>') is at the top-left corner and facing east. You can find the meaning of all symbols here:

```text
    Symbol mapping:
    -  0: ' ', empty (passable)
    -  1: '#', wall (not passable)
    -  2: '^', agent is facing up (north)
    -  3: '>', agent is facing right (east)
    -  4: 'v', agent is facing down (south)
    -  5: '^', agent is facing left (west)
    -  6: 'G', goal
    -  7: '~', mud (passable, but cost more)
```

For each step, the agent can change its facing direction or move forward (no action is not an option here). The agent has to pay for each action. You can design the cost of the actions. Whenever the agent gets into the mud, it must pay an additional cost and the mud will disappear.

### Choices

1. Answer Question 1 and Question 2
2. Implement TODO 1 - 6, and answer Question 2
3. Implement TODO 1 - 9

### Question 1

Please formulate this problem into a search problem.

Criteria (3 points):

1. 0.5pt: You correctly identify information to maintain in the state.
2. 0.5pt: You correctly identify a list of all possible actions
3. 1.0pt: You correctly describe the transition function and you show examples of how the transition function works for all actions. (show input state, action, and the output state).
4. 0.5pt: You correctly describe the cost function and show some examples.
5. 0.5pt: You correctly describe a heuristic function and show at least 3 examples (state and its heuristic value)

### Question 2

Please perform an A* search on a 6x6 maze with random muds. You can get a randomly generated maze from checkpoint1.py or checkpoint2.py. If you are allergic to code, you can ask one of your friends to generate mazes for you.

Criteria 4 points:

1. 0.5pt: Most of the f-values that you compute are correct.
2. 0.5pt: You select a node to explore next correctly for all steps.
3. 1pt: You show a correct search tree.
4. 1pt: You show a correct open set (frontier) and a correct closed set (explored)
5. 1pt: You correctly summarize the plan and its cost.

### Checkpoint 1

Implement TODO 1 to 6

Criteria (4 points)

1. 1pt: Your code run (checkpoint1.py) without any error.
2. 0.5pt: For each correct implementation of the TODO 1 to 6

### Checkpoint 2

Implement TODO 7 to 9

Criteria (6 points)

1. 1pt: Your code run (checkpoint2.py) without any error.
2. 0.5pt: TODO 7
3. 0.5pt: TODO 8
4. 1pt: TODO 9, termination condition is correct
5. 1pt: TODO 9, new nodes are created correctly
6. 1pt: TODO 9, nodes are correctly pushed into the queue (open-set)
7. 1pt: TODO 9, a correct plan and a correct cost is returned
