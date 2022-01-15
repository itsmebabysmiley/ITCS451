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

Credit to: Dr.Thanapon Noraset