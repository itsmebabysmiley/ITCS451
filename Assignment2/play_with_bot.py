# %%

import time

from tabulate import tabulate
import numpy as np
from hw2 import tictactoe as TTT

# Change this to your bots.
botX = TTT.StupidBot(TTT.Player.X)
botO = TTT.StupidBot(TTT.Player.O)
# botO = TTT.HumanPlayer(TTT.Player.O)


init_state = TTT.TicTacToeState(np.zeros((3, 3), dtype=np.uint8), TTT.Player.X)
# %%
cur_state = init_state
while not TTT.TicTacToeState.isTerminal(cur_state):
    print(cur_state)

    if cur_state.curPlayer == TTT.Player.X:
        action = botX.play(cur_state)
    else:
        action = botO.play(cur_state)

    cur_state = TTT.TicTacToeState.transition(cur_state, action)

# %%
print(cur_state)
scoreX = TTT.TicTacToeState.utility(cur_state, TTT.Player.X)
scoreO = TTT.TicTacToeState.utility(cur_state, TTT.Player.O)
print(f'X score: {scoreX}')
print(f'O score: {scoreO}')

# %%
