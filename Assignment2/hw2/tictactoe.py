"""This module contains classes and functions for Tic-Tac-Toe.

Members:
1. Name:  ID:
2. Name:  ID:
3. Name:  ID:
4. Name:  ID:

"""
from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

from tabulate import tabulate
import numpy as np

symbol_map = ['_', 'X', 'O']
win = [
        [0,1,2],
        [3,4,5],
        [6,7,8],
        [0,3,6],
        [1,4,7],
        [2,5,8],
        [0,4,8],
        [2,4,6]
      ]
borad_grid = {  
                0 : (0,0), 1 : (0,1), 2 : (0,2),
                3 : (1,0), 4 : (1,1), 5 : (1,2),
                6 : (2,0), 7 : (2,1), 8 : (2,2)}


class Player(Enum):
    X = 1
    O = 2

@dataclass(frozen=True)
class TicTacToeState:
    # TODO 1: Add state information that you need.
    board: np.ndarray
    # The board position is numbered as follow:
    # 0 | 1 | 2
    # ----------
    # 3 | 4 | 5
    # ----------
    # 6 | 7 | 8
    #
    # If you need anything more than `board`, please provide them here.

    curPlayer: Player  # keep track of the current player

    # TODO 2: Create actions function
    @classmethod
    def actions(cls, state: TicTacToeState) -> List[int]:
        """Return a list of valid position (from 0 to 8) for the current player.

        In Tic-Tac-Toe, a player can always make a move as long as there is
        an empty spot. If the board is full, however, return an empty list.
        """
        board = np.array(state.board).flatten()
        return [i for i in range(len(board)) if board[i] == 0]

    # TODO 3: Create a transtion function
    @classmethod
    def transition(cls, state: TicTacToeState, action: Union[int, None]) -> TicTacToeState:
        """Return a new state after a player plays `action`.

        If `action` is None, skip the player turn.

        The current player is in the `state.curPlayer`. If the action is not
        valid, skip the player turn.

        Note
        --------------------------
        Keep in mind that you should not modify the previous state
        If you need to clone a numpy's array, you can do so like this:
        >>> y = np.array(x)
        >>> # make change
        >>> y.flags.writeable = False
        This will create an array y as a copy of array x and then make
        array y immutable (cannot be changed, for safty).
        """
        board = np.array(state.board).flatten()
        #action is impossible?
        if board[action] != 0:
            return None
        else:
            board[action] = 1 if state.curPlayer == Player.X else 2
            board_2d = np.reshape(board, (3, 3))
            next_player = Player.O if state.curPlayer == Player.X else Player.X
            board_2d.flags.writeable = False
            new_board = TicTacToeState(board_2d, next_player) 
            return new_board


    # TODO 4: Create a terminal test function
    @classmethod
    def isTerminal(cls, state: TicTacToeState) -> bool:
        """Return `True` is the `state` is terminal (end of the game)."""
        #3 possible ways that borad will terminate; playerX won or playerO won or Tie
        positions = np.where(state.board.flatten() == 1)
        for w in win:
            if all(x in positions[0] for x in w):
                return True #plauer X won
        positions = np.where(state.board.flatten() == 2)
        for w in win:
            if all(x in positions[0] for x in w):
                return True #player O won
        full = np.where(state.board.flatten() == 0)
        if full[0].size == 0:
            return True #borad full. Tie?
        
        #not terminate
        return False

    # TODO 5: Create a utility function
    @classmethod
    def utility(cls, state: TicTacToeState, player: Player) -> Union[float, None]:
        """Return the utility of `player` for the `state`.

        If the state is non-terminal, return None.

        The `player` can be different than the `state`.`curPlayer`.
        """
        #You can use any value but I prefer -1,0,1 which are lose, tie and won respectively.
        positions = np.where(state.board.flatten() == 1)
        for w in win:
            if all(x in positions[0] for x in w):
                if player == Player.X:
                    return 1 #player X won
                else:
                    return -1
        positions = np.where(state.board.flatten() == 2)
        for w in win:
            if all(x in positions[0] for x in w):
                if player == Player.O:
                    return 1 #player O won
                else:
                    return -1
        
        return 0 #tie

    def __repr__(self) -> str:
        a = [[symbol_map[c] for c in row] for row in self.board]
        if TicTacToeState.isTerminal(self):
            return tabulate(a)
        else:
            return tabulate(a) + '\n' + 'Turn: ' + self.curPlayer.name

class StupidBot:

    def __init__(self, player: Player) -> None:
        self.player = player

    def play(self, state: TicTacToeState) -> Union[int, None]:
        """Return an action to play or None to skip."""
        # pretend to be thinking
        # time.sleep(1)

        # return random action
        valid_actions = TicTacToeState.actions(state)
        if len(valid_actions) == 0:
            return None
        else:
            return valid_actions[np.random.randint(0, len(valid_actions))]


class HumanPlayer(StupidBot):

    def __init__(self, player: Player) -> None:
        super().__init__(player)

    def play(self, state: TicTacToeState) -> Union[int, None]:
        valid_actions = TicTacToeState.actions(state)
        if len(valid_actions) == 0:
            return None
        else:
            action = int(input(f'Valid: {valid_actions} Your move: '))
            while action not in valid_actions:
                print('Your move is invalid. Try again:')
                action = int(input(f'Valid: {valid_actions} Your move: '))
            return action




class MinimaxBot(StupidBot):

    def __init__(self, player: Player) -> None:
        super().__init__(player)

    # TODO 6: Implement Minimax Decision algorithm
    def play(self, state: TicTacToeState) -> Union[int, None]:
        """Return an action to play or None to skip."""
        
        valid_actions = TicTacToeState.actions(state)
        if len(valid_actions) == 0:
            return None
        else:
            #board empty. randomly choose! don't have to thinking!
            if len(valid_actions) == 9:
                return np.random.randint(0, 9)
            
            move = None
            maxEva = np.NINF
            board = np.array(state.board).flatten()
            for action in valid_actions:
                #whos turn?
                board[action] = 1 if self.player == Player.X else 2 

                temp_state = TicTacToeState(np.reshape(board, (3, 3)),self.player)
                value = self.minimax(temp_state, False)   
                temp_state.board[borad_grid[action]] = 0 #reset board
                temp_state.board.flags.writeable = False
                if value > maxEva:
                    maxEva = value
                    move = action
            
            return move
    
    
    def minimax(self, state, maximizingPlayer):
        
        #terminal case
        if TicTacToeState.isTerminal(state):
            return TicTacToeState.utility(state, state.curPlayer)
        
        if maximizingPlayer:
            value = np.NINF
            board = np.array(state.board).flatten()
            valid_actions = TicTacToeState.actions(state)
            for action in valid_actions:
                #whos turn
                board[action] = 1 if state.curPlayer == Player.X else 2
                
                temp_state = TicTacToeState(np.reshape(board, (3, 3)), state.curPlayer)
                value = max(value, self.minimax(temp_state, False))
                temp_state.board[borad_grid[action]] = 0 #reset board
                temp_state.board.flags.writeable = False
            
            return value
        #minimizing
        else:
            value = np.Inf
            board = np.array(state.board).flatten()
            valid_actions = TicTacToeState.actions(state)
            for action in valid_actions:
                #whos turn
                board[action] = 2 if state.curPlayer == Player.X else 1

                temp_state = TicTacToeState(np.reshape(board, (3, 3)), state.curPlayer)
                value = min(value, self.minimax(temp_state, True))
                temp_state.board[borad_grid[action]] = 0 #reset board
                temp_state.board.flags.writeable = False
            return value
class AlphaBetaBot(StupidBot):

    def __init__(self, player: Player) -> None:
        super().__init__(player)

    # TODO 7: Implement Alpha-Beta Decision algorithm
    def play(self, state: TicTacToeState) -> Union[int, None]:
        """Return an action to play or None to skip."""
        
        valid_actions = TicTacToeState.actions(state)
        if len(valid_actions) == 0:
            return None
        else:
            #board empty. randomly choose! don't have to thinking!
            if len(valid_actions) == 9:
                return np.random.randint(0, 9)
            
            move = None
            maxEva = np.NINF
            board = np.array(state.board).flatten()
            for action in valid_actions:
                #whos turn?
                board[action] = 1 if self.player == Player.X else 2 

                temp_state = TicTacToeState(np.reshape(board, (3, 3)),self.player)
                alpha = np.NINF
                beta = np.Inf
                value = self.min_value(temp_state, alpha, beta) 
                temp_state.board[borad_grid[action]] = 0 #reset board
                temp_state.board.flags.writeable = False
                # print(value)
                if value > maxEva:
                    maxEva = value
                    move = action
            
            return move
    
    
    def max_value(self, state, alpha, beta):
        #terminal case
        if TicTacToeState.isTerminal(state):
            return TicTacToeState.utility(state, state.curPlayer)
        
        value = np.NINF
        board = np.array(state.board).flatten()
        valid_actions = TicTacToeState.actions(state)
        for action in valid_actions:
            #whos turn
            board[action] = 1 if state.curPlayer == Player.X else 2
            
            temp_state = TicTacToeState(np.reshape(board, (3, 3)), state.curPlayer)
            value = max(value, self.min_value(temp_state, alpha, beta))
            temp_state.board[borad_grid[action]] = 0 #reset board
            temp_state.board.flags.writeable = False
            if value >= beta:
                break
            alpha = max(alpha, value)
        return value
    
    
    def min_value(self, state, alpha, beta):
        #terminal case
        if TicTacToeState.isTerminal(state):
            return TicTacToeState.utility(state, state.curPlayer)
        
        value = np.Inf
        board = np.array(state.board).flatten()
        valid_actions = TicTacToeState.actions(state)
        for action in valid_actions:
            #whos turn
            board[action] = 2 if state.curPlayer == Player.X else 1

            temp_state = TicTacToeState(np.reshape(board, (3, 3)), state.curPlayer)
            value = min(value, self.max_value(temp_state, alpha, beta))
            temp_state.board[borad_grid[action]] = 0 #reset board
            temp_state.board.flags.writeable = False
            if value <= alpha:
                break
            beta = min(beta, value)
        return value
    
    
'''
There are two ways to write alpha-beta. First is fail-hard. Second is fail-soft. I guess you
you will learn it in CSPs, if Aj. teach.
'''
#itsmebabysmiley:)