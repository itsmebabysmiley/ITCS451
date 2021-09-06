"""This module contains classes and functions for Tic-Tac-Toe.

Members:
1. Name: Nopparat Pengsuk  ID: 6288103
2. Name:  ID:
3. Name:  ID:
4. Name:  ID:

"""
from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Tuple

from tabulate import tabulate
import numpy as np

symbol_map = ['_', 'X', 'O']

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
        #start here
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
        # return TicTacToeState(.....)
        
        #start here
        board = np.array(state.board).flatten()
        if board[action] != 0 or action == None:
            return None
        else:
            board[action] = 1 if state.curPlayer == Player.X else 2
            board_2d = np.reshape(board, (3, 3))
            if state.curPlayer == Player.X:
                next_play = Player.O
            else:
                next_play = Player.X

            board_2d.flags.writeable = False
            new_board = TicTacToeState(board_2d, next_play)
            return new_board


    @classmethod
    def is_someone_win(cls, board: np.array) -> Tuple[bool, Player] :

        win = [
                [0,1,2],
                [3,4,5],
                [6,7,8],
                [0,3,6],
                [1,4,7],
                [2,5,8],
                [0,4,8],
                [2,4,6]]
       
       
        position = np.where(board.flatten() == 1)
        for w in win:
            if all(x in position[0] for x in w):
                return True, Player.X
        position = np.where(board.flatten() == 2)
        for w in win:
            if all(x in position[0] for x in w):
                return True, Player.O
        #someone win?
        
        return False, None


    # TODO 4: Create a terminal test function
    @classmethod
    def isTerminal(cls, state: TicTacToeState) -> bool:
        """Return `True` is the `state` is terminal (end of the game)."""

        #start here
        win = TicTacToeState.is_someone_win(state.board)
        if win[0]:
            return True

        # #board full? tie?
        full = np.where(state.board.flatten() == 0)
        if full[0].size == 0:
            return True

        #not full and not win
        return False

    # TODO 5: Create a utility function
    @classmethod
    def utility(cls, state: TicTacToeState, player: Player) -> Union[float, None]:
        """Return the utility of `player` for the `state`.

        If the state is non-terminal, return None.

        The `player` can be different than the `state`.`curPlayer`.
        """

        #utilities are -1,0,+1
        
        if not TicTacToeState.isTerminal(state):
            return None

        win = TicTacToeState.is_someone_win(state.board)
        win, p = win
                
        if not win:
            return 0

        if win and (p == player):
            return 1
        else:
            return -1


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
        gBoard = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8],
            ])

        maxEval = np.NINF
        action = None
        board = np.array(state.board)
        for i in range(3):
            for j in range(3):
                if board[i,j] == 0:
                    if self.player == Player.X:
                        board[i,j] = 1
                    else:
                        board[i,j] = 2
                    eva = MinimaxBot.minimax(board, 0 , False)
                    board[i,j] = 0
                    if(eva > maxEval):
                        maxEval = eva
                        action = (i,j)
                        
        return gBoard[action] 

    #my utility
    def checkWinner(board):
        win = [
                [0,1,2],
                [3,4,5],
                [6,7,8],
                [0,3,6],
                [1,4,7],
                [2,5,8],
                [0,4,8],
                [2,4,6]]

        position = np.where(board.flatten() == 1)
        for w in win:
            if all(x in position[0] for x in w):
                return 1
        position = np.where(board.flatten() == 2)
        for w in win:
            if all(x in position[0] for x in w):
                return -1
        full = np.where(board.flatten() == 0)
        if full[0].size == 0:
            return 0
        
        return None


    def minimax(board, depth , maximizingPlayer):
        # print(f'depth = {depth}')
        #terminal case
        win = MinimaxBot.checkWinner(board)
        if win != None:
            return win
        
        if maximizingPlayer:
            value = np.NINF
            for i in range(3):
                for j in range(3):
                    if board[i,j] == 0:
                        board[i,j] = 1
                        score = MinimaxBot.minimax(board, depth+1, False)
                        board[i,j] = 0
                        value = max(value, score)
            # print(f'max value = {value}')
            return value

        # minimizingPlayer
        else:
            value = np.Inf
            for i in range(3):
                for j in range(3):
                    if board[i,j] == 0:
                        board[i,j] = 2
                        score = MinimaxBot.minimax(board, depth+1, True)
                        board[i,j] = 0
                        value = min(value, score)
            return value
        


class AlphaBetaBot(StupidBot):

    def __init__(self, player: Player) -> None:
        super().__init__(player)

    # TODO 7: Implement Alpha-Beta Decision algorithm
    def play(self, state: TicTacToeState) -> Union[int, None]:
        """Return an action to play or None to skip.""" 
        gBoard = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8],
            ])

        maxEval = np.NINF
        action = None
        board = np.array(state.board)
        for i in range(3):
            for j in range(3):
                if board[i,j] == 0:
                    board[i,j] = 1
                    eva = MinimaxBot.minimax(board, 0 , False)
                    board[i,j] = 0
                    if(eva > maxEval):
                        maxEval = eva
                        action = (i,j)
                        
        return gBoard[action] 

    
    def minimax( board, depth, maximizingPlayer, alpha, beta ):
        #terminal case
        win = MinimaxBot.checkWinner(board)
        if win != None:
            return win
        
        if maximizingPlayer:
            value = np.NINF
            flag = False
            for i in range(3):
                for j in range(3):
                    if board[i,j] == 0:
                        board[i,j] = 1
                        score = AlphaBetaBot.minimax(board, depth+1, False, alpha, beta)
                        board[i,j] = 0
                        value = max(value, score)
                        alpha = max(alpha, score)
                        if beta <= alpha:
                            flag = True
                            break
                if flag:
                    break
            # print(f'max value = {value}')
            return value

        # minimizingPlayer
        else:
            value = np.Inf
            flag = False
            for i in range(3):
                for j in range(3):
                    if board[i,j] == 0:
                        board[i,j] = 2
                        score = AlphaBetaBot.minimax(board, depth+1, True, alpha, beta)
                        board[i,j] = 0
                        value = min(value, score)
                        beta = min(beta, score)
                        if beta <= alpha:
                            flag = True
                            break
                if flag:
                    break
            return value