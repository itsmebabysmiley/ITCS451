"""This module contains classes and functions for Tic-Tac-Toe.
V2
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
from typing import List, Union

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
        position = np.array(
            [
                [0,1,2],
                [3,4,5],
                [6,7,8]
            ]
        )
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if state.board[i,j] == 0:
                    valid_actions.append(position[i,j])
        return valid_actions

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
        assert action != None, 'action is NOne????? #actions()'
        # print(f'action = {action}')
        new_board = np.array(state.board)
        next_player = None
        
        if state.curPlayer == Player.X:
            next_player = Player.O
        else:
            next_player = Player.X

        count = 0
        for i in range(3):
            for j in range(3):
                if count == action:
                    new_board[i,j] = state.curPlayer.value
                    return TicTacToeState(new_board, next_player)
                count+= 1
        #error debugging
        return 'asdfsadfsdfasdfasdfasdf'


    def checkWinner(board: np.array) -> Union[int, None]:
        '''
        If player won, then return that player.
        
        If tie, then return tie value(0).  
        
        If the game isn't finish, return None  
        
        source: https://stackoverflow.com/questions/39922967/python-determine-tic-tac-toe-winner
        '''
        #rows
        rows = np.array(board)
        for row in rows:
            if len(set(row)) == 1 and (0 not in set(row)):
                return row[0]
        #columns by transpose
        columns = np.transpose(board)
        for column in columns:
            if len(set(column)) == 1 and (0 not in set(column)):
                return column[0]
        #diagonals
        if len(set( board[i,i] for i in range(3))) == 1 and (0 not in set( board[i,i] for i in range(3))):
            return board[0,0]
        if len(set( board[i,3-i-1] for i in range(3))) == 1 and (0 not in set( board[i,3-i-1] for i in range(3))):
            return board[0,len(board)-1]
        
        #tie?
        for row in board:
            if len(set(row)) > 1 and (0 not in set(row)) :
                continue
            else:
                return None
        return 0


    # TODO 4: Create a terminal test function
    @classmethod
    def isTerminal(cls, state: TicTacToeState) -> bool:
        """Return `True` is the `state` is terminal (end of the game)."""
        
        board = np.array(state.board)
        if TicTacToeState.checkWinner(board) == None:
            return False
        
        return True

    # TODO 5: Create a utility function
    @classmethod
    def utility(cls, state: TicTacToeState, player: Player) -> Union[float, None]:
        """Return the utility of `player` for the `state`.

        If the state is non-terminal, return None.

        The `player` can be different than the `state`.`curPlayer`.
        """
        score = TicTacToeState.checkWinner(state.board)
        assert score != -1, "It's not terminate #utility()"
        
        if score == 0:
            return 0 #tie
        if score == player.value:
            return 1 #Player won
        else:
            return -1 #Player lost
        


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
        board = np.array(state.board)
        maxValue = np.NINF
        position = np.array(
            [
                [0,1,2],
                [3,4,5],
                [6,7,8]
            ]
        )
        move = None
        valid_actions = TicTacToeState.actions(state)
        for i in range(3):
            for j in range(3):
                if board[i,j] == 0:
                    temp_state = TicTacToeState(board,self.player)
                    next_temp_state = TicTacToeState.transition(temp_state, position[i,j])
                    value = self.minimax(next_temp_state, False)
                    next_temp_state.board[i,j] = 0
                    print(f'value = {value}')
                    if value > maxValue:
                        maxValue = value
                        move = position[i,j]
                    
        return move

    
    def minimax(self, node, maximizingPlayer):
        #basecase
        if TicTacToeState.isTerminal(node):
            u = TicTacToeState.utility(node, self.player)
            # print(u)
            return u
        
        if maximizingPlayer:
            board = np.array(node.board)
            value = np.NINF
            position = np.array(
                [
                    [0,1,2],
                    [3,4,5],
                    [6,7,8]
                ]
            )
            for i in range(3):
                for j in range(3):
                    if board[i,j] == 0:
                        temp_state = TicTacToeState(board, node.curPlayer)
                        next_temp_state = TicTacToeState.transition(temp_state, position[i,j])
                        temp_score = self.minimax(next_temp_state, False)
                        # next_temp_state.board[i,j] = 0
                        value = max(value, temp_score)
            return value
        #minimize
        else:
            board = np.array(node.board)
            value = np.inf
            position = np.array(
                [
                    [0,1,2],
                    [3,4,5],
                    [6,7,8]
                ]
            )
            for i in range(3):
                for j in range(3):
                    if board[i,j] == 0:
                        temp_state = TicTacToeState(board, node.curPlayer)
                        next_temp_state = TicTacToeState.transition(temp_state, position[i,j])
                        temp_score = self.minimax(next_temp_state, True)
                        # next_temp_state.board[i,j] = 0
                        value = min(value, temp_score)
            return value


class AlphaBetaBot(StupidBot):

    def __init__(self, player: Player) -> None:
        super().__init__(player)

    # TODO 7: Implement Alpha-Beta Decision algorithm
    def play(self, state: TicTacToeState) -> Union[int, None]:
        """Return an action to play or None to skip."""
        return super().play(state)