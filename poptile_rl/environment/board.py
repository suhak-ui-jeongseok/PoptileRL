from typing import List, Tuple

import numpy as np


class Board:
    def __init__(self, n_color: int, row: int, column: int, state: List[List[int]] = None):
        self.n_color: int = n_color
        self.row: int = row
        self.column: int = column

        if state is None:
            state = [[-1 for _ in range(self.column)] for _ in range(self.row)]
        elif isinstance(state, list):
            pass
        else:
            raise Exception('state type error')

        self.state: List[List[int]] = state

    def copy(self):
        return Board(self.n_color, self.row, self.column, self.state)

    def pop_and_push(self, line: List[int]):
        self.state.pop()
        self.state.insert(0, line)

    def __setitem__(self, key: Tuple[int, int], value: int):
        row, column = key
        self.state[row][column] = value

    def __getitem__(self, key: Tuple[int, int]) -> int:
        row, column = key
        return self.state[row][column]

    def state_as_numpy(self) -> np.ndarray:
        # 0: empty, 1: color1, 2: color2, ...
        array = np.zeros((self.n_color + 1, self.row, self.column))
        for row in range(self.row):
            for column in range(self.column):
                if self.state[row][column] == -1:
                    array[0][row][column] = 1
                else:
                    array[self.state[row][column] + 1][row][column] = 1
        return array

    def state_as_array(self) -> List[List[int]]:
        return self.state
