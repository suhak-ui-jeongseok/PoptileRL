from typing import List, Tuple


class Board:
    def __init__(self, n_color: int, row: int, column: int, state: List[List[int]] = None):
        self.n_color: int = n_color
        self.row: int = row
        self.column: int = column

        if state is None:
            state = [[-1 for _ in range(self.column)] for _ in range(self.row)]
        elif isinstance(state, list):
            state = [[ele for ele in line] for line in state]
        else:   
            pass

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

    # TODO: Board to numpy data
    def state_as_numpy(self) -> object:
        pass

    def state_as_array(self) -> List[List[int]]:
        return self.state
