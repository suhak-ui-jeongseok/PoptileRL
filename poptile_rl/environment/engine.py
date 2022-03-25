from copy import deepcopy
from collections import deque
from random import randrange
from typing import List, Tuple

from poptile_rl.environment.board import Board


# TODO: n_color, row, column은 계속 같이 재활용된다 -> 클래스 하나로 묶을 것

class Engine:
    def __init__(
        self,
        score: int,
        board: Board,
    ):
        self.score: int = score
        self.board: Board = board

        self.state_history: List[Board] = [board]
        self.action_history: List[Tuple[int, int]] = []


    def generate_row(self):
        rand_row: List[int] = [randrange(self.board.n_color) for _ in range(self.board.column)]
        self.board.pop_and_push(rand_row)


    def pop_tile(self, row: int, col: int):
        if self.board[row, col] == -1:
            raise Exception('pop_tile action should be called in valid tile.')
        n_tiles = self._search_connected_component(row, col)
        self._update_board()
        self.generate_row()

        self.score += n_tiles ** 2
        self.state_history.append(deepcopy(self.board))
        self.action_history.append((row, col))


    def _search_connected_component(self, row, col):
        queue: deque[Tuple[int, int]] = deque([(row, col)])
        color: int = self.board[row, col]

        visited: List[List[bool]] = [[False for _ in range(self.board.column)] for _ in range(self.board.row)]

        n_tiles = self._bfs(queue, visited, color)

        return n_tiles


    def _bfs(self, queue: deque, visited: List[List[bool]], color: int) -> int:
        n_tiles: int = 1

        while queue:
            cur_row, cur_col = queue.popleft()

            visited[cur_row][cur_col] = True
            self._delete_tile(cur_row, cur_col)
            n_tiles += 1

            for d_row, d_col in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                mv_row, mv_col = cur_row + d_row, cur_col + d_col
                if 0 <= mv_row < self.board.row and 0 <= mv_col < self.board.column:
                    if not visited[mv_row][mv_col] and self.board[mv_row, mv_col] == color:
                        queue.append((mv_row, mv_col))

        return n_tiles


    def _update_board(self):
        for col in range(self.board.column):
            for row in range(self.board.row):
                if self.board[row, col] >= 0:
                    tile_value = self.board[row, col]
                    self._delete_tile(row, col)

                    now_tile_row = row
                    while now_tile_row > 0:
                        if self.board[now_tile_row - 1, col] == -1:
                            now_tile_row -= 1
                        else:
                            break

                    self.board[now_tile_row, col] = tile_value


    def _delete_tile(self, row, col):
        self.board[row, col] = -1


class NewGame(Engine):
    def __init__(
        self,
        n_color: int,
        row: int,
        column: int,
    ):
        super().__init__(0, Board(n_color, row, column))
