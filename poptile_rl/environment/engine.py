from collections import deque
from random import randrange
from typing import List, Tuple

from poptile_rl.environment.board import Board


class Engine:
    def __init__(self, score: int, board: Board):
        self.gameover_state = False
        self.score = score
        self.board = board

        self.last_action: Tuple[int, int] = -1, 1

    def copy(self):
        return Engine(self.score, self.board.copy())

    def generate_row(self):
        rand_row: List[int] = [randrange(self.board.n_color) for _ in range(self.board.column)]
        self.board.pop_and_push(rand_row)

    def pop_tile(self, row: int, col: int):
        if self.board[row, col] == -1:
            raise Exception('pop_tile action should be called in valid tile.')

        num_tiles: int = self.search_connected_component(row, col)
        self.update_board()

        self.update_gameover_state()
        self.generate_row()

        self.score += num_tiles ** 2
        self.last_action = (row, col)

    def is_gameover(self):
        return self.gameover_state

    def update_gameover_state(self):
        result = False
        for value in self.board.state[-1]:
            if value != -1:
                result = True

        self.gameover_state |= result

    def search_connected_component(self, row: int, col: int) -> int:
        queue: deque[Tuple[int, int]] = deque([(row, col)])
        color: int = self.board[row, col]

        visited: List[List[bool]] = [[False for _ in range(self.board.column)] for _ in range(self.board.row)]

        num_tiles = self.bfs(queue, visited, color)

        return num_tiles

    def bfs(self, queue: deque, visited: List[List[bool]], color: int) -> int:
        num_tiles: int = 1

        while queue:
            cur_row, cur_col = queue.popleft()

            visited[cur_row][cur_col] = True
            self._delete_tile(cur_row, cur_col)
            num_tiles += 1

            for d_row, d_col in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                mv_row, mv_col = cur_row + d_row, cur_col + d_col
                if 0 <= mv_row < self.board.row and 0 <= mv_col < self.board.column:
                    if not visited[mv_row][mv_col] and self.board[mv_row, mv_col] == color:
                        queue.append((mv_row, mv_col))

        return num_tiles

    def update_board(self):
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
    def __init__(self, n_color: int, row: int, column: int):
        super().__init__(0, Board(n_color, row, column))
