from collections import deque
from random import randrange
from typing import List


class Map:
    """
    맵을 관리하는 클래스
    기본 맵은 15x8(row, col)로 구성
    """
    def __init__(self, row=15, col=8, color=3):
        self.row = row
        self.col = col
        self.color = color

        self.dx = (-1, 1, 0, 0)
        self.dy = (0, 0, -1, 1)

        # init map
        # [TODO]: 맵을 내부에서 생성하는 경우와 맵을 외부에서 받아서 생성하는 경우 두 가지로 나눠야 함
        self.map: List[List[int]] = [[-1 for _ in range(self.col)] for _ in range(self.row)]
        self.generate_row()
        self.update_map()

    def pop_tile(self, row: int, col: int) -> int:
        """
        :param: row: int, col: int (시작 row, 시작 col) 2가지 요소가 필요
        선택한 타일에서 같은 색상이 있는지 탐색하고, 그 개수를 반환하는 함수
        :return:
        """

        n_tiles: int = self.matching_color_tiles(row, col)
        self.update_map()
        check_game_over: bool = self.check_game_over()
        if check_game_over:
            return -1

        self.generate_row()
        return n_tiles

    def matching_color_tiles(self, row, col) -> int:
        queue: deque[Tuple[int, int]] = deque([(row, col)])
        color: int = self.map[row][col]
        visited: List[List[bool]] = [[False for _ in range(self.col)] for _ in range(self.row)]
        visited[row][col] = True
        self.delete_tile(row, col)

        n_tiles = self._bfs(queue, visited, color)

        return n_tiles

    def _bfs(self, queue: deque, visited: List[List[bool]], color: int) -> int:
        n_tiles = 1

        while queue:
            cur_row, cur_col = queue.popleft()
            for i in range(4):
                mv_row = cur_row + self.dx[i]
                mv_col = cur_col + self.dy[i]
                if 0 <= mv_row < self.row and 0 <= mv_col < self.col:
                    if not visited[mv_row][mv_col] and self.map[mv_row][mv_col] == color:
                        queue.append((mv_row, mv_col))
                        visited[mv_row][mv_col] = True
                        self.delete_tile(mv_row, mv_col)
                        n_tiles += 1

        return n_tiles

    def delete_tile(self, row, col):
        self.map[row][col] = -1

    def generate_row(self):
        self.map.pop()
        rand_row: List[int] = [randrange(self.color) for _ in range(self.col)]
        self.map.insert(0, rand_row)

    def update_map(self):
        for col in range(self.col):
            for row in range(self.row):
                if self.map[row][col] >= 0:
                    tile_value = self.map[row][col]
                    self.delete_tile(row, col)

                    now_tile_row = row
                    while now_tile_row > 0:
                        if self.map[now_tile_row - 1][col] == -1:
                            now_tile_row -= 1
                        else:
                            break

                    self.map[now_tile_row][col] = tile_value

    def display(self):
        for row in self.map:
            print(row)

    def check_game_over(self) -> bool:
        """
        check game over.
        if top tile row >= max_row
        game over -> True
        :return: bool, game over state
        """
        for col in range(self.col):
            if self.map[-1][col] >= 0:
                return True

        return False
