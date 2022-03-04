import sys
from typing import List, Tuple
from dataclasses import dataclass
from collections import deque
from random import randrange


class Map:
    """
    맵을 관리하는 클래스
    맵은 8x15(row, col)로 구성
    """
    def __init__(self, row=15, col=8, color=3):
        self.row = row
        self.col = col
        self.color = color

        self.dx = (-1, 1, 0, 0)
        self.dy = (0, 0, -1, 1)

        # init map
        self.map: List[List[int]] = [[-1 for _ in range(self.col)] for _ in range(self.row)]
        self.generate_row()
        self.update_map()

    def pop_tile(self, row: int, col: int) -> int:
        """
        :param: row: int, col: int (시작 row, 시작 col) 2가지 요소가 필요
        선택한 타일에서 같은 색상이 있는지 탐색하고, 그 개수를 반환하는 함수
        :return:
        """
        check_game_over: bool = self.check_game_over()
        if check_game_over:
            return -1
        n_tiles: int = self.matching_color_tiles(row, col)
        self.generate_row()
        self.update_map()

        return n_tiles

    def matching_color_tiles(self, row, col):
        color: int = self.map[row][col]
        queue: deque[Tuple[int, int, int]] = deque([(row, col, color)])
        visited: List[List[bool]] = [[False for _ in range(self.col)] for _ in range(self.row)]
        visited[row][col] = True
        n_tiles = self._bfs(queue, visited, color)

        return n_tiles

    def _bfs(self, queue, visited, color):
        n_tiles = 1
        while queue:
            cur_row, cur_col, _ = queue.popleft()
            for i in range(4):
                mv_row = cur_row + self.dx[i]
                mv_col = cur_col + self.dy[i]
                if 0 <= mv_row < self.row and 0 <= mv_col < self.col:
                    if not visited[mv_row][mv_col] and self.map[mv_row][mv_col] == color:
                        queue.append((mv_row, mv_col, color))
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
                    self.map[row][col] = -1

                    now_tile_row = row
                    while now_tile_row > 0:
                        if self.map[now_tile_row - 1][col] == -1:
                            now_tile_row -= 1
                        else:
                            break

                    self.map[now_tile_row][col] = tile_value

    def display_map(self):
        for row in self.map:
            print(row)

    def check_game_over(self) -> bool:
        """
        game over -> True
        :return:
        """
        for col in range(self.col):
            if self.map[self.row-1][col] >= 0:
                return True

        return False


class MapMovement:
    def __init__(self):
        self.touch = 0


class PopTile:
    """
    팝 타일은 3개 색상으로 이루어짐
    한번 클릭할 때 로직은 다음과 같음
    1) 해당 색상에 연결된 (상하좌우) pop
        score = (pop 개수) ^ 2
    2) 한 줄에 대해 랜덤으로 생성
        if col > 15 then Game Over

    """
    def __init__(self):
        self.map: Map = Map()
        self.score: int = 0
        self.touch: int = 0
        self.tiles: int = 0

    def run(self):
        while True:
            self.map.display_map()
            x, y = map(int, input().split())
            n_tile: int = self.map.pop_tile(x, y)
            if n_tile <= 0:
                self.game_over()
                return
            else:
                self.cal_score(n_tile)
            self.touch += 1
            self.tiles += n_tile

    def cal_score(self, n: int):
        self.score += pow(n, 2)
        sys.stdout.write(f'Score: {str(self.score)}\n')

    def game_over(self):
        game_info: Dict[str, object] = {
            'Score': self.score,
            'Touch': self.touch,
            'Score/touch': self.tiles/self.touch
        }
        sys.stdout.write(str(game_info))


pop_tile = PopTile()
pop_tile.run()
