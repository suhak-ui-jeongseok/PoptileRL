import sys

from game_map import Map


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
            self.map.display()
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
        game_result: Dict[str, int] = {
            'Score': self.score,
            'Touch': self.touch,
            'Tiles/touch': self.tiles/self.touch
        }
        sys.stdout.write(str(game_result))


pop_tile = PopTile()
pop_tile.run()
