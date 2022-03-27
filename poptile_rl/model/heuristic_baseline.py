from typing import Tuple
from collections import deque

from poptile_rl.environment.board import Board
from poptile_rl.environment.engine import Engine


def variance(data):
    n = len(data)
    mean = sum(data) / n
    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / n
    return variance


def avoid_valley(board: Board):
    row, column = board.row, board.column
    heights = []
    for column_idx in range(column):
        for row_idx in range(row):
            if board[row_idx, column_idx] == -1:
                break
        else:
            row_idx += 1

        heights.append(row_idx)

    return variance(heights) ** 0.5


def count_components(board: Board):
    row, column = board.row, board.column

    mod_board: Board = board.copy()

    n_components = 0
    for row_idx in range(row):
        for column_idx in range(column):
            if mod_board[row_idx, column_idx] == -1:
                continue

            n_components += 1
            _bfs(mod_board, deque([(row_idx, column_idx)]), mod_board[row_idx, column_idx])

    return n_components


def largest_block(board: Board):
    row, column = board.row, board.column

    mod_board: Board = board.copy()

    n_max_block = 0
    for row_idx in range(row):
        for column_idx in range(column):
            if mod_board[row_idx, column_idx] == -1:
                continue

            n_max_block = max(
                n_max_block,
                _bfs(mod_board, deque([(row_idx, column_idx)]), mod_board[row_idx, column_idx])
            )

    return n_max_block


def _bfs(mod_board: Board, queue: deque, color: int):
    count = 0
    while queue:
        now_row, now_col = queue.pop()
        if mod_board[now_row, now_col] != color:
            continue

        count += 1
        mod_board[now_row, now_col] = -1
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = now_row + dr, now_col + dc
            if 0 <= new_row and new_row < mod_board.row and 0 <= new_col and new_col < mod_board.column:
                    queue.appendleft((new_row, new_col))

    return count


def get_top_height(board: Board) -> int:
    row, column = board.row, board.column
    heights = []
    for column_idx in range(column):
        for row_idx in range(row):
            if board[row_idx, column_idx] == -1:
                break
        else:
            row_idx += 1

        heights.append(row_idx)

    heights.sort()

    return heights[-1]


def search_best(engine: Engine) -> Tuple[int, int]:
    board = engine.board
    row, column = board.row, board.column


    best_action = (-1, -1)
    best_value = (100, 100)
    for row_idx in range(row):
        for column_idx in range(column):
            if board[row_idx, column_idx] == -1:
                continue

            new_engine = engine.copy()
            new_engine.pop_tile(row_idx, column_idx)

            if new_engine.is_gameover():
                continue

            temp_value = (get_top_height(new_engine.board), count_components(new_engine.board))
            if best_value > temp_value:
                best_action = (row_idx, column_idx)
                best_value = temp_value

    if best_action == (-1, -1):
        best_action = (0, 0)

    print(best_value, best_action)
    return best_action