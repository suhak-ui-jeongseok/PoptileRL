from collections import deque
from typing import Tuple

import numpy as np

from poptile_rl.environment.board import Board
from poptile_rl.environment.engine import Engine


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

    return np.var(heights)


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
                n_max_block, _bfs(mod_board, deque([(row_idx, column_idx)]), mod_board[row_idx, column_idx])
            )

    return n_max_block


def _bfs(mod_board: Board, queue: deque, color: int):
    count = 0
    while queue:
        now_row, now_col = queue.pop()

        count += 1
        mod_board[now_row, now_col] = -1
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = now_row + dr, now_col + dc
            # pylint: disable=chained-comparison
            if new_row >= 0 and new_row < mod_board.row and new_col >= 0 and new_col < mod_board.column:
                if mod_board[now_row, now_col] == color:
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

    return max(heights)


def state_value(board: Board) -> int:
    temp_values = {
        'top_height': get_top_height(board),
        'components': count_components(board),
        'var': avoid_valley(board),
    }
    return sum([temp_values['top_height'] * 100, (temp_values['components'] ** 2) / 10, temp_values['var'] * 10])


def sub_search(engine: Engine, step: int) -> Tuple[Tuple, int]:
    board = engine.board

    if step == 0:
        return (0, 0), state_value(board)

    best_action = (-1, -1)
    best_value = 10000000

    for row_idx in range(board.row):
        for column_idx in range(board.column):
            if board[row_idx, column_idx] == -1:
                continue

            action = (row_idx, column_idx)
            new_engine = engine.copy()
            new_engine.pop_tile(*action)

            if new_engine.is_gameover:
                continue

            _, value = sub_search(new_engine, step - 1)

            if best_value > value:
                best_value = value
                best_action = action

    if best_value == 10000000:
        best_action = (0, 0)

    return best_action, best_value


def search_best(engine: Engine) -> Tuple[int, int]:
    return sub_search(engine, 2)[0]
