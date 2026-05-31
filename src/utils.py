"""
Funções utilitárias para validação, impressão e visualização de tabuleiros Sudoku 4x4.
"""

import numpy as np
import matplotlib.pyplot as plt

S = {1, 2, 3, 4}


def is_valid_grid(grid):
    """
    Verifica se uma grade 4x4 é uma solução válida de Sudoku.

    Regras:
    - cada linha deve conter 1, 2, 3 e 4;
    - cada coluna deve conter 1, 2, 3 e 4;
    - cada bloco 2x2 deve conter 1, 2, 3 e 4.
    """
    grid = np.array(grid)

    if grid.shape != (4, 4):
        return False

    for row in grid:
        if set(row) != S:
            return False

    for col in grid.T:
        if set(col) != S:
            return False

    for r in range(0, 4, 2):
        for c in range(0, 4, 2):
            block = grid[r:r + 2, c:c + 2].flatten()
            if set(block) != S:
                return False

    return True


def respects_given_values(puzzle, solution):
    """
    Verifica se a solução preserva os valores originais do tabuleiro inicial.
    """
    puzzle = np.array(puzzle)
    solution = np.array(solution)

    fixed_positions = puzzle != 0
    return np.all(solution[fixed_positions] == puzzle[fixed_positions])


def plot_board(board, title, path=None):
    """
    Gera uma imagem simples do tabuleiro.
    """
    board = np.array(board)

    plt.figure(figsize=(4, 4))
    plt.imshow(np.ones((4, 4)), cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    for i in range(5):
        linewidth = 2 if i % 2 == 0 else 1
        plt.axhline(i - 0.5, color="black", linewidth=linewidth)
        plt.axvline(i - 0.5, color="black", linewidth=linewidth)

    for i in range(4):
        for j in range(4):
            value = board[i, j]
            text = "" if value == 0 else str(value)
            plt.text(j, i, text, ha="center", va="center", fontsize=18)

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.close()
