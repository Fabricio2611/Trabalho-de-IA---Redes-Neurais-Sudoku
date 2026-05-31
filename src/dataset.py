"""
Geração do conjunto de dados para treino e teste.

O projeto usa Sudoku 4x4 porque esse tamanho permite gerar todas as soluções válidas
sem custo computacional alto.
"""

import itertools
import random
import numpy as np
import torch

from utils import is_valid_grid


def generate_all_4x4_solutions():
    """
    Gera todas as soluções válidas de Sudoku 4x4.
    """
    solutions = []

    for values in itertools.product(range(1, 5), repeat=16):
        grid = np.array(values).reshape(4, 4)

        # filtro inicial para reduzir verificações desnecessárias
        if all(set(row) == {1, 2, 3, 4} for row in grid):
            if is_valid_grid(grid):
                solutions.append(grid)

    return solutions


def create_puzzle(solution, empty_rate=0.5):
    """
    Cria um tabuleiro inicial removendo valores de uma solução válida.
    """
    puzzle = solution.copy()

    for i in range(4):
        for j in range(4):
            if random.random() < empty_rate:
                puzzle[i, j] = 0

    return puzzle


def encode_board(board):
    """
    Codifica o tabuleiro de entrada em one-hot.

    Valores possíveis na entrada:
    - 0: vazio
    - 1, 2, 3, 4: números do Sudoku

    Resultado: vetor de tamanho 80.
    """
    encoded = np.zeros((4, 4, 5), dtype=np.float32)

    for i in range(4):
        for j in range(4):
            value = int(board[i, j])
            encoded[i, j, value] = 1.0

    return encoded.flatten()


def encode_solution(solution):
    """
    Codifica a solução como classes de 0 a 3.

    Classe 0 representa número 1.
    Classe 1 representa número 2.
    Classe 2 representa número 3.
    Classe 3 representa número 4.
    """
    return (solution.flatten() - 1).astype(np.int64)


def build_dataset(samples_per_solution=30):
    """
    Monta o dataset de treinamento e teste.
    """
    solutions = generate_all_4x4_solutions()

    X = []
    Y = []

    for solution in solutions:
        for _ in range(samples_per_solution):
            puzzle = create_puzzle(solution, empty_rate=random.uniform(0.35, 0.75))
            X.append(encode_board(puzzle))
            Y.append(encode_solution(solution))

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    split = int(0.8 * len(X))

    X_train = torch.tensor(X[:split])
    Y_train = torch.tensor(Y[:split])
    X_test = torch.tensor(X[split:])
    Y_test = torch.tensor(Y[split:])

    return X_train, Y_train, X_test, Y_test, solutions
