"""
Teste da RNA treinada com tabuleiro inicial aleatório.
"""

import os
import random
import torch

from dataset import generate_all_4x4_solutions, create_puzzle
from model import SudokuMLP
from solver import solve_with_model
from utils import is_valid_grid, respects_given_values, plot_board


def test():
    os.makedirs("outputs", exist_ok=True)

    model_path = "outputs/sudoku_mlp.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Modelo não encontrado. Execute primeiro: python src/train.py"
        )

    model = SudokuMLP()
    model.load_state_dict(torch.load(model_path))

    solutions = generate_all_4x4_solutions()
    solution_original = random.choice(solutions)
    puzzle = create_puzzle(solution_original, empty_rate=0.55)

    predicted_solution = solve_with_model(model, puzzle)

    print("Tabuleiro inicial:")
    print(puzzle)

    print("\nSolução real:")
    print(solution_original)

    print("\nSolução gerada pela RNA:")
    print(predicted_solution)

    print("\nSolução válida?", is_valid_grid(predicted_solution))
    print("Preserva valores iniciais?", respects_given_values(puzzle, predicted_solution))

    plot_board(puzzle, "Tabuleiro inicial", "outputs/puzzle.png")
    plot_board(predicted_solution, "Solução prevista pela RNA", "outputs/predicted_solution.png")

    print("Imagens salvas em outputs/puzzle.png e outputs/predicted_solution.png")


if __name__ == "__main__":
    test()
