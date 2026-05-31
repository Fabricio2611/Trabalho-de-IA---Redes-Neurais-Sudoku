"""
Funções para usar a RNA treinada como resolvedor aproximado.
"""

import numpy as np
import torch

from dataset import encode_board


def solve_with_model(model, puzzle):
    """
    Recebe um tabuleiro parcial e retorna a solução prevista pela RNA.
    """
    model.eval()

    x = torch.tensor(
        encode_board(np.array(puzzle)),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    predicted = torch.argmax(output, dim=2).numpy().reshape(4, 4) + 1

    return predicted
