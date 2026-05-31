"""
Modelo de Rede Neural Artificial Multicamadas para Sudoku 4x4.
"""

import torch.nn as nn


class SudokuMLP(nn.Module):
    """
    RNA multicamadas.

    Entrada:
    - 80 valores, pois o tabuleiro 4x4 é codificado como 4x4x5.

    Saída:
    - 16 células;
    - cada célula possui 4 classes possíveis: 1, 2, 3 ou 4.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(80, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, 16, 4)
