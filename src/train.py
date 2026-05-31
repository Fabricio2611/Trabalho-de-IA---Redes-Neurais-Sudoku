"""
Código principal de treinamento da RNA.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import build_dataset
from model import SudokuMLP


def train():
    os.makedirs("outputs", exist_ok=True)

    X_train, Y_train, X_test, Y_test, solutions = build_dataset(samples_per_solution=30)

    print("Total de soluções válidas 4x4:", len(solutions))
    print("Amostras de treino:", X_train.shape[0])
    print("Amostras de teste:", X_test.shape[0])

    model = SudokuMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)

        loss = 0
        for cell in range(16):
            loss += criterion(outputs[:, cell, :], Y_train[:, cell])

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "outputs/sudoku_mlp.pth")
    print("Modelo salvo em outputs/sudoku_mlp.pth")

    plt.figure()
    plt.plot(losses)
    plt.title("Evolução da perda durante o treinamento")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.savefig("outputs/loss_curve.png", bbox_inches="tight")
    plt.close()

    print("Curva de perda salva em outputs/loss_curve.png")


if __name__ == "__main__":
    train()
