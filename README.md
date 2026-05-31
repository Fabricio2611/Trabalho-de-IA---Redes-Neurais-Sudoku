# RNA Multicamadas para Resolver Sudoku 4x4

Este projeto propõe uma solução com Rede Neural Artificial multicamadas, implementada em Python com PyTorch, para reconhecer e completar tabuleiros de Sudoku 4x4 com subgrupos 2x2.

**Integrantes**
  - Alexandre Antonaccio Senna
  - Fabricio Lessa Lorenzi Filho
  - Jurandy Alves Nogueira Junior
  - Tiago Rodrigues Bezerra


O conjunto de símbolos utilizado é:

```text
S = {1, 2, 3, 4}
```

As células vazias são representadas por `0`.

---

## Objetivo

A RNA recebe um tabuleiro inicial 4x4 parcialmente preenchido e tenta gerar uma solução completa válida.

O projeto também discute uma limitação importante: Sudoku é um problema de raciocínio lógico e satisfação de restrições. Portanto, uma RNA pura pode aprender padrões, mas não garante validade matemática sem uma etapa de verificação.

---

## Regras consideradas

O Sudoku 4x4 deve respeitar:

1. Cada célula recebe apenas um número de `S = {1, 2, 3, 4}`.
2. Não há repetição em linhas, colunas ou subgrupos 2x2.
3. Cada linha e coluna da grade 4x4 contém exatamente os números 1, 2, 3 e 4.
4. Cada subgrupo 2x2 contém exatamente os números 1, 2, 3 e 4.

---

## Estrutura do projeto

```text
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── solver.py
│   └── utils.py

```

---

# Explicação da solução

## Por que usar uma RNA?

A Rede Neural Artificial multicamadas é usada para aprender uma relação entre um tabuleiro parcial de Sudoku 4x4 e sua respectiva solução completa.

A entrada da rede é um vetor codificado em one-hot representando o tabuleiro inicial. A saída é composta por 16 classificações, uma para cada célula do tabuleiro.

Cada célula pode assumir uma das quatro classes possíveis:

```text
1, 2, 3 ou 4
```

## Codificação

Como o tabuleiro também pode possuir células vazias, a entrada utiliza cinco possibilidades:

```text
0, 1, 2, 3, 4
```

O valor `0` representa uma célula vazia.

Assim, um tabuleiro 4x4 gera:

```text
4 x 4 x 5 = 80 entradas
```

## Saída da rede

A saída possui:

```text
16 células x 4 classes = 64 valores
```

Depois, a classe de maior probabilidade em cada célula é convertida para um número de 1 a 4.

## Problema da geração de amostras

Gerar amostras e testá-las parece simples, mas isso é força bruta. Para Sudoku 4x4, ainda é viável. Para Sudoku NxN, o número de combinações cresce rapidamente.

Além disso, Sudoku não é apenas reconhecimento de padrões. É um problema de raciocínio com restrições globais.

Uma escolha errada em uma célula pode invalidar uma linha, uma coluna e um subgrupo ao mesmo tempo.

## Melhorias possíveis

Uma solução mais forte poderia usar a RNA apenas para sugerir probabilidades e depois aplicar um algoritmo lógico para corrigir conflitos.

Exemplos:

- backtracking;
- SAT solver;
- programação por restrições;
- Logic Tensor Networks;
- redes neurais com perdas baseadas em restrições.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/sudoku-rna-4x4.git
cd sudoku-rna-4x4
```

Crie um ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente:

No Windows:

```bash
venv\Scripts\activate
```

No Linux/Mac:

```bash
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Como treinar

Execute:

```bash
python src/train.py
```

O modelo treinado será salvo em:

```text
outputs/sudoku_mlp.pth
```

Também será gerada uma imagem da curva de perda:

```text
outputs/loss_curve.png
```

---

## Como testar

Depois do treino, execute:

```bash
python src/test.py
```

O programa irá:

- gerar um tabuleiro inicial aleatório;
- pedir para a RNA gerar uma solução;
- comparar com a solução real;
- verificar se a saída é válida;
- salvar imagens do tabuleiro inicial e da solução prevista.

---

## Exemplo de saída

```text
Tabuleiro inicial:
[[0 2 0 4]
 [3 0 1 0]
 [0 1 0 3]
 [4 0 2 0]]

Solução real:
[[1 2 3 4]
 [3 4 1 2]
 [2 1 4 3]
 [4 3 2 1]]

Solução gerada pela RNA:
[[1 2 3 4]
 [3 4 1 2]
 [2 1 4 3]
 [4 3 2 1]]

Solução válida? True
```

---
# Outputs

Esta pasta recebe os arquivos gerados pelo projeto, como:

- modelo treinado `.pth`;
- curva de perda do treinamento;
- imagem do tabuleiro inicial;
- imagem da solução prevista pela RNA.

## Observação importante

Este projeto usa uma RNA como aproximação. A rede aprende padrões a partir de exemplos, mas Sudoku não é naturalmente um problema apenas estatístico. Ele é um problema de restrições.

Por isso, a função de validação é essencial. Em aplicações mais robustas, o ideal seria combinar:

- RNA;
- backtracking;
- SAT solver;
- programação por restrições;
- ou Logic Tensor Networks.

---

## Limitação da generalização para NxN

A solução 4x4 é pequena e permite gerar todas as soluções válidas. Porém, ao generalizar para NxN, o espaço de possibilidades cresce de forma explosiva.

Gerar amostras aleatórias e testá-las se torna ineficiente porque a maioria das matrizes geradas não respeita as restrições do Sudoku.

Portanto, para Sudoku 9x9 ou maior, não basta gerar exemplos e treinar uma RNA simples. É necessário incorporar raciocínio lógico ou mecanismos explícitos de restrição.

