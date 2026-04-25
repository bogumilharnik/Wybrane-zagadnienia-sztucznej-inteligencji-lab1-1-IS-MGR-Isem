import numpy as np

def sigmoid(x, beta):
    return 1.0 / (1.0 + np.exp(-beta * x))

def tanh(x, beta):
    return np.tanh(beta * x)

def mlp(x, w1, w2, beta):
    # Krok 1: warstwa ukryta
    # x ma już bias: [1, x1, x2]
    # W1 to macierz (2×3), więc w1 @ x daje wektor 2 liczb
    v_hidden = tanh(w1 @ x, beta)          # → [v1, v2]

    # Krok 2: doklejamy bias do wyjść ukrytej
    v = np.concatenate([[1.0], v_hidden])  # → [1, v1, v2]

    # Krok 3: warstwa wyjściowa
    # W2 to macierz (1×3), wynik to [y] (tablica z 1 elementem)
    y = sigmoid(w2 @ v, beta)              # → [y]

    return y, v, v_hidden

# --- Test ---
np.random.seed(0)
w1_test = np.random.randn(2, 3) * 0.5
w2_test = np.random.randn(1, 3) * 0.5
y_out, v_out, v_hidden_out = mlp(np.array([1, 0, 1]), w1_test, w2_test, beta=1.0)
print(f"Wyjście sieci:        y        = {y_out}")
print(f"Wyjścia ukrytej+bias v        = {v_out}")
print(f"Wyjścia ukrytej      v_hidden = {v_hidden_out}")
