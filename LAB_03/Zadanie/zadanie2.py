import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Dane XOR
# ------------------------------------------------------------
# Każdy wektor wejściowy ma już bias na pozycji 0:
# [1, x1, x2]
xx = np.array([
    [1, -1, -1],
    [1, -1,  1],
    [1,  1, -1],
    [1,  1,  1]
], dtype=float)

# Oczekiwane odpowiedzi XOR:
# -1 XOR -1 -> 0
# -1 XOR  1 -> 1
#  1 XOR -1 -> 1
#  1 XOR  1 -> 0
d = np.array([0, 1, 1, 0], dtype=float)


# ------------------------------------------------------------
# Funkcje aktywacji
# ------------------------------------------------------------
def sigmoid(x, beta):
    return 1.0 / (1.0 + np.exp(-beta * x))


def tanh(x, beta):
    return np.tanh(beta * x)


# ------------------------------------------------------------
# Pochodne funkcji aktywacji
# ------------------------------------------------------------
def sigmoid_diff(y, beta):
    return beta * y * (1 - y)


def tanh_diff(y, beta):
    return beta * (1 - y * y)


# ------------------------------------------------------------
# Forward pass z Zadania 1
# ------------------------------------------------------------
def mlp(x, w1, w2, beta):
    # Warstwa ukryta:
    # w1 ma rozmiar (2, 3)
    # x ma rozmiar (3,)
    # wynik: 2 wartości, po jednej dla każdego neuronu ukrytego
    v_hidden = tanh(w1 @ x, beta)

    # Dodanie biasu do wyjść warstwy ukrytej:
    # v = [1, v1, v2]
    v = np.concatenate([[1.0], v_hidden])

    # Warstwa wyjściowa:
    # w2 ma rozmiar (1, 3)
    # v ma rozmiar (3,)
    # wynik: jedna wartość wyjściowa sieci
    y = sigmoid(w2 @ v, beta)

    return y, v, v_hidden


# ------------------------------------------------------------
# Funkcja pomocnicza: błąd MSE
# ------------------------------------------------------------
def mse_error(xx, d, w1, w2, beta):
    errors = []

    for x, target in zip(xx, d):
        y, _, _ = mlp(x, w1, w2, beta)
        y_value = y[0]

        errors.append((target - y_value) ** 2)

    return np.mean(errors)


# ------------------------------------------------------------
# Funkcja pomocnicza: klasyfikacja wyniku
# ------------------------------------------------------------
def classify(y):
    y_value = y[0]

    if y_value > 0.9:
        return 1
    elif y_value < 0.1:
        return 0
    else:
        return -1


# ------------------------------------------------------------
# Funkcja pomocnicza: liczba błędów klasyfikacji
# ------------------------------------------------------------
def classification_error(xx, d, w1, w2, beta):
    errors = 0

    for x, target in zip(xx, d):
        y, _, _ = mlp(x, w1, w2, beta)
        predicted = classify(y)

        if predicted != int(target):
            errors += 1

    return errors


# ------------------------------------------------------------
# Wariant 1: aktualizacja wag po każdej próbce
# ------------------------------------------------------------
def train_sample(xx, d, eta, beta, max_epochs=100_000, return_weights=False):
    """
    Wariant 1:
    Wagi są aktualizowane po każdej próbce uczącej.

    Czyli:
    - bierzemy jedną próbkę,
    - liczymy wynik sieci,
    - liczymy błąd,
    - liczymy poprawki wag,
    - od razu poprawiamy wagi.
    """

    np.random.seed(0)

    # Wagi warstwy ukrytej: 2 neurony ukryte, każdy ma 3 wejścia: bias, x1, x2
    w1 = np.random.randn(2, 3) * 0.5

    # Wagi warstwy wyjściowej: 1 neuron wyjściowy, wejścia: bias, v1, v2
    w2 = np.random.randn(1, 3) * 0.5

    errors_history = []

    for epoch in range(max_epochs):

        for x, target in zip(xx, d):
            # -----------------------------
            # 1. Forward pass
            # -----------------------------
            y, v, v_hidden = mlp(x, w1, w2, beta)
            y_value = y[0]

            # -----------------------------
            # 2. Delta dla neuronu wyjściowego
            # -----------------------------
            # target - y_value mówi, czy wynik jest za mały, czy za duży.
            # sigmoid_diff mówi, jak mocno neuron wyjściowy reaguje na zmianę.
            delta_out = (target - y_value) * sigmoid_diff(y_value, beta)

            # -----------------------------
            # 3. Delta dla warstwy ukrytej
            # -----------------------------
            # Bierzemy wagi z warstwy wyjściowej bez biasu: w2[0, 1:]
            # ponieważ bias nie jest wyjściem żadnego neuronu ukrytego.
            delta_hidden = tanh_diff(v_hidden, beta) * (w2[0, 1:] * delta_out)

            # -----------------------------
            # 4. Aktualizacja wag warstwy wyjściowej
            # -----------------------------
            # v = [1, v1, v2]
            # czyli wejście do neuronu wyjściowego
            w2 += eta * delta_out * v.reshape(1, -1)

            # -----------------------------
            # 5. Aktualizacja wag warstwy ukrytej
            # -----------------------------
            # np.outer(delta_hidden, x) tworzy macierz (2, 3),
            # czyli taki sam rozmiar jak w1.
            w1 += eta * np.outer(delta_hidden, x)

        # Po każdej epoce liczymy MSE
        mse = mse_error(xx, d, w1, w2, beta)
        errors_history.append(mse)

        # Sprawdzamy, czy klasyfikacja jest już idealna
        class_error = classification_error(xx, d, w1, w2, beta)

        if class_error == 0:
            print(f"train_sample: zatrzymano po epoce {epoch + 1}, MSE = {mse:.6f}")
            break

    if return_weights:
        return errors_history, w1, w2

    return errors_history


# ------------------------------------------------------------
# Wariant 2: aktualizacja wag po epoce
# ------------------------------------------------------------
def train_epoch(xx, d, eta, beta, max_epochs=100_000, return_weights=False):
    """
    Wariant 2:
    Wagi są aktualizowane dopiero po przejściu wszystkich próbek.

    Czyli:
    - liczymy poprawki wag dla każdej próbki,
    - sumujemy poprawki,
    - po całej epoce aktualizujemy wagi jednym ruchem.
    """

    np.random.seed(0)

    w1 = np.random.randn(2, 3) * 0.5
    w2 = np.random.randn(1, 3) * 0.5

    errors_history = []

    for epoch in range(max_epochs):

        # Tutaj akumulujemy gradienty z całej epoki
        grad_w1 = np.zeros_like(w1)
        grad_w2 = np.zeros_like(w2)

        for x, target in zip(xx, d):
            # -----------------------------
            # 1. Forward pass
            # -----------------------------
            y, v, v_hidden = mlp(x, w1, w2, beta)
            y_value = y[0]

            # -----------------------------
            # 2. Delta wyjściowa
            # -----------------------------
            delta_out = (target - y_value) * sigmoid_diff(y_value, beta)

            # -----------------------------
            # 3. Delta warstwy ukrytej
            # -----------------------------
            delta_hidden = tanh_diff(v_hidden, beta) * (w2[0, 1:] * delta_out)

            # -----------------------------
            # 4. Akumulacja gradientów
            # -----------------------------
            grad_w2 += delta_out * v.reshape(1, -1)
            grad_w1 += np.outer(delta_hidden, x)

        # -----------------------------
        # 5. Aktualizacja wag po całej epoce
        # -----------------------------
        w2 += eta * grad_w2
        w1 += eta * grad_w1

        mse = mse_error(xx, d, w1, w2, beta)
        errors_history.append(mse)

        class_error = classification_error(xx, d, w1, w2, beta)

        if class_error == 0:
            print(f"train_epoch: zatrzymano po epoce {epoch + 1}, MSE = {mse:.6f}")
            break

    if return_weights:
        return errors_history, w1, w2

    return errors_history


# ------------------------------------------------------------
# Trening
# ------------------------------------------------------------
errors_sample, w1_sample, w2_sample = train_sample(
    xx, d, eta=0.5, beta=1.0, return_weights=True
)

errors_epoch, w1_epoch, w2_epoch = train_epoch(
    xx, d, eta=0.5, beta=1.0, return_weights=True
)


# ------------------------------------------------------------
# Sprawdzenie wyników
# ------------------------------------------------------------
print("\nWyniki dla train_sample:")
for x, target in zip(xx, d):
    y, _, _ = mlp(x, w1_sample, w2_sample, beta=1.0)
    print(f"x = {x}, oczekiwane = {int(target)}, y = {y[0]:.4f}, klasa = {classify(y)}")

print("\nWyniki dla train_epoch:")
for x, target in zip(xx, d):
    y, _, _ = mlp(x, w1_epoch, w2_epoch, beta=1.0)
    print(f"x = {x}, oczekiwane = {int(target)}, y = {y[0]:.4f}, klasa = {classify(y)}")


# ------------------------------------------------------------
# Wykres błędu
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(errors_sample, label='Aktualizacja po każdej próbce', alpha=0.8)
plt.plot(errors_epoch, label='Aktualizacja po epoce', alpha=0.8)

plt.xlabel('Epoka')
plt.ylabel('Błąd MSE')
plt.title('Porównanie dwóch wariantów aktualizacji wag')
plt.legend()
plt.yscale('log')
plt.grid(alpha=0.3)
plt.show()
