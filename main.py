import matplotlib.pyplot as plt
import numpy as np
import random

def unipolar_function(s):
    return 1 if s > 0 else 0

def generate_points():
    x = np.linspace(0, 10, 20)
    y = random.uniform(-1, 1) * x + random.uniform(-1, 1) + np.random.randn(20)
    return x, y

def generate_weights():
    weights = np.array([random.uniform(0, 1) for _ in range(2)])

    return weights

def train_perceptron(x, y, learning_rate=0.01, epochs=50, delta=0.01):
    number_of_points = len(x)
    a, b = generate_weights()

    for curr_epoch in range(epochs):
        predicted = a * x + b
        errors = y - predicted

        gradient_a = (-2 / number_of_points) * np.dot(errors, x)
        gradient_b = (-2 / number_of_points) * np.sum(errors)

        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b

        mean_squared_error = np.mean(errors ** 2)

        if mean_squared_error < delta:
            break

    return a, b