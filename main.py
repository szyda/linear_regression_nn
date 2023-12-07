import matplotlib.pyplot as plt
import numpy as np
import random

def unipolar_function(s):
    return 1 if s > 0 else 0

def generate_points(number_of_points):
    x = np.linspace(0, 10, number_of_points)
    y = random.uniform(-2, 2) * x + random.uniform(-1, 1) + np.random.randn(number_of_points)
    return x, y

def generate_weights():
    weights = np.array([random.uniform(0, 1) for _ in range(2)])

    return weights

def train_perceptron(x, y, learning_rate=0.01, epochs=50, delta=0.01):
    number_of_points = len(x)
    a, b = generate_weights()

    for curr_epoch in range(epochs):
        predicted = a * x + b
        error = y - predicted

        gradient_a = (-2 / number_of_points) * np.dot(error, x)
        gradient_b = (-2 / number_of_points) * np.sum(error)

        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b

        mean_squared_error = np.mean(error ** 2)

        if mean_squared_error < delta:
            break

    return a, b