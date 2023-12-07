import matplotlib.pyplot as plt
import numpy as np
import random

def generate_points(number_of_points):
    x = np.linspace(0, 10, number_of_points)
    y = random.uniform(-2, 2) * x + random.uniform(-1, 1) + np.random.randn(number_of_points)
    return x, y

def generate_weights():
    weights = np.array([random.uniform(0, 1) for _ in range(2)])

    return weights

def train_perceptron(x, y, learning_rate=0.01, epochs=50, delta=0.01):
    number_of_points = len(x)
    weight_a, weight_b = generate_weights()

    for curr_epoch in range(epochs):
        predicted = weight_a * x + weight_b
        error = y - predicted

        gradient_descent_a = (-2 / number_of_points) * np.dot(error, x)
        gradient_descent_b = (-2 / number_of_points) * np.sum(error)

        weight_a -= learning_rate * gradient_descent_a
        weight_b -= learning_rate * gradient_descent_b

        mean_squared_error = np.mean(error ** 2)

        if mean_squared_error < delta:
            break

    return weight_a, weight_b

def generate_plot(x, y, a, b):
    plt.scatter(x, y, label='Generated points')
    plt.plot(x, a * x + b, color='red', label=f'Approximation: y = {a:.2f}x + {b:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def main():
    x, y = generate_points(30)
    a, b = train_perceptron(x, y)
    generate_plot(x, y, a, b)

main()