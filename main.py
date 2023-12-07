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