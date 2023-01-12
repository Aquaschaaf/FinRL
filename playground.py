import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import entropy

n = 10
wights = [1/n] * n
max_entropy = entropy(wights)

test_weights = [0.5, 0.5, 0.0, 0.0]

print(max_entropy)
print(entropy(test_weights))
print(entropy([1.0, 0., 0.]))
