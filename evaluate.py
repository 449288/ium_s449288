import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# Załadowanie modelu z pliku
model = keras.models.load_model('lego_reg_model')

# Załadowanie zbioru testowego
data_test = pd.read_csv('lego_sets_clean_test.csv')
test_piece_counts = np.array(data_test['piece_count'])
test_prices = np.array(data_test['list_price'])

# Prosta ewaluacja (mean absolute error)
test_results = model.evaluate(
    test_piece_counts,
    test_prices, verbose=0)

# Zapis wartości liczbowej metryki do pliku
with open('eval_results.txt', 'a+') as f:
    f.write(str(test_results) + '\n')

# Wygenerowanie i zapisanie do pliku wykresu
with open('eval_results.txt') as f:
    scores = [float(line) for line in f if line]
    builds = list(range(1, len(scores) + 1))

    plot = plt.plot(builds, scores)
    plt.xlabel('Build number')
    plt.xticks(range(1, len(scores) + 1))
    plt.ylabel('Mean absolute error')
    plt.title('Model error by build')
    plt.savefig('error_plot.jpg')
    plt.show()
    