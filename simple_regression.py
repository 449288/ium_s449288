import tensorflow as tf
from keras import layers
from keras.models import save_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Pobranie przykładowego argumentu trenowania
EPOCHS_NUM = int(sys.argv[1])

# Wczytanie danych
data_train = pd.read_csv('lego_sets_clean_train.csv')
data_test = pd.read_csv('lego_sets_clean_test.csv')

# Wydzielenie zbiorów dla predykcji ceny zestawu na podstawie liczby klocków, którą zawiera
train_piece_counts = np.array(data_train['piece_count'])
train_prices = np.array(data_train['list_price'])
test_piece_counts = np.array(data_test['piece_count'])
test_prices = np.array(data_test['list_price'])

# Normalizacja
normalizer = layers.Normalization(input_shape=[1, ], axis=None)
normalizer.adapt(train_piece_counts)

# Inicjalizacja
model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

# Kompilacja
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

# Trening
history = model.fit(
    train_piece_counts,
    train_prices,
    epochs=EPOCHS_NUM,
    verbose=0,
    validation_split=0.2
)

# Wykonanie predykcji na danych ze zbioru testującego
y_pred = model.predict(test_piece_counts)

# Zapis predykcji do pliku
results = pd.DataFrame({'test_set_piece_count': test_piece_counts.tolist(), 'predicted_price': [round(a[0], 2) for a in y_pred.tolist()]})
results.to_csv('lego_reg_results.csv', index=False, header=True)

# Zapis modelu do pliku
model.save('lego_reg_model')

# Opcjonalne statystyki, wykresy
'''
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plt.scatter(train_piece_counts, train_prices, label='Data')
plt.plot(x, y_pred, color='k', label='Predictions')
plt.xlabel('pieces')
plt.ylabel('price')
plt.legend()
plt.show()
'''
