import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import time
import pandas as pd

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define hyperparameter configurations
experiments = [
    # Table 1 - Single Hyperparameter Changes
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    # 1
    {'learning_rate': 0.01,  'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    {'learning_rate': 0.1,   'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    # 2
    {'learning_rate': 0.001, 'batch_size': 16, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    {'learning_rate': 0.001, 'batch_size': 64, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    # 3
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs':  5, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    # 4
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.4, 'hidden_layer_size': 64},
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.8, 'hidden_layer_size': 64},
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.95, 'hidden_layer_size': 64},
    # 5
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 16},
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 128},

    # Table 2 - Double and Triple Hyperparameter Changes
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    # 6
    {'learning_rate':  0.01, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    {'learning_rate':   0.1, 'batch_size': 32, 'epochs': 40, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    # 7
    {'learning_rate': 0.001, 'batch_size': 128, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 256},
    {'learning_rate': 0.001, 'batch_size': 256, 'epochs': 10, 'dropout_rate': 0.2, 'hidden_layer_size': 512},
    # 8
    {'learning_rate': 0.001, 'batch_size':   32, 'epochs': 2, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    {'learning_rate': 0.001, 'batch_size': 1024, 'epochs': 2, 'dropout_rate': 0.2, 'hidden_layer_size': 64},
    # 9
    {'learning_rate': 0.001, 'batch_size': 4096, 'epochs': 20, 'dropout_rate': 0.2, 'hidden_layer_size': 4096},
]

# Prepare results list
results = []

# Run each experiment
for i, params in enumerate(experiments):
    print(f"Running Experiment {i+1} with Parameters: {params}")
    
    # Build the model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(params['hidden_layer_size'], activation='relu'),
        Dropout(params['dropout_rate']),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    start_time = time.perf_counter()
    model.fit(train_images, train_labels, epochs=params['epochs'], batch_size=params['batch_size'], validation_split=0.1, verbose=0)
    end_time = time.perf_counter()

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    time_lapse = end_time - start_time

    # Save results
    results.append({
        'Experiment': i+1,
        'Learning Rate': params['learning_rate'],
        'Batch Size': params['batch_size'],
        'Epochs': params['epochs'],
        'Dropout Rate': params['dropout_rate'],
        'Hidden Layer Size': params['hidden_layer_size'],
        'Training Time (s)': time_lapse,
        'Test Accuracy': test_acc
    })

# Convert results to DataFrame and display
df_results = pd.DataFrame(results)
print(df_results)

'''

0            1          0.001          32      10          0.20                 64          27.600037         0.9751

1            2          0.010          32      10          0.20                 64          28.617818         0.9573
2            3          0.100          32      10          0.20                 64          27.308780         0.4365

3            4          0.001          16      10          0.20                 64          55.145989         0.9717
4            5          0.001          64      10          0.20                 64          17.459394         0.9739

5            6          0.001          32       5          0.20                 64          16.582185         0.9709
6            7          0.001          32      20          0.20                 64          61.248714         0.9778

7            8          0.001          32      10          0.40                 64          30.778587         0.9726
8            9          0.001          32      10          0.80                 64          31.209493         0.9378
9           10          0.001          32      10          0.95                 64          30.725528         0.8854

10          11          0.001          32      10          0.20                 16          25.917016         0.9363
11          12          0.001          32      10          0.20                128          37.923286         0.9790


12          13          0.001          32      10          0.20                 64          30.249077         0.9739

13          14          0.010          32      20          0.20                 64          46.470757         0.9598
14          15          0.100          32      40          0.20                 64         103.625153         0.2881

15          16          0.001         128      10          0.20                256          11.917594         0.9809
16          17          0.001         256      10          0.20                512          13.343946         0.9796

17          18          0.001          32       2          0.20                 64           7.228268         0.9609
18          19          0.001        1024       2          0.20                 64           1.850264         0.9128

19          20          0.001        4096      20          0.20               4096          54.871067         0.9819

'''
