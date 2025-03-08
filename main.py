"""
@author: ismailozgenc
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import struct

def read_images(file_path):
    with open(file_path, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)

def read_labels(file_path):
    with open(file_path, 'rb') as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data
    
def step_function(x):
    return 1 if x >= 0 else 0

def PerceptronTrainingAlgorithm(inputs, desired_output, random_weights, learning_rate, error_threshold, n = None):
    if n is None: n = inputs.shape[0] 
    desired_output_formatted = np.eye(10)[desired_output]
    epoch = 0
    weights = random_weights.copy()
    errors_per_epoch = []

    while True:
        errors = 0

        for i in range(n):
            v = np.dot(weights, inputs[i].flatten())
            predicted_label = np.argmax(v)
            true_label = np.argmax(desired_output_formatted[i])

            if predicted_label != true_label:
                errors += 1

        errors_per_epoch.append(errors)
        if errors / n <= error_threshold:
            break

        for i in range(n):
            v = np.dot(weights, inputs[i].flatten())
            for j in range(10):
                weights[j] += learning_rate * inputs[i].flatten() * (desired_output_formatted[i][j] - step_function(v[j]))
                
        epoch += 1

    return weights, errors_per_epoch

def TestPerceptron(weights, test_inputs, test_labels):
    errors = 0
    n_samples = test_inputs.shape[0]

    for i in range(n_samples):
        v = np.dot(weights, test_inputs[i].flatten())
        predicted_label = np.argmax(v)
        true_label = test_labels[i]

        if predicted_label != true_label:
            errors += 1

    return (errors / n_samples) * 100

train_images = read_images('train-images.idx3-ubyte')
train_labels = read_labels('train-labels.idx1-ubyte')
test_images = read_images('t10k-images.idx3-ubyte')
test_labels = read_labels('t10k-labels.idx1-ubyte')

random_weights = np.random.uniform(low=-1.0, high=1.0, size=(10, 784))

def run_perceptron_training(n, learning_rate, error_threshold):
    final_weights, errors_per_epoch = PerceptronTrainingAlgorithm(train_images[:n], train_labels[:n], random_weights, learning_rate, error_threshold, n)
    error_percentage_test = TestPerceptron(final_weights, test_images, test_labels)

    # Plotting epoch vs misclassification errors
    plt.plot(range(len(errors_per_epoch)), errors_per_epoch, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Misclassification Errors')
    plt.title(f'Epoch vs Misclassification Errors for η={learning_rate}, ϵ={error_threshold}, n={n}')
    plt.grid(True)
    plt.show()

    print(f'Error percentage on test set: {error_percentage_test:.2f}%')
    print("----------------------------------------")

print("Training on 50 samples")
run_perceptron_training(n=50, learning_rate=1, error_threshold=0)

print("Training on 1000 samples")
run_perceptron_training(n=1000, learning_rate=1, error_threshold=0)


print("Training on 60000 samples with epsilon=0.15")
all_weights, errors_per_epoch = PerceptronTrainingAlgorithm(train_images, train_labels, random_weights, learning_rate=1, error_threshold=0.15)
error_percentage_all = TestPerceptron(all_weights, test_images, test_labels)
plt.plot(range(len(errors_per_epoch)), errors_per_epoch, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassification Errors')
plt.title('Epoch vs Misclassification Errors for η=1, ϵ=0.15, n=60000')
plt.grid(True)
plt.show()

print(f'Error percentage on test set: {error_percentage_all:.2f}%')

def run_full_experiment(n, learning_rate, error_threshold, repeat=3):
    test_results = []
    for trial in range(repeat):
        print(f"\nTrial {trial + 1} with random initial weights:")
        random_weights = np.random.uniform(low=-1.0, high=1.0, size=(10, 784))

        final_weights, errors_per_epoch = PerceptronTrainingAlgorithm(
            train_images[:n], train_labels[:n], random_weights, learning_rate, error_threshold, n
        )

        plt.plot(range(len(errors_per_epoch)), errors_per_epoch, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Misclassification Errors')
        plt.title(f'Trial {trial + 1}: Epoch vs Misclassification Errors')
        plt.grid(True)
        plt.show()

        error_percentage_test = TestPerceptron(final_weights, test_images, test_labels)
        print(f'Trial {trial + 1}: Error percentage on test set: {error_percentage_test:.2f}%')
        test_results.append(error_percentage_test)

    print("\nSummary of Test Errors across Trials:")
    for trial, error in enumerate(test_results, 1):
        print(f"Trial {trial}: {error:.2f}% error")
    print(f"Average Error: {np.mean(test_results):.2f}%")
    print(f"Standard Deviation: {np.std(test_results):.2f}%")

selected_error_threshold = 0.15
learning_rate = 1
run_full_experiment(n=60000, learning_rate=learning_rate, error_threshold=selected_error_threshold)
