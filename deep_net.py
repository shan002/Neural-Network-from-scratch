import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from prepare_image_data import load_image_data

epsilon = 1e-15

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)

def leaky_relu(z):
    return (z < 0) * (0.001 * z) + (z >=0) * z


def sigmoid_backwards(A):
    return A * (1 - A)

def relu_backwards(A):
    return A > 0

def leaky_relu_backwards(A):
    return (A < 0) * 0.001 + (A > 0)

def init_parameters(layer_dims):
    parameters = {}
    momentums = {}
    L = len(layer_dims)
    for i in range(1, L):
        if g[i] == 'relu':
            parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2.0 / layer_dims[i-1])
        elif g[i] == 'tanh' or g[i] == 'sigmoid':
            parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(1.0 / layer_dims[i-1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
        momentums['VdW' + str(i)] = np.zeros_like(parameters['W' + str(i)])
        momentums['SdW' + str(i)] = np.zeros_like(parameters['W' + str(i)])
        momentums['Vdb' + str(i)] = np.zeros_like(parameters['b' + str(i)])
        momentums['Sdb' + str(i)] = np.zeros_like(parameters['b' + str(i)])
    return parameters, momentums


def forward_pass(X, parameters, g):
    layer_vals = [X]
    A = X
    L = len(g)
    for i in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(i)], A_prev) + parameters['b' + str(i)]
        A = activations[g[i]](Z)
        layer_vals.append(A)

    return layer_vals

def predict(X, parameters, g):
    layer_vals = [X]
    A = X
    L = len(g)
    for i in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(i)], A_prev) + parameters['b' + str(i)]
        A = activations[g[i]](Z)
        layer_vals.append(A)

    return layer_vals[-1][0]


def backward_pass(y_true, layer_vals, parameters, momentums, g, learning_rate=0.01, beta1=0.9, beta2=0.999, cost_func='mse'):
    m = y_true.shape[1]
    dA = compute_cost_grad(y_true, layer_vals[-1], cost_func=cost_func)
    for i in range(len(layer_vals)-1, 0, -1):
        dZ = dA * activations_backwards[g[i]](layer_vals[i])
        dA_prev = np.dot(parameters['W' + str(i)].T, dZ)
        dW = 1/m * np.dot(dZ, layer_vals[i-1].T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = dA_prev

        momentums['VdW' + str(i)] = beta1 * momentums['VdW' + str(i)] + (1-beta1) * dW
        momentums['SdW' + str(i)] = beta2 * momentums['SdW' + str(i)] + (1-beta2) * (dW ** 2)
        momentums['Vdb' + str(i)] = beta1 * momentums['Vdb' + str(i)] + (1-beta1) * db
        momentums['Sdb' + str(i)] = beta2 * momentums['Sdb' + str(i)] + (1-beta2) * (db ** 2)

        parameters['W' + str(i)] -= learning_rate * momentums['VdW' + str(i)] / np.sqrt(momentums['SdW' + str(i)] + epsilon)
        parameters['b' + str(i)] -= learning_rate * momentums['Vdb' + str(i)] / np.sqrt(momentums['Sdb' + str(i)] + epsilon)

    return parameters, momentums

def compute_cost(y, output, cost_func):
    m = y.shape[1]
    if cost_func == 'binary_cross_entropy':
        return -1/m * np.sum(y * np.log(output+epsilon) + (1-y) * np.log(1-output+epsilon))
    elif cost_func == 'mse':
        return 1/(2*m) * np.sum((output - y) ** 2)

def compute_cost_grad(y, output, cost_func):
    m = y.shape[1]
    if cost_func == 'binary_cross_entropy':
        return -y/(output + epsilon) + (1-y)/(1-output + epsilon)
    elif cost_func == 'mse':
        return output - y

def plot_decision_boundary():
    X1, X2 = np.meshgrid(np.arange(np.min(X[0, :]) - 0.5, np.max(X[0, :]) + 0.5, 0.01),
                         np.arange(np.min(X[1, :]) - 0.5, np.max(X[1, :]) + 0.5, 0.01))

    output = predict(np.array([X1.ravel(), X2.ravel()]), parameters, g)

    plt.contourf(X1, X2, output.reshape(X1.shape), cmap=ListedColormap(['red', 'blue']), alpha=0.3)
    plt.scatter(X[0, y.ravel()==1], X[1, y.ravel()==1], color='blue')
    plt.scatter(X[0, y.ravel()==0], X[1, y.ravel()==0], color='red')


activations = {
    'sigmoid': sigmoid,
    'relu': relu,
    'leaky relu': leaky_relu,
    'tanh': np.tanh
}

activations_backwards = {
    'sigmoid': sigmoid_backwards,
    'relu': relu_backwards,
    'leaky relu': leaky_relu_backwards,
    'tanh': lambda A: 1 - A ** 2
}



# First Classification
X = np.array([[0.0, 0.0, 1.0, 1.0],
              [0.0, 1.0, 0.0, 1.0]], dtype=float)
y = np.array([[0.0, 1.0, 1.0, 0.0]], dtype=float)

layer_dims = (2, 32, 32, 1)
cost_func = 'binary_cross_entropy'
# g = ['linear', 'sigmoid', 'sigmoid']
g = ['linear', 'tanh', 'tanh', 'sigmoid']
epochs = 50
learning_rate = 0.001
parameters, momentums = init_parameters(layer_dims)
layer_vals = forward_pass(X, parameters, g)
costs = []
for i in range(epochs):
    parameters, momentums = backward_pass(y, layer_vals, parameters, momentums, g, learning_rate=learning_rate, beta1=0.9, beta2=0.999, cost_func=cost_func)
    layer_vals = forward_pass(X, parameters, g)
    cost = compute_cost(y, layer_vals[-1], cost_func=cost_func)
    costs.append(cost)
    if (i+1) % 1000 == 0:
        print(f"After {i+1} epochs at learning rate {learning_rate:.4f}, cost: ", cost)

print("Loss:", costs[-1])
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.subplot(1, 2, 2)
plot_decision_boundary()
plt.show()











# Second Classification
# data = pd.read_csv('data.csv', header=None)
# X = data.iloc[:, :-1].values.T
# y = data.iloc[:, -1].values.reshape(1, -1)
# X, y = np.float_(X), np.float_(y)

# layer_dims = (2, 16, 1)
# g = ['linear', 'relu', 'sigmoid']
# cost_func = 'binary_cross_entropy'
# epochs = 200
# learning_rate = 0.01
# decay_rate = 0.0003

# parameters, momentums = init_parameters(layer_dims)
# layer_vals = forward_pass(X, parameters, g)
# costs = []
# for i in range(epochs):
#     parameters, momentums = backward_pass(y, layer_vals, parameters, momentums, g, learning_rate=learning_rate, beta1=0.9, beta2=0.999, cost_func=cost_func)
#     layer_vals = forward_pass(X, parameters, g)
#     cost = compute_cost(y, layer_vals[-1], cost_func=cost_func)
#     learning_rate = 1 / (1 + decay_rate * (i+1)) * learning_rate
#     costs.append(cost)
#     if (i+1) % 1000 == 0:
#         print(f"After {i+1} epochs at learning rate {learning_rate:.4f}, cost: ", cost)

# print("Loss:", costs[-1])
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(costs)
# plt.subplot(1, 2, 2)
# plot_decision_boundary()
# plt.show()










# # Third Classification: Hotdog Classification
# X, y = load_image_data()
# print('X data information: dtype =', X.dtype, 'Shape =', X.shape)
# print('y data information: dtype =', y.dtype, 'Shape =', y.shape)
# X, y = np.float_(X/255.0), np.float_(y)
# # print(X[:2, :])

# layer_dims = (X.shape[0], 128, 64, 20, 1)
# g = ['linear', 'relu', 'relu', 'relu', 'sigmoid']
# cost_func = 'binary_cross_entropy'
# epochs = 2000
# learning_rate = 0.001

# parameters, momentums = init_parameters(layer_dims)
# layer_vals = forward_pass(X, parameters, g)
# costs = []
# for i in range(epochs):
#     parameters, momentums = backward_pass(y, layer_vals, parameters, momentums, g, learning_rate=learning_rate, beta1=0.9, beta2=0.999, cost_func=cost_func)
#     layer_vals = forward_pass(X, parameters, g)
#     cost = compute_cost(y, layer_vals[-1], cost_func=cost_func) 
#     # plt.scatter(i, cost, s=1, color='blue')
#     # plt.pause(0.05)
#     costs.append(cost)
#     if (i+1) % 10 == 0:
#         print(f"After {i+1} epochs at learning rate {learning_rate:.4f}, cost: ", cost)

# print("Loss:", costs[-1])
# plt.plot(costs)
# plt.show()

# with open('parameters.npy', 'wb') as f:
#     np.save(f, parameters)



# # Prediction on test set
# import glob
# import cv2
# g = ['linear', 'relu', 'relu', 'sigmoid']
# with open('parameters.npy', 'rb') as f:
#     parameters = np.load(f, allow_pickle=True).item()
# WIDTH = 64
# HEIGHT = 64
# img_type = ['not_hot_dog', 'hot_dog']
# correct = 0
# n_samples = 200
# for i in range(n_samples):
#     choice_id = np.random.choice([0, 1])
#     img_num = np.random.choice(np.arange(0, n_samples, 1))
#     test_image_name = glob.glob('dataset/train/' + img_type[choice_id] + '/*.jpg')[img_num]
#     img = cv2.imread(test_image_name)
#     img_resized = cv2.resize(img, (WIDTH, HEIGHT))
#     img_processed = img_resized.reshape(1, -1).T
#     img_processed = img_processed / 255.0

#     prob = predict(img_processed, parameters, g).squeeze()
#     predicted_id = int(prob > 0.5)
#     img_class = ('not hotdog', 'hotdog')[predicted_id]
#     print('Actual:', img_type[choice_id], 'Predicted:', img_class, prob)
#     correct += choice_id == predicted_id
#     # if cv2.waitKey(1) & 0xFF != ord('q'):
#     #     cv2.imshow(img_class, img)

# print('Accuracy: ', correct/n_samples * 100.0)

#============================================================================================================================#
### In the last image classification for good result the images need to resized properly using cv2.resize() or tf.resize() ###
