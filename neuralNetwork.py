import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from IPython.display import Image

nn_architecture = [
	{"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

#初始化网络权重
def init_layers(nn_architecture, seed = 99):
	np.random.seed(seed)
	number_of_layers = len(nn_architecture)
	params_values = {}

	for idx, layer in enumerate(nn_architecture):
		layer_idx = idx + 1
		layer_input_size = layer["input_dim"]
		layer_output_size = layer["output_dim"]
		params_values["W" + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
		params_values["b" + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

	return params_values

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def relu(Z):
	return np.maximum(0,Z)

def sigmoid_backword(dA, Z):
	sig = sigmoid(Z)
	return dA * sig * (1 - sig)

def relu_backword(dA, Z):
	dZ = np.array(dA, copy = True)
	dZ[Z <= 0] = 0
	return dZ

def single_layer_forward_propagation(A_prew, W_curr, b_curr, activation="relu"):
	#计算激活函数的输入
	Z_curr = np.dot(W_curr, A_prew) + b_curr

	if activation is "relu":
		activation_func = relu
	elif activation is "sigmoid":
		activation_func = sigmoid
	else:
		raise Exception("Non-support activation function")

	#返回activation值及中间值Z_curr
	return activation_func(Z_curr), Z_curr

def full_forward_forward_propagation(X, params_values, nn_architecture):
	#临时内存，存放backward步骤所需要的信息
	memory = {}
	#X向量是第0层的activation
	A_curr = X

	for idx, layer in enumerate(nn_architecture):
		#网络从第1层开始
		layer_idx = idx + 1
		#把前一层的activation传递过来
		A_prew = A_curr

		active_function_curr = layer["activation"]
		W_curr = params_values["W" + str(layer_idx)]
		b_curr = params_values["b" + str(layer_idx)]
		A_curr, Z_curr = single_layer_forward_propagation(A_prew, W_curr, b_curr, active_function_curr)

		#存储前一层的activation
		memory["A" + str(idx)] = A_prew
		memory["Z" + str(layer_idx)] = Z_curr

	return A_curr, memory

def get_cost_value(Y_hat, Y):
	#example的数量
	m = Y_hat.shape[1]
	cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
	return np.squeeze(cost)

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
	Y_hat_ = convert_prob_into_class(Y_hat)
	return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prew, activation="relu"):
	#example的数量
	m = A_prew.shape[1]

	if activation is "relu":
		backward_activation_func = relu_backword
	elif activation is "sigmoid":
		backward_activation_func = sigmoid_backword
	else:
		raise Exception("Non-support activation_func")
	#激活函数的偏导
	dZ_curr = backward_activation_func(dA_curr, Z_curr)
	#矩阵W的偏导
	dW_curr = np.dot(dZ_curr, A_prew.T) / m
	#向量b的偏导
	db_curr = np.sum(dZ_curr, axis = 1, keepdims = True) / m
	#A_prew的偏导
	dA_prew = np.dot(W_curr.T, dZ_curr)

	return dA_prew, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
	grads_values = {}
	#examples的数量
	m = Y.shape[1]

	Y = Y.reshape(Y_hat.shape)
	#梯度下降算法初始化
	dA_prew = - (np.divide(Y, Y_hat) - np.divide(1-Y, 1 - Y_hat))

	for layer_idx_prew, layer in reversed(list(enumerate(nn_architecture))):
		layer_idx_curr = layer_idx_prew + 1
		active_function_curr = layer["activation"]
		
		dA_curr	 = dA_prew
		A_prew = memory["A" + str(layer_idx_prew)]
		Z_curr = memory["Z" + str(layer_idx_curr)]
		W_curr = params_values["W" + str(layer_idx_curr)]
		b_curr = params_values["b" + str(layer_idx_curr)]

		dA_prew, dW_curr, db_curr = single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prew, active_function_curr)
		grads_values["dW" + str(layer_idx_curr)] = dW_curr
		grads_values["db" + str(layer_idx_curr)] = db_curr

	return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):
	for layer_idx , layer in enumerate(nn_architecture, 1):
		params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
		params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

	return params_values

def train(X, Y, nn_architecture, epochs, learning_rate):
	params_values = init_layers(nn_architecture, 2)
	cost_history = []
	accuracy_history = []

	for i in range(epochs):
		Y_hat, cache = full_forward_forward_propagation(X, params_values, nn_architecture)
		cost = get_cost_value(Y_hat, Y)
		cost_history.append(cost)
		accuracy = get_accuracy_value(Y_hat, Y)
		accuracy_history.append(accuracy)

		grads_values = full_backward_propagation(Y_hat, Y, cache, params_values, nn_architecture)
		params_values = update(params_values, grads_values, nn_architecture, learning_rate)

	return params_values, cost_history, accuracy_history


N_SAMPLES = 1000
TEST_SIZE = 0.1

X, y = make_moons(n_samples = N_SAMPLES, noise = 0.2, random_state = 100)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = TEST_SIZE, random_state = 42)

params_values, cost_history, accuracy_history =  train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0],1))), nn_architecture, 10000, 0.1)

Y_test_hat, _ = full_forward_forward_propagation(np.transpose(X_test), params_values, nn_architecture)

acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0],1))))
print("Test set acc :{:.2f}".format(acc_test))














