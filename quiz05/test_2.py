import numpy as np, sys
#np.random.seed(0)

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Input datasets
inputs = np.array([[1,0]])
# expected_output = np.array([[1]])
expected_output = 1

epochs = int(sys.argv[1])
lr = float(sys.argv[2])
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

#Random weights and bias initialization
hidden_weights = np.array([[0.2,0.12],[0.35,0.2]])
hidden_bias = np.array([[0.22,0.05]])
output_weights = np.array([[0.1],[0.23]])
output_bias = np.array([[0.02]])

print("Initial hidden weights: ")
print(hidden_weights)
print("Initial hidden biases: ")
print(hidden_bias)
print("Initial output weights: ")
print(output_weights)
print("Initial output biases: ")
print(output_bias)
print("_"*50)


#Training algorithm
for _ in range(epochs):
	#Forward Propagation
	hidden_layer_activation = np.dot(inputs,hidden_weights)
	hidden_layer_activation += hidden_bias
	hidden_layer_output = sigmoid(hidden_layer_activation)
	print("Hidden_layer_output: ")
	print(hidden_layer_output)
	print()
  
	output_layer_activation = np.dot(hidden_layer_output,output_weights)
	output_layer_activation += output_bias
	predicted_output = sigmoid(output_layer_activation)[0,0]
	print("Predicted_output: ")
	print(predicted_output)
	print()

	# Calcs error
	error = expected_output - predicted_output
	# error = 0.5*(expected_output - predicted_output)**2
	print(f'Output error:\n{error}\n')

	#Backpropagation
	# Output to hidden layer derivative
	d_predicted_output = error * sigmoid_derivative(predicted_output)
	print(f'd_predicted_output:\n{d_predicted_output}\n')
	# error_hidden_layer = d_predicted_output.dot(output_weights.T)
	error_hidden_layer = d_predicted_output * output_weights
	print("Error hidden Layer: ")
	print(error_hidden_layer)
	print()

	# Derivative of hidden layer to inputs
	d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
	print("Derivative hidden layer output: ")
	print(d_hidden_layer)
	print()

	#Updating Weights and Biases
	output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
 
	output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
	
	hidden_weights += (d_hidden_layer @ inputs.T) * lr
	
	hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

print("Final hidden bias: ")
print(hidden_bias)
print()

print("Final output bias: ")
print(output_bias)
print()

print("_"*50)
print("Final hidden weights: ")
print(f'w2: {hidden_weights[0,0]}\nw3: {hidden_weights[0,1]}\nw4: {hidden_weights[1,0]}\nw5: {hidden_weights[1,1]}')
print()

print("Final output weights: ")
print(f'w7: {output_weights[1, 0]}\nw8: {output_weights[0, 0]}')
print()


