import sys
import numpy as np
from utils import load_network, normalize_data_spec


def test(X, network):
	X = normalize_data_spec(X[:].astype(float), network.mean, network.std_dev)

	inputs = X
	for layer in network.network:
		layer.forward(inputs)
		layer.activation.forward(layer.output)
		inputs = layer.activation.output

	print(f"Result have {round(inputs[0][0] * 100, 2)}% chance to be Benign and {round(inputs[0][1] * 100, 2)}% chance to be Malignant.")


def main():
	if len(sys.argv) != 2:
		print('Please use this program as follow:\nPython3 test.py "30 elements data describing characteristics of a cell nucleus of breast mass extracted with fine-needle aspiration"')
		return 1
	network = load_network()
	values = sys.argv[1].split(',')
	X = np.array([float(value) for value in values])
	test(X, network)
	return 0


if __name__ == "__main__":
    main()