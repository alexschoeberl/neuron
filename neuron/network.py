from random import random

from .neuron import Neuron, OutputNeuron

class MultilayerPerceptron:
	def __init__(self, input, hidden, output, alpha):
		"""
		Args:
			input: Number of input neurons.
			hidden: Number of hidden neurons.
			output: Number of output neurons.
			alpha: Learning rate of the network.
		"""
		self.input_size = input
		self.hidden_size = hidden
		self.output_size = output
		self.hidden = []
		self.output = []

		for u in range(self.hidden_size):
			inputs = {}
			for v in range(self.input_size):
				inputs['X{}'.format(v)] = random()
			neuron = Neuron(bias=random(), inputs=inputs, alpha=alpha)
			self.hidden.append(neuron)
		
		for u in range(output):
			inputs = {}
			for neuron in self.hidden:
				inputs[neuron] = random()
			neuron = OutputNeuron(bias=random(), inputs=inputs, alpha=alpha)
			self.output.append(neuron)
		
		for neuron in self.hidden:
			neuron.outputs.extend(self.output)
	
	def forward(self, values):
		"""Forward feeds the given input values through the network."""
		for hidden_neuron in self.hidden:
			for input_neuron in values:
				hidden_neuron.forward(input_neuron, values[input_neuron])
		return {output_neuron: output_neuron.result for output_neuron in self.output}
	
	def propagate(self, targets):
		"""Backward propagates the target values through the network, considering the last output."""
		for output_neuron in self.output:
			output_neuron.propagate(targets[output_neuron])
	
	def error(self, values, targets):
		"""Calculates the error value of the network."""
		result = 0
		for idx in range(len(values)):
			actual = self.forward(values[idx])
			for output_neuron in actual:
				result += (targets[idx][output_neuron] - actual[output_neuron])**2
		return result