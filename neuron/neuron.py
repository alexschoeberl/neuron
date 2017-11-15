import math

class Neuron:
	def __init__(self, bias=None, inputs=None, outputs=None, alpha=1):
		"""
		Args:
			bias: The weight of the bias unit for this neuron.
			input: A dictionary of the input neurons and their corresponding weights.
			outputs: A list of the output neurons.
			alpha: The learning rate of this neuron.
		"""
		self.bias = bias
		self.weights = inputs or {}
		self.outputs = outputs or []
		self.alpha = alpha
		self.values = {}
		self.recent = None
		self.result = None
		self.errors = {}
	
	def transfer(self, values):
		"""Calculates the output of the neuron for the given input values."""
		return self.activation(self.net(values))
	
	def activation(self, value):
		"""Calculates the activation function for the given value."""
		return 1 / (1 + math.exp(-value))
	
	def derivative(self, value):
		"""Calculates the derivative of the activation function for the given value."""
		return value * (1.0 - value)

	def net(self, values):
		"""Calculates the net input from the given input values."""
		result = self.bias or 0
		for neuron in values:
			result += values[neuron] * self.weights[neuron]
		return result
	
	def update(self, delta, values):
		"""Updates the weights of the neuron considering the given delta and input values."""
		for neuron in self.weights:
			pd = delta * values[neuron]
			self.weights[neuron] -= self.alpha * pd
		if self.bias is not None:
			self.bias -= self.alpha * delta

	def delta(self, weighted_deltas, output):
		"""Calculates the delta from the given output and weighted deltas from the upper layer."""
		return sum(weighted_deltas) * self.derivative(output)

	def forward(self, source, value):
		"""Called from neurons in lower layers to feed their output forward."""
		self.values[source] = value
		if len(self.values) >= len(self.weights):
			self.result = self.transfer(self.values)
			for neuron in self.outputs:
				if isinstance(neuron, Neuron):
					neuron.forward(self, self.result)
			self.recent = self.values
			self.values = {}

	def propagate(self, source, weighted_delta):
		"""Called from neurons in upper layers to propagate their errors backwards."""
		self.errors[source] = weighted_delta
		if len(self.errors) >= len(self.outputs):
			delta = self.delta(self.errors.values(), self.result)
			for neuron in self.weights:
				if isinstance(neuron, Neuron):
					neuron.propagate(self, self.weights[neuron] * delta)
			self.update(delta, self.recent)
			self.errors.clear()


class OutputNeuron(Neuron):
	def delta(self, target, output):
		"""Calculates the delta from the given output and the target value."""
		return (output - target) * self.derivative(output)

	def propagate(self, target):
		delta = self.delta(target, self.result)
		for neuron in self.weights:
			if isinstance(neuron, Neuron):
				neuron.propagate(self, self.weights[neuron] * delta)
		self.update(delta, self.recent)
