#!/usr/bin/env python3

import json
from os import path
import random

import matplotlib as mpl

from neuron.network import MultilayerPerceptron as MP


random.seed(5)
net = MP(2, 3, 1, 0.01)

with open(path.join('data', 'values.json')) as f:
	values = json.load(f)
with open(path.join('data', 'targets.json')) as f:
	targets = json.load(f)
for idx in range(len(targets)):
	targets[idx][net.output[0]] = targets[idx]['Y']
	del targets[idx]['Y']

print(net.error(values, targets))

for _ in range(1000):
	for idx in range(len(targets)):
		net.forward(values[idx])
		net.propagate(targets[idx])

print(net.error(values, targets))

mpl.use('Agg')
import matplotlib.pyplot as pypl

for x_0 in [p/5 for p in range(-10, 20)]:
	for x_1 in [p/5 for p in range(-5, 30)]:
		actual = net.forward({'X0': x_0, 'X1': x_1})
		if actual[net.output[0]] > 0.5:
			pypl.plot(x_0, x_1, 'r+')

for idx in range(len(targets)):
	if targets[idx][net.output[0]] == 0:
		pypl.plot(values[idx]['X0'], values[idx]['X1'], 'go')
	else:
		pypl.plot(values[idx]['X0'], values[idx]['X1'], 'bo')

pypl.savefig('./multilayer_perceptron.png')
