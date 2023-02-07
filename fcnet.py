# fcnet.py
# MLP-stype model for continuous outputs

# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy
from sklearn.utils import shuffle
from prettytable import PrettyTable
from scipy.interpolate import make_interp_spline

import torch
import torch.nn as nn
import plotly.express as px

# import local libraries
from network_interpret import StaticInterpret
from data_formatter import FormatDeliveries as Format
from data_formatter import Format as GeneralFormat

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLayerPerceptron(nn.Module):

	def __init__(self, input_size, output_size, return_embeddings=True):

		super().__init__()
		self.input_size = input_size
		hidden1_size = 500
		hidden2_size = 100
		hidden3_size = 20
		self.input2hidden = nn.Linear(input_size, hidden1_size)
		self.hidden2hidden = nn.Linear(hidden1_size, hidden2_size)
		self.hidden2hidden2 = nn.Linear(hidden2_size, hidden3_size)
		self.hidden2output = nn.Linear(hidden3_size, output_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.)
		self.return_embeddings = return_embeddings

	def forward(self, input):
		"""
		Forward pass through network

		Args:
			input: torch.Tensor object of network input, size [n_letters * length]

		Return: 
			output: torch.Tensor object of size output_size

		"""

		out = self.input2hidden(input)
		em1 = out
		out = self.relu(out)
		out = self.dropout(out)
		em2 = out

		out = self.hidden2hidden(out)
		em3 = out
		out = self.relu(out)
		out = self.dropout(out)
		em4 = out
		
		out = self.hidden2hidden2(out)
		em5 = out
		out = self.relu(out)
		out = self.dropout(out) 	
		em6 = out

		output = self.hidden2output(out)

		# returns embeddings, but with performance cost
		if self.return_embeddings:
			return output, em1, em2, em3, em4, em5, em6
		else:
			return output


class ActivateNet:

	def __init__(self, prediction_feature, epochs, ints_only=True):

		# specify training and test data
		file = 'data/linear_historical.csv'
		if ints_only:
			n_letters = len('0123456789. -:_') # 15 possible characters
			form = Format(file, prediction_feature, training=True)
			self.input_tensors, self.output_tensors = form.sequential_tensors(training=True)
			self.validation_inputs, self.validation_outputs = form.sequential_tensors(training=False)
		else:
			n_letters = len(string.printable)
			form = GeneralFormat(file, prediction_feature, ints_only=False)
			self.input_tensors, self.output_tensors = form.transform_to_tensors(training=True)
			self.validation_inputs, self.validation_outputs = form.transform_to_tensors(training=False)


		self.epochs = epochs

		output_size = 1
		input_size = len(self.input_tensors[0])
		self.model = MultiLayerPerceptron(input_size, output_size, return_embeddings=True).to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.biases_arr = [[], [], []] # for plotting bias
		self.embedding_dim = n_letters

	@staticmethod
	def count_parameters(model):
		"""
		Display the tunable parameters in the model of interest

		Args:
			model: torch.nn object

		Returns:
			total_params: the number of model parameters

		"""

		table = PrettyTable(['Modules', 'Parameters'])
		total_params = 0
		for name, parameter in model.named_parameters():
			if not parameter.requires_grad:
				continue
			param = parameter.numel()
			table.add_row([name, param])
			total_params += param 

		print (table)
		print (f'Total trainable parameters: {total_params}')
		return total_params


	def weighted_mseloss(self, output, target):
		"""
		We are told that the true cost of underestimation is twice
		that of overestimation, so MSEloss is customized accordingly.

		Args:
			output: torch.tensor
			target: torch.tensor

		Returns:
			loss: float

		"""
		if output < target:
			loss = torch.mean((2*(output - target))**2)
		else:
			loss = torch.mean((output - target)**2)

		return loss


	def weighted_l1loss(self, output, target):
		"""
		Assigned double the weight to underestimation with L1 cost

		Args:
			output: torch.tensor
			target: torch.tensor

		Returns:
			loss: float
		"""

		if output < target:
			loss = abs(2 * (output - target))

		else:
			loss = abs(output - target)

		return loss


	def train_minibatch(self, input_tensor, output_tensor, minibatch_size):
		"""
		Train a single minibatch

		Args:
			input_tensor: torch.Tensor object 
			output_tensor: torch.Tensor object
			optimizer: torch.optim object
			minibatch_size: int, number of examples per minibatch
			model: torch.nn

		Returns:
			output: torch.Tensor of model predictions
			loss.item(): float of loss for that minibatch

		"""
		self.model.train()
		output, _, _, _, _, _, _ = self.model(input_tensor)
		output_tensor = output_tensor.reshape(minibatch_size, 1)
		loss_function = torch.nn.L1Loss()
		loss = loss_function(output, output_tensor)

		self.optimizer.zero_grad() # prevents gradients from adding between minibatches
		loss.backward()
		nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
		self.optimizer.step()

		return output, loss.item()


	def plot_predictions(self, epoch_number):
		"""
		Plots the model predictions (y-axis) versus the true output (x-axis)

		Args:
			epoch_number: int, 

		"""
		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		model_outputs, targets, origin_ids = [], [], []

		with torch.no_grad():
			total_error = 0
			for i in range(200):
				input_tensor = self.validation_inputs[i].to(device)
				output_tensor = self.validation_outputs[i].to(device)
				model_output, _, _, _, _, _, _ = self.model(input_tensor)
				model_outputs.append(float(model_output))
				origin_ids.append(i)
				targets.append(float(output_tensor))


		plt.scatter(targets, model_outputs, c=origin_ids, cmap='hsv', s=1.5)
		fig = px.scatter(x=targets, 
						 y=model_outputs, 
						 color=origin_ids, 
						 color_continuous_scale=["red", "orange","yellow", "green","blue", "purple", "violet"])

		fig.update_traces(textposition='top center')
		fig.show()

		# _, _, rval, _, _ = scipy.stats.linregress([float(i) for i in self.validation_outputs], model_outputs)
		# print (f'R2 value: {rval**2}')
		# plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
		# plt.show()
		plt.tight_layout()
		plt.savefig('regression{0:04d}.png'.format(epoch_number), dpi=400)
		plt.close()

		return


	def plot_biases(self, index):
		"""
		Image model biases as a scatterplot

		Args:
			index: int

		Returns:
			None
		"""
		self.model.eval()
		arr = self.model.hidden2hidden.bias[:6].detach().numpy()
		self.biases_arr[0].append([arr[0], arr[1]])
		self.biases_arr[1].append([arr[2], arr[3]])
		self.biases_arr[2].append([arr[4], arr[5]])
		plt.style.use('dark_background')
		plt.plot([i[0] for i in self.biases_arr[0]], [i[1] for i in self.biases_arr[0]], '^', color='white', alpha=0.7, markersize=0.1)
		plt.plot([i[0] for i in self.biases_arr[1]], [i[1] for i in self.biases_arr[1]], '^', color='red', alpha=0.7, markersize=0.1)
		plt.plot([i[0] for i in self.biases_arr[2]], [i[1] for i in self.biases_arr[2]], '^', color='blue', alpha=0.7, markersize=0.1)
		plt.axis('on')
		plt.savefig('Biases_{0:04d}.png'.format(index), dpi=400)
		plt.close()

		return

	def heatmap_weights(self, index):
		"""
		Plot model weights of one layer as a heatmap

		Args:
			index: int

		Returns:
			None

		"""
		self.model.eval()
		arr = self.model.hidden2hidden2.weight.detach()
		arr = torch.reshape(arr, (2000, 1))
		arr = arr[:44*44]
		arr = torch.reshape(arr, (44, 44))
		arr = arr.numpy()
		plt.imshow(arr, interpolation='spline16', aspect='auto', cmap='inferno')
		plt.style.use('dark_background')
		plt.axis('off')
		plt.savefig('heatmap_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight', pad_inches=0)
		plt.close()
		return


	def heatmap_biases(self, index):
		"""
		Plot model biases as a rectangular heatmap

		Args:
			index: int

		Returns:
			None

		"""
		self.model.eval()
		arr = self.model.hidden2hidden.bias.detach().numpy()

		# convert to 10x10 array
		arr2 = []
		j = 0
		while j in range(len(arr)):
			ls = []
			while len(ls) < 10 and j < len(arr):
				ls.append(round(arr[j], 3))
				j += 1
			arr2.append(ls)
		
		arr2 = np.array(arr2)
		plt.imshow(arr2, interpolation='spline16', aspect='auto', cmap='inferno')
		for (y, x), label in np.ndenumerate(arr2):
			plt.text(x, y, '{:1.3f}'.format(label), ha='center', va='center')

		plt.axis('off')
		plt.rcParams.update({'font.size': 6})
		# plt.style.use('dark_background')
		plt.savefig('heatmap_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight', pad_inches=0)
		plt.close()

		return

	def plot_embedding(self, index=0):
		"""
		Generates a scatterplot of all pairs of embeddings versus the input
		distance of significance (ie the output for a control experiment)

		Args:
			None

		Returns:
			None (saves .png)
		"""

		for k in range(0, 1):
			self.model.eval() # switch to evaluation mode (silence dropouts etc.)
			number_of_examples = 200
			actual_arr, embedding_arr, input_arr = [], [], []
			for i in range(number_of_examples):
				input_tensor = self.validation_inputs[i].to(device)
				output_tensor = self.validation_outputs[i].to(device)
				model_output, embedding = self.model(input_tensor)[0], self.model(input_tensor)[k]
				actual_arr.append(float(output_tensor))
				embedding_arr.append(embedding)
				input_arr.append(input_tensor)

			actual_distances, embedding_distances, input_distances = [], [], []
			origin_id, target_id = [], []
			for i in range(len(embedding_arr) - 1):
				for j in range(i+1, len(embedding_arr)):
					actual_distances.append(np.abs(actual_arr[j] - actual_arr[i]))
					embedding_distance = torch.sum(torch.abs(embedding_arr[j] - embedding_arr[i])).cpu().detach().numpy()
					input_distance = torch.sum(torch.abs(input_arr[j][14*15:31*15] - input_arr[i][14*15:31*15])).cpu().detach().numpy()
					embedding_distances.append(embedding_distance)
					input_distances.append(input_distance)
					origin_id.append(i)
					target_id.append(j)

			plt.rcParams.update({'font.size': 17})
			plt.scatter(actual_distances, embedding_distances, c=origin_id, cmap='hsv', s=0.3)
			# plt.scatter(actual_distances, embedding_distances, s=0.3)
			plt.xlabel('Actual Distance')
			plt.ylabel('Embedding Distance')
			plt.tight_layout()
			plt.savefig('embedding{0:04d}'.format(k), dpi=350)
			plt.close()

			fig = px.scatter(x=actual_distances, 
							 y=embedding_distances, 
							 color=origin_id, 
							 hover_data=[origin_id, target_id], 
							 color_continuous_scale=["red", "orange","yellow", "green","blue", "purple", "violet"])
			fig.update_traces(textposition='top center')
			fig.show()

		# plt.scatter(input_distances, embedding_distances, s=0.3)
		# plt.xlabel('Input Distance')
		# plt.ylabel('Embedding Distance')
		# plt.rcParams.update({'font.size': 18})
		# plt.savefig('input_embedding'.format(index), dpi=390)
		# plt.close()
		return


	def train_model(self, minibatch_size=128):
		"""
		Train the neural network.

		Args:
			model: MultiLayerPerceptron object
			optimizer: torch.optim object
			minibatch_size: int

		Returns:
			None

		"""

		self.model.train()
		epochs = self.epochs
		count = 0
		for epoch in range(epochs):
			pairs = [[i, j] for i, j in zip(self.input_tensors, self.output_tensors)]
			random.shuffle(pairs)
			input_tensors = [i[0] for i in pairs]
			output_tensors = [i[1] for i in pairs]
			total_loss = 0

			for i in range(0, len(input_tensors) - minibatch_size, minibatch_size):
				# print (count)
				input_batch = torch.stack(input_tensors[i:i + minibatch_size]).to(device)
				output_batch = torch.stack(output_tensors[i:i + minibatch_size]).to(device)

				# skip the last batch if it is smaller than the other batches
				if len(input_batch) < minibatch_size:
					break

				output, loss = self.train_minibatch(input_batch, output_batch, minibatch_size)
				total_loss += loss
				# if count % 100 == 0:
				# 	self.plot_embedding(index=count//100)
					# interpret = StaticInterpret(self.model, self.validation_inputs, self.validation_outputs)
					# self.plot_predictions(count//25)
					# interpret.heatmap(count, method='combined')
				count += 1

			print (f'Epoch {epoch} complete: {total_loss} loss')

		return

	@torch.no_grad()
	def test_model(self, n_taken=2000):
		"""
		Test the model using a validation set

		kwargs:
			n_taken: int, number of rows used to test model

		Returns:
			None (prints test results)

		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()

		total_error = 0
		for i in range(len(self.validation_inputs)):
			input_tensor = self.validation_inputs[i].to(device)
			output_tensor = self.validation_outputs[i].to(device)
			model_output, *_  = self.model(input_tensor)
			total_error += loss(model_output, output_tensor).item()

		print (f'Test Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')

		total_error = 0
		for i in range(len(self.input_tensors[:n_taken])):
			input_tensor = self.input_tensors[i]
			output_tensor = self.output_tensors[i]
			model_output, *_ = self.model(input_tensor.to(device))
			total_error += loss(model_output, output_tensor.to(device)).item()

		print (f'Training Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')
		return


if __name__ == '__main__':
	network = ActivateNet('positive_three', 200)
	network.train_model()
	torch.save(network.model.state_dict(), 'model_weights')
	network.model.load_state_dict(torch.load('model_weights'))
	network.test_model()
	network.plot_embedding()
	network.plot_predictions(0)

	# Note that interpretations are not compatible with embedding information
	input_tensors, output_tensors = network.validation_inputs, network.validation_outputs
	model = network.model.to(device)
	interpret = StaticInterpret(model, input_tensors, output_tensors, network.embedding_dim)
	interpret.readable_interpretation(0, method='occlusion', aggregation='max')
	interpret.heatmap(0, method='occlusion')

	interpret.readable_interpretation(1, method='gradientxinput', aggregation='max')
	interpret.heatmap(1, method='gradientxinput')

	interpret.readable_interpretation(2, method='combined', aggregation='max')
	interpret.heatmap(2, method='combined')






