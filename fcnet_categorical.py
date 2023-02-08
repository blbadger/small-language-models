# fcnet_categorical.py
# MLP-style model for categorical outputs

# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import sklearn
from sklearn.utils import shuffle
import scipy

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from network_interpret import CategoricalStaticInterpret as interpret
from data_formatter import Format as GeneralFormat

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

class MultiLayerPerceptron(nn.Module):

	def __init__(self, input_size, output_size):

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

	def forward(self, input):
		"""
		Forward pass through network

		Args:
			input: torch.Tensor object of network input, size [n_letters * length]

		Return: 
			output: torch.Tensor object of size output_size

		"""

		out = self.input2hidden(input)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.hidden2hidden(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.hidden2hidden2(out)
		out = self.relu(out)
		out = self.dropout(out)

		output = self.hidden2output(out)
		return output


class Format():

	def __init__(self, file, training=True, n_per_field=False, deliveries=False):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower() == 'nan' else x)
		df = df[:100000]
		length = len(df[:])
		self.input_fields = ['PassengerId',
							 'Pclass',
							 'Name',
							 'Sex',
							 'Age',
							 'SibSp',
							 'Parch',
							 'Ticket',
							 'Fare',
							 'Cabin',
							 'Embarked']
		if n_per_field:
			self.taken_ls = [4 for i in self.input_fields]
		else:
			self.taken_ls = [3, 1, 5, 2, 3, 2, 4, 5, 4, 4, 1]

		if training:
			df = shuffle(df)
			df.reset_index(inplace=True)

			# 80/20 training/test split
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[self.input_fields]
			self.training_outputs = [i for i in training['Survived'][:]]

			df2 = pd.read_csv('titanic/test.csv')
			validation_size = len(df2)
			validation = df2
			self.validation_inputs = validation[self.input_fields]
			df3 = pd.read_csv('titanic/gender_submission.csv')
			self.validation_outputs = [i for i in df3['Survived'][:]] 
			self.validation_inputs = self.validation_inputs.reset_index()

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify

	def unstructured_stringify(self, index, training=True, pad=True, length=75):
		"""
		Compose array of string versions of relevant information in self.df 
		Does not maintain a consistant structure to inputs regardless of missing 
		values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""

		string_arr = []
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]

		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])
			string_arr.append(entry)


		string = ''.join(string_arr)
		if pad:
			if len(string) < length:
				string += '_' * (length - len(string))
			if len(string) > length:
				string = string[:length]
		return string

	def stringify_input(self, index, training=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""
		taken_ls = self.taken_ls

		string_arr = []
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]

		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])[:taken_ls[i]]
			while len(entry) < taken_ls[i]:
				entry += '_'
			string_arr.append(entry)

		string = ''.join(string_arr)
		return string


	@classmethod
	def string_to_tensor(self, input_string, ints_only=False):
		"""
		Convert a string into a tensor. For numerical inputs.

		Args:
			string: str, input as a string

		Returns:
			tensor
		"""

		if ints_only:
			places_dict = {s:i for i, s in enumerate('0123456789. -:_')}

		else:
			chars = string.printable
			places_dict = {s:i for i, s in enumerate(chars)}

		self.embedding_dim = len(places_dict)
		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(input_string), 1, len(places_dict)) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(input_string):
			tensor[i][0][places_dict[letter]] = 1.

		tensor = tensor.flatten()
		return tensor 


	def sequential_tensors(self, training=True):
		"""
		
		"""

		input_tensors = []
		output_tensors = []
		if training:
			inputs = self.training_inputs
			outputs = self.training_outputs
		else:
			inputs = self.validation_inputs
			outputs = self.validation_outputs

		for i in range(len(inputs)):
			input_string = self.stringify_input(i, training=training)
			input_tensor = self.string_to_tensor(input_string)
			input_tensors.append(input_tensor)

			# convert output float to tensor directly
			output_tensors.append(torch.tensor([outputs[i]]))

		return input_tensors, output_tensors


class ActivateNet:

	def __init__(self, epochs, deliveries=False):

		if deliveries:
			# specific dataset initialization and encoding
			file = 'titanic/train.csv'
			df = pd.read_csv(file)
			input_tensors = Format(file, 'Survived')
			self.input_tensors, self.output_tensors = input_tensors.sequential_tensors(training=True) 
			self.validation_inputs, self.validation_outputs = input_tensors.sequential_tensors(training=False)
			self.n_letters = input_tensors.embedding_dim

		else:
			# general dataset initialization and encoding
			file = 'titanic/train.csv'
			form = GeneralFormat(file, 'Survived', ints_only=False)
			self.input_tensors, self.output_tensors = form.transform_to_tensors(training=True)
			self.validation_inputs, self.validation_outputs = form.transform_to_tensors(training=False)
			self.taken_ls = [form.n_taken for i in range(len(form.training_inputs.loc[0]))]
			self.n_letters = len(form.places_dict)

		print (len(self.input_tensors), len(self.validation_inputs))
		self.epochs = epochs
		output_size = 2
		input_size = len(self.input_tensors[0])
		self.model = MultiLayerPerceptron(input_size, output_size)
		self.model.to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.biases_arr = [[], []]


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

		output = self.model(input_tensor.to(device))
		output_tensor = output_tensor.reshape(minibatch_size).to(device)
		output_tensor = output_tensor.long()
		# loss_function = torch.nn.L1Loss()
		loss_function = torch.nn.CrossEntropyLoss()
		loss = loss_function(output, output_tensor)

		self.optimizer.zero_grad() # prevents gradients from adding between minibatches
		loss.backward()
		self.optimizer.step()

		return output, loss.item()


	def plot_predictions(self, epoch_number):
		"""

		"""
		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()
		model_outputs = []

		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i].to(device)
				output_tensor = self.validation_outputs[i].to(device)
				model_output = self.model(input_tensor)
				model_outputs.append(float(model_output))

		plt.scatter([float(i) for i in self.validation_outputs], model_outputs, s=1.5)
		plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
		# plt.show()
		plt.tight_layout()
		plt.savefig('regression{0:04d}.png'.format(epoch_number), dpi=400)
		plt.close()
		return

	def plot_biases(self, index):
		"""

		"""
		x, y = self.model.hidden2hidden2[:2].detach().numpy()
		self.biases_arr[0].append(x)
		self.biases_arr[1].append(y)
		plt.style.use('dark_background')
		plt.plot(x_arr, y_arr, '^', color='white', alpha=2, markersize=0.1)
		plt.axis('on')
		plt.savefig('Biases_{0:04d}.png'.format(index), dpi=400)
		plt.close()
		return

	def quiver_gradients(self, index, input_tensor, output_tensor, minibatch_size=64):
		"""
		plots

		"""
		self.model.eval()
		x, y = self.model.hidden2hidden.bias[:2].detach().numpy()
		print (x, y)
		plt.style.use('dark_background')

		x_arr = np.arange(x - 0.01, x + 0.01, 0.001)
		y_arr = np.arange(y - 0.01, y + 0.01, 0.001)

		XX, YY = np.meshgrid(x_arr, y_arr)
		dx, dy = np.meshgrid(x_arr, y_arr) # copy that will be overwritten
		for i in range(len(x_arr)):
			for j in range(len(y_arr)):
				with torch.no_grad():
					self.model.hidden2hidden.bias[0] = torch.nn.Parameter(torch.Tensor([x_arr[i]]))
					self.model.hidden2hidden.bias[1] = torch.nn.Parameter(torch.Tensor([y_arr[j]]))
				output = self.model(input_tensor)
				output_tensor = output_tensor.reshape(minibatch_size, 1)
				loss_function = torch.nn.L1Loss()
				loss = loss_function(output, output_tensor)
				self.optimizer.zero_grad()
				loss.backward()
				dx[j][i], dy[j][i] = self.model.hidden2hidden.bias.grad[:2]

		matplotlib.rcParams.update({'font.size': 8})
		color_array = 2*(np.abs(dx) + np.abs(dy))
		plt.quiver(XX, YY, dx, dy, color_array)
		plt.plot(x, y, 'o', markersize=1)
		plt.savefig('quiver_{0:04d}.png'.format(index), dpi=400)
		plt.close()
		with torch.no_grad():
			self.model.hidden2hidden.bias[:2] = torch.Tensor([x, y])
		return

	def quiver_gradients_double(self, index, input_tensor, output_tensor, minibatch_size=64):
		"""

		"""
		self.model.eval()
		x, y = self.model.hidden2hidden.bias[:2].detach().numpy()
		x_arr = np.arange(x - 0.1, x + 0.1, 0.02)
		y_arr = np.arange(y - 0.1, y + 0.1, 0.01)

		XX, YY = np.meshgrid(x_arr, y_arr)
		dx, dy = np.meshgrid(x_arr, y_arr) # copy that will be overwritten
		for i in range(len(x_arr)):
			for j in range(len(y_arr)):
				with torch.no_grad():
					self.model.hidden2hidden.bias[0] = torch.nn.Parameter(torch.Tensor([x_arr[i]]))
					self.model.hidden2hidden.bias[1] = torch.nn.Parameter(torch.Tensor([y_arr[j]]))
				output = self.model(input_tensor)
				output_tensor = output_tensor.reshape(minibatch_size, 1)
				loss_function = torch.nn.L1Loss()
				loss = loss_function(output, output_tensor)
				self.optimizer.zero_grad()
				loss.backward()
				dx[j][i], dy[j][i] = self.model.hidden2hidden.bias.grad[:2]

		x2, y2 = self.model.hidden2hidden2.bias[:2].detach().numpy()

		x_arr2 = np.arange(x2 - 0.1, x2 + 0.1, 0.02)
		y_arr2 = np.arange(y2 - 0.1, y2 + 0.1, 0.01)

		XX2, YY2 = np.meshgrid(x_arr2, y_arr2)
		dx2, dy2 = np.meshgrid(x_arr2, y_arr2) # copy that will be overwritten
		for i in range(len(x_arr2)):
			for j in range(len(y_arr2)):
				with torch.no_grad():
					self.model.hidden2hidden2.bias[0] = torch.nn.Parameter(torch.Tensor([x_arr2[i]]))
					self.model.hidden2hidden2.bias[1] = torch.nn.Parameter(torch.Tensor([y_arr2[j]]))
				output = self.model(input_tensor)
				output_tensor = output_tensor.reshape(minibatch_size, 1)
				loss_function = torch.nn.L1Loss()
				loss = loss_function(output, output_tensor)
				self.optimizer.zero_grad()
				loss.backward()
				dx2[j][i], dy2[j][i] = self.model.hidden2hidden2.bias.grad[:2]

		
		color_array = 2*(np.abs(dx) + np.abs(dy))
		matplotlib.rcParams.update({'font.size': 7})
		plt.style.use('dark_background')
		plt.subplot(1, 2, 1)
		plt.quiver(XX, YY, dx, dy, color_array)
		plt.title('Hidden Layer 1')

		plt.subplot(1, 2, 2)
		color_array2 = 2*(np.abs(dx2) + np.abs(dy2))
		plt.quiver(XX2, YY2, dx2, dy2, color_array2)
		plt.title('Hidden Layer 2')
		plt.savefig('quiver_{0:04d}.png'.format(index), dpi=400)
		plt.close()

		with torch.no_grad():
			self.model.hidden2hidden.bias[:2] = torch.Tensor([x, y])
			self.model.hidden2hidden2.bias[:2] = torch.Tensor([x2, y2])
		return


	def train_model(self, minibatch_size=128):
		"""
		Train the mlp model

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
			print (f'Epoch {epoch}')
			pairs = [[i, j] for i, j in zip(self.input_tensors, self.output_tensors)]
			random.shuffle(pairs)
			input_tensors = [i[0] for i in pairs]
			output_tensors = [i[1] for i in pairs]
			total_loss = 0
			correct, total = 0, 0
			for i in range(0, len(input_tensors) - minibatch_size, minibatch_size):
				# stack tensors to make shape (minibatch_size, input_size)
				input_batch = torch.stack(input_tensors[i:i + minibatch_size])
				output_batch = torch.stack(output_tensors[i:i + minibatch_size])

				# skip the last batch if too small
				if len(input_batch) < minibatch_size:
					break
				output, loss = self.train_minibatch(input_batch, output_batch, minibatch_size)
				total_loss += loss
				output_batch = output_batch.reshape(minibatch_size).to(device)
				correct += torch.sum(torch.argmax(output, dim=1) == output_batch)
				total += minibatch_size
				# if i % 1 == 0:
				# 	print (f'Epoch {epoch} complete: {total_loss} loss')
				# 	self.quiver_gradients(count, input_batch, output_batch)
				# 	count += 1
			print (f'Train Accuracy: {correct / total}')
			print (f'Loss: {total_loss}')
		return


	def train_online(self, file, minibatch_size=1):
		"""
		On-line training with random samples

		Args:
			model: Transformer object
			optimizer: torch.optim object of choice

		kwags:
			minibatch_size: int, number of samples per gradient update

		Return:
			none (modifies model in-place)

		"""

		self.model.train()
		current_loss = 0
		training_data = Format(file, training=True)

		# training iteration and epoch number specs
		n_epochs = 10

		start = time.time()
		for i in range(n_epochs):
			random.shuffle(input_samples)
			for i in range(0, len(self.input_samples), minibatch_size):
				if len(input_samples) - i < minibatch_size:
					break

				input_tensor = torch.cat([input_samples[i+j] for j in range(minibatch_size)])
				output_tensor = torch.cat([output_samples[i+j] for j in range(minibatch_size)])

				# define the output and backpropegate loss
				output, loss = train_random_input(output_tensor, input_tensor)

				# sum to make total loss
				current_loss += loss 

				if i % n_per_epoch == 0 and i > 0:
					etime = time.time() - start
					ave_error = round(current_loss / n_per_epoch, 2)
					print (f'Epoch {i//n_per_epoch} complete \n Average error: {ave_error} \n Elapsed time: {round(etime, 2)}s \n' + '~'*30)
					current_loss = 0 
		return


	def test_model(self):
		"""

		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()

		model_outputs, true_outputs = [], []
		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i].to(device)
				output_tensor = self.validation_outputs[i].to(device)
				model_output = self.model(input_tensor)
				total_error += loss(model_output, output_tensor).item()
				model_outputs.append(float(model_output))
				true_outputs.append(float(output_tensor))

		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(model_outputs, true_outputs)
		print (f'Mean Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')
		print (f'R2 value: {r_value**2}')
		return

	def test_model_categories(self):
		"""

		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)

		model_outputs, true_outputs = [], []
		minibatch_size = 1
		with torch.no_grad():
			correct, count = 0, 0
			for i in range(0, len(self.validation_inputs), minibatch_size):
				input_batch = torch.stack(self.validation_inputs[i:i + minibatch_size])
				output_batch = torch.stack(self.validation_outputs[i:i + minibatch_size])
				input_tensor = input_batch.to(device)
				output_tensor = output_batch.reshape(minibatch_size).to(device)
				model_output = self.model(input_tensor)
				correct += torch.sum(torch.argmax(model_output, dim=1) == output_tensor)
				count += minibatch_size

		print (correct, count)
		print (f'Test Accuracy: {correct / count}')
		return


	def predict(self, model, test_inputs):
		"""
		Make predictions with a model.

		Args:
			model: Transformer() object
			test_inputs: torch.tensor inputs of prediction desired

		Returns:
			prediction_array: arr[int] of model predictions

		"""
		model.eval()
		prediction_array = []

		with torch.no_grad():
			for i in range(len(test_inputs['index'])):
				prediction_array.append(model_output)

		return prediction_array

epochs = 15

network = ActivateNet(epochs)
network.train_model()
network.test_model_categories()
interpretation = interpret(network.model, network.validation_inputs, network.validation_outputs)
# interpretation.heatmap(0)
# interpretation.readable_interpretation(0)





