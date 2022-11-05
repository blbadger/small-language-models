# network_interpret.py
import torch
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import string

plt.rcParams.update({'font.size': 15})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)


class StaticInterpret:

	def __init__(self, model, input_tensors, output_tensors):
		self.model = model 
		self.model.eval()
		self.output_tensors = output_tensors
		self.input_tensors = input_tensors
		self.fields_ls = ['Store Number', 
						'Market', 
						'Order Made',
						'Cost',
						'Total Deliverers', 
						'Busy Deliverers', 
						'Total Orders',
						'Estimated Transit Time',
						'Linear Estimation']
		self.embedding_dim = 15
		# self.taken_ls = [4, 1, 15, 5, 4, 4, 4, 4, 4]
		self.taken_ls = [4, 1, 8, 5, 3, 3, 3, 4, 4]


	def occlusion(self, input_tensor, occlusion_size=2):
		"""
		Generates a perturbation-type attribution using occlusion.

		Args:
			input_tensor: torch.Tensor
		kwargs:
			occlusion_size: int, number of sequential elements zeroed
							out at once.

		Returns:
			occlusion_arr: array[float] of scores per input index

		"""
		input_tensor = input_tensor.flatten().to(device)
		output_tensor = self.model(input_tensor)
		zeros_tensor = torch.zeros(input_tensor[:occlusion_size*self.embedding_dim].shape)
		total_index = 0
		occlusion_arr = [0 for i in range(sum(self.taken_ls))]
		i = 0
		while i in range(len(input_tensor)-(occlusion_size-1)*self.embedding_dim):
			# set all elements of a particular field to 0
			input_copy = torch.clone(input_tensor)
			input_copy[i:i + occlusion_size*self.embedding_dim] = zeros_tensor

			output_missing = self.model(input_copy)
			occlusion_val = abs(float(output_missing) - float(output_tensor))

			for j in range(occlusion_size):
				occlusion_arr[i // self.embedding_dim + j] += occlusion_val

			i += self.embedding_dim

		# max-normalize occlusions
		if max(occlusion_arr) != 0:
			correction_factor = 1 / (max(occlusion_arr))
			occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return occlusion_arr


	def gradientxinput(self, input_tensor):
		"""
		 Compute a gradientxinput attribution score

		 Args:
		 	input: torch.Tensor() object of input
		 	model: Transformer() class object, trained neural network

		 Returns:
		 	gradientxinput: arr[float] of input attributions

		"""

		# enforce the input tensor to be assigned a gradient
		input_tensor = input_tensor.flatten().to(device)
		input_tensor.requires_grad = True
		output = self.model(input_tensor)

		# only scalars may be assigned a gradient
		output_shape = 1
		output = output.reshape(1, output_shape).sum()

		# backpropegate output gradient to input
		output.backward(retain_graph=True)

		# compute gradient x input
		final = torch.abs(input_tensor.grad) * input_tensor

		# separate out individual characters 
		saliency_arr = []
		s = 0
		for i in range(len(final)):
			if i % self.embedding_dim == 0 and i > 0: 
				saliency_arr.append(s)
				s = 0
			s += float(final[i])

		# append final element
		saliency_arr.append(s)

		# max norm
		maximum = 0
		for i in range(len(saliency_arr)):
			maximum = max(saliency_arr[i], maximum)

		# prevent a divide by zero error
		if maximum != 0:
			for i in range(len(saliency_arr)):
				saliency_arr[i] /= maximum

		return saliency_arr


	def heatmap(self, count, n_observed=50, method='combined', normalized=True):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""
		attributions_array = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [(i+j)/2 for i, j in zip(occlusion, gradxinput)]

			elif method == 'gradientxinput':
				attribution = self.gradientxinput(input_tensor)

			else:
				attribution = self.occlusion(input_tensor)

			attributions_array.append(attribution)

		# max-normalize occlusions
		if normalized and max(attributions_array) != 0:
			for i, row in enumerate(attributions_array):
				maximum_attribute = max(row)
				correction_factor = 1 / (maximum_attribute + 1e-9)
				attributions_array[i] = [j*correction_factor for j in row]

		# plt.style.use('dark_background')
		plt.imshow(attributions_array, vmin=0, vmax=1)
		plt.colorbar()
		plt.xlabel('Position')
		plt.ylabel('Sample')
		plt.tight_layout
		plt.savefig('attributions_{0:04d}.png'.format(count), dpi=400)
		plt.close()
		return

	def readable_interpretation(self, count, n_observed=100, method='combined', normalized=True, aggregation='average'):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'
			aggregation: str, one of 'max', 'average'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""
		attributions_arr = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [(i+j)/2 for i, j in zip(occlusion, gradxinput)]

			elif method == 'gradientxinput':
				attribution = self.gradientxinput(input_tensor)

			else:
				attribution = self.occlusion(input_tensor)

			attributions_arr.append(attribution)

		average_arr = []
		for i in range(len(attributions_arr[0])):
			average = sum([attribute[i] for attribute in attributions_arr])
			average_arr.append(average)

		# max-aggregated attributions per field
		if aggregation == 'max':
			final_arr = []
			index = 0
			for c, field in zip(self.taken_ls, self.fields_ls):
				max_val = 0
				for k in range(index, index + c):
					max_val = max(max_val, average_arr[k])
				final_arr.append([field, max_val])
				index += c

		elif aggregation == 'average':
			final_arr = []
			index = 0
			for c, field in zip(self.taken_ls, self.fields_ls):
				sum_val = 0
				for k in range(index, index + c):
					sum_val += average_arr[k]
				final_arr.append([field, sum_val/c])
				index += c

		# max-normalize occlusions
		maximum_attribute = max([i[1] for i in final_arr])
		if normalized and max(final_arr) != 0:
			correction_factor = 1 / maximum_attribute
			final_arr = [i[1]*correction_factor for i in final_arr]

		# plt.style.use('dark_background')
		my_cmap = plt.cm.get_cmap('viridis')
		colors = my_cmap(final_arr)
		plt.barh(self.fields_ls, final_arr, color=colors, edgecolor='black')
		plt.yticks(np.arange(0, len(self.fields_ls)),
				[i for i in self.fields_ls],
				rotation='horizontal')

		plt.tight_layout()
		plt.xlabel('Importance')
		plt.savefig('readable_{}'.format(count), dpi=400, bbox_inches='tight')
		plt.close()
		return


	def graph_attributions(self):
		"""
		Plot the attributions of the model for 5 inputs

		Args:
			None

		Returns:
			None (dislays matplotlib.pyplot object)
		
		"""

		# view horizontal bar charts of occlusion attributions for five input examples
		for i in range(5):
			occlusion = self.occlusion(self.input_tensors[i])
			gradientxinput = self.gradientxinput(self.input_tensors[i])
			indicies_arr = [i for i in range(len(occlusion))]
			plt.style.use('dark_background')
			plt.barh(indicies, occlusion)
			plt.yticks(np.arange(0, len(self.fields_ls)), [i for i in self.fields_ls])
			plt.tight_layout()
			plt.show()
			plt.close()

		return


class CategoricalStaticInterpret:

	def __init__(self, model, input_tensors, output_tensors):
		self.model = model.to(device) 
		self.model.eval()
		self.output_tensors = output_tensors
		self.input_tensors = input_tensors
		self.fields_ls = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
		self.taken_ls = [3, 1, 5, 2, 3, 2, 4, 5, 4, 4, 1] 
		self.embedding_dim = len(string.printable)

	def occlusion(self, input_tensor, occlusion_size=2):
		"""
		Generates a perturbation-type attribution using occlusion.

		Args:
			input_tensor: torch.Tensor
		kwargs:
			occlusion_size: int, number of sequential elements zeroed
							out at once.

		Returns:
			occlusion_arr: array[float] of scores per input index

		"""
		input_tensor = input_tensor.flatten().to(device)
		output_tensor = self.model(input_tensor)
		zeros_tensor = torch.zeros(input_tensor[:occlusion_size*self.embedding_dim].shape)
		total_index = 0
		occlusion_arr = [0 for i in range(sum(self.taken_ls))]
		i = 0
		while i in range(len(input_tensor)-(occlusion_size-1)*self.embedding_dim):
			# set all elements of a particular field to 0
			input_copy = torch.clone(input_tensor)
			input_copy[i:i + occlusion_size*self.embedding_dim] = zeros_tensor

			output_missing = self.model(input_copy)
			occlusion_val = torch.sum(torch.abs(output_missing - output_tensor))

			for j in range(occlusion_size):
				occlusion_arr[i // self.embedding_dim + j] += occlusion_val

			i += self.embedding_dim

		# max-normalize occlusions
		if max(occlusion_arr) != 0:
			correction_factor = 1 / (max(occlusion_arr))
			occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return occlusion_arr


	def gradientxinput(self, input_tensor):
		"""
		 Compute a gradientxinput attribution score

		 Args:
		 	input: torch.Tensor() object of input
		 	model: Transformer() class object, trained neural network

		 Returns:
		 	gradientxinput: arr[float] of input attributions

		"""

		# enforce the input tensor to be assigned a gradient
		input_tensor = input_tensor.to(device)
		input_tensor.requires_grad = True
		output = self.model.forward(input_tensor)

		# only scalars may be assigned a gradient
		output_shape = 2
		output = output.reshape(1, output_shape).sum()

		# backpropegate output gradient to input
		output.backward(retain_graph=True)

		# compute gradient x input
		final = torch.abs(input_tensor.grad) * input_tensor

		# separate out individual characters 
		saliency_arr = []
		s = 0
		for i in range(len(final)):
			if i % self.embedding_dim == 0 and i > 0: 
				saliency_arr.append(s)
				s = 0
			s += float(final[i])

		# append final element
		saliency_arr.append(s)

		# max norm
		maximum = 0
		for i in range(len(saliency_arr)):
			maximum = max(saliency_arr[i], maximum)

		# prevent a divide by zero error
		if maximum != 0:
			for i in range(len(saliency_arr)):
				saliency_arr[i] /= maximum

		return saliency_arr


	def heatmap(self, count, n_observed=50, method='combined', normalized=True):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""
		attributions_array = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [float((i + j)/2) for i, j in zip(occlusion, gradxinput)]


			elif method == 'gradientxinput':
				attribution = self.gradientxinput(input_tensor)

			else:
				attribution = self.occlusion(input_tensor)

			attributions_array.append(attribution)

		# max-normalize occlusions
		if normalized and max(attributions_array) != 0:
			for i, row in enumerate(attributions_array):
				maximum_attribute = max(row)
				correction_factor = 1 / maximum_attribute
				attributions_array[i] = [j*correction_factor for j in row]

		plt.imshow(attributions_array, vmin=0, vmax=1)
		plt.colorbar()
		plt.xlabel('Position')
		plt.ylabel('Sample')
		plt.savefig('attributions_{0:04d}.png'.format(count), dpi=400)
		plt.close()
		return

	def readable_interpretation(self, count, n_observed=100, method='combined', normalized=True, aggregation='average'):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""
		attributions_arr = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [(i+j)/2 for i, j in zip(occlusion, gradxinput)]

			elif method == 'gradientxinput':
				attribution = self.gradientxinput(input_tensor)

			else:
				attribution = self.occlusion(input_tensor)

			attributions_arr.append(attribution)

		average_arr = []
		for i in range(len(attributions_arr[0])):
			average = sum([attribute[i] for attribute in attributions_arr])
			average_arr.append(average)

		# max-aggregated attributions per field
		if aggregation == 'max':
			final_arr = []
			index = 0
			for c, field in zip(self.taken_ls, self.fields_ls):
				max_val = 0
				for k in range(index, index + c):
					max_val = max(max_val, average_arr[k])
				final_arr.append([field, max_val])
				index += c

		elif aggregation == 'average':
			final_arr = []
			index = 0
			for c, field in zip(self.taken_ls, self.fields_ls):
				sum_val = 0
				for k in range(index, index + c):
					sum_val += average_arr[k]
				final_arr.append([field, sum_val/c])
				index += c

		# max-normalize occlusions
		maximum_attribute = max([i[1] for i in final_arr])
		if normalized and max(final_arr) != 0:
			correction_factor = 1 / maximum_attribute
			final_arr = [float(i[1]*correction_factor) for i in final_arr]

		my_cmap = plt.cm.get_cmap('viridis')
		colors = my_cmap(final_arr)
		plt.barh(self.fields_ls, final_arr, color=colors, edgecolor='black')
		plt.yticks(np.arange(0, len(self.fields_ls)),
				[i for i in self.fields_ls],
				rotation='horizontal')

		plt.tight_layout()
		plt.xlabel('Importance')
		plt.savefig('readable_{}'.format(count), dpi=400, bbox_inches='tight')
		plt.close()
		return


	def graph_attributions(self):
		"""
		Plot the attributions of the model for 5 inputs

		Args:
			None

		Returns:
			None (dislays matplotlib.pyplot object)
		
		"""

		# view horizontal bar charts of occlusion attributions for five input examples
		for i in range(5):
			occlusion = self.occlusion(self.input_tensors[i])
			gradientxinput = self.gradientxinput(self.input_tensors[i])
			indicies_arr = [i for i in range(len(occlusion))]
			plt.style.use('dark_background')
			plt.barh(indicies, occlusion)
			plt.yticks(np.arange(0, len(self.fields_ls)), [i for i in self.fields_ls])
			plt.tight_layout()
			plt.show()
			plt.close()

		return


class TransformerCatInterpret:

	def __init__(self, model, input_tensors, output_tensors):
		self.model = model.to(device) 
		self.model.eval()
		self.output_tensors = output_tensors
		self.input_tensors = input_tensors
		self.fields_ls = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
		self.taken_ls = [3, 1, 5, 2, 3, 2, 4, 5, 4, 4, 1] 
		self.embedding_dim = len(string.printable)

	def occlusion(self, input_tensor, occlusion_size=2):
		"""
		Generates a perturbation-type attribution using occlusion.

		Args:
			input_tensor: torch.Tensor
		kwargs:
			occlusion_size: int, number of sequential elements zeroed
							out at once.

		Returns:
			occlusion_arr: array[float] of scores per input index

		"""
		input_tensor = input_tensor.reshape(1, sum(self.taken_ls), self.embedding_dim).to(device) # minbatch_size = 1
		output_tensor = self.model(input_tensor)
		zeros_tensor = torch.zeros(input_tensor[0][0].shape)
		total_index = 0
		occlusion_arr = [0 for i in range(sum(self.taken_ls))]
		i = 0
		while i in range(len(input_tensor[0])-(occlusion_size-1)):
			# set all elements of a particular field to 0
			input_copy = torch.clone(input_tensor)
			input_copy[0][i] = zeros_tensor

			output_missing = self.model(input_copy)
			occlusion_val = torch.sum(torch.abs(output_missing - output_tensor))

			for j in range(occlusion_size):
				occlusion_arr[i + j] += occlusion_val
			i += 1

		# max-normalize occlusions
		if max(occlusion_arr) != 0:
			correction_factor = 1 / (max(occlusion_arr))
			occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return occlusion_arr


	def gradientxinput(self, input_tensor):
		"""
		 Compute a gradientxinput attribution score

		 Args:
		 	input: torch.Tensor() object of input
		 	model: Transformer() class object, trained neural network

		 Returns:
		 	gradientxinput: arr[float] of input attributions

		"""

		# enforce the input tensor to be assigned a gradient
		input_tensor = input_tensor.reshape(1, sum(self.taken_ls), self.embedding_dim).to(device)
		input_tensor.requires_grad = True
		output = self.model.forward(input_tensor)

		# only scalars may be assigned a gradient
		output_shape = 2
		output = output.reshape(1, output_shape).sum()

		# backpropegate output gradient to input
		output.backward()

		# compute gradient x input
		final = torch.abs(input_tensor.grad) * input_tensor

		# separate out individual characters 
		saliency_arr = [float(sum(i)) for i in final[0]]
		# max norm
		maximum = 0
		for i in range(len(saliency_arr)):
			maximum = max(saliency_arr[i], maximum)


		# prevent a divide by zero error
		if maximum != 0:
			for i in range(len(saliency_arr)):
				saliency_arr[i] /= maximum

		return saliency_arr


	def heatmap(self, count, n_observed=50, method='gradientxinput', normalized=True):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""
		attributions_array = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [float((i + j)/2) for i, j in zip(occlusion, gradxinput)]

			elif method == 'gradientxinput':
				attribution = [float(i) for i in self.gradientxinput(input_tensor)]

			else:
				attribution = [float(i) for i in self.occlusion(input_tensor)]

			attributions_array.append(attribution)

		# max-normalize occlusions
		if normalized and max(attributions_array) != 0:
			for i, row in enumerate(attributions_array):
				maximum_attribute = max(row)
				if maximum_attribute > 0:
					correction_factor = 1 / maximum_attribute
				else:
					correction_factor = 0
				attributions_array[i] = [j*correction_factor for j in row]

		plt.imshow(attributions_array, vmin=0, vmax=1)
		plt.colorbar()
		plt.xlabel('Position')
		plt.ylabel('Sample')
		plt.savefig('attributions_{0:04d}.png'.format(count), dpi=400)
		plt.close()
		return

	def readable_interpretation(self, count, n_observed=100, method='gradientxinput', normalized=True, aggregation='average'):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""
		attributions_arr = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [(i+j)/2 for i, j in zip(occlusion, gradxinput)]

			elif method == 'gradientxinput':
				attribution = self.gradientxinput(input_tensor)

			else:
				attribution = self.occlusion(input_tensor)

			attributions_arr.append(attribution)

		average_arr = []
		for i in range(len(attributions_arr[0])):
			average = sum([attribute[i] for attribute in attributions_arr])
			average_arr.append(average)

		# max-aggregated attributions per field
		if aggregation == 'max':
			final_arr = []
			index = 0
			for c, field in zip(self.taken_ls, self.fields_ls):
				max_val = 0
				for k in range(index, index + c):
					max_val = max(max_val, average_arr[k])
				final_arr.append([field, max_val])
				index += c

		elif aggregation == 'average':
			final_arr = []
			index = 0
			for c, field in zip(self.taken_ls, self.fields_ls):
				sum_val = 0
				for k in range(index, index + c):
					sum_val += average_arr[k]
				final_arr.append([field, sum_val/c])
				index += c

		# max-normalize occlusions
		maximum_attribute = max([i[1] for i in final_arr])
		if normalized and max(final_arr) != 0:
			correction_factor = 1 / maximum_attribute
			final_arr = [float(i[1]*correction_factor) for i in final_arr]

		my_cmap = plt.cm.get_cmap('viridis')
		colors = my_cmap(final_arr)
		plt.barh(self.fields_ls, final_arr, color=colors, edgecolor='black')
		plt.yticks(np.arange(0, len(self.fields_ls)),
				[i for i in self.fields_ls],
				rotation='horizontal')

		plt.tight_layout()
		plt.xlabel('Importance')
		plt.savefig('readable_{}'.format(count), dpi=400, bbox_inches='tight')
		plt.close()
		return


	def graph_attributions(self):
		"""
		Plot the attributions of the model for 5 inputs

		Args:
			None

		Returns:
			None (dislays matplotlib.pyplot object)
		
		"""

		# view horizontal bar charts of occlusion attributions for five input examples
		for i in range(5):
			occlusion = self.occlusion(self.input_tensors[i])
			gradientxinput = self.gradientxinput(self.input_tensors[i])
			indicies_arr = [i for i in range(len(occlusion))]
			plt.style.use('dark_background')
			plt.barh(indicies, occlusion)
			plt.yticks(np.arange(0, len(self.fields_ls)), [i for i in self.fields_ls])
			plt.tight_layout()
			plt.show()
			plt.close()

		return


class Interpret:

	def __init__(self, model, input_tensors, output_tensors, fields):
		self.model = model 
		self.field_array = fields
		self.output_tensors = output_tensors
		self.input_tensors = input_tensors


	def occlusion(self, input_tensor, field_array):
		"""
		Generates a perturbation-type attribution using occlusion.

		Args:
			input_tensor: torch.Tensor
			field_array: arr[int], indicies that mark ends of each field 

		Returns:
			occlusion_arr: array[float] of scores per input index

		"""

		occl_size = 1

		output = self.model(input_tensor)
		zeros_tensor = torch.zeros(input_tensor)
		occlusion_arr = [0 for i in range(len(input_tensor))]
		indicies_arr = []
		total_index = 0

		for i in range(len(field_array)):

			# set all elements of a particular field to 0
			input_copy = torch.clone(input_tensor)
			for j in range(total_index, total_index + field_array[i]):
				input_copy[j] = 0.

			total_index += field_array[i]

			output_missing = self.model(input_copy)

			# assumes a 1-dimensional output
			occlusion = abs(float(output) - float(output_missing))
			indicies_arr.append(i)

		# max-normalize occlusions
		if max(occlusion_arr) != 0:
			correction_factor = 1 / (max(occlusion_arr))
			occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return indicies_arr, occlusion_arr


	def gradientxinput(self, input_tensor, output_shape, model):
		"""
		 Compute a gradientxinput attribution score

		 Args:
		 	input: torch.Tensor() object of input
		 	model: Transformer() class object, trained neural network

		 Returns:
		 	gradientxinput: arr[float] of input attributions

		"""

		# change output to float
		input.requires_grad = True
		output = model.forward(input_tensor)

		# only scalars may be assigned a gradient
		output = output.reshape(1, output_shape).sum()

		# backpropegate output gradient to input
		output.backward(retain_graph=True)

		# compute gradient x input
		final = torch.abs(input_tensor.grad) * input_tensor

		# separate out individual characters
		saliency_arr = []
		s = 0
		for i in range(len(final)):
			if i % 67 == 0 and i > 0: # assumes ASCII character set
				saliency_arr.append(s)
				s = 0
			s += float(final[i])

		# append final element
		saliency_arr.append(s)
		inputxgradient = saliency

		# max norm
		for i in range(len(inputxgrad)):
			maximum = max(inputxgrad[i], maximum)

		# prevent a divide by zero error
		if maximum != 0:
			for i in range(len(inputxgrad)):
				inputxgrad[i] /= maximum

		return inputxgrad


	def heatmap(self, n_observed=100, method='combined'):
		"""
		Generates a heatmap of attribution scores per input element for
		n_observed inputs

		Args:
			n_observed: int, number of inputs
			method: str, one of 'combined', 'gradientxinput', 'occlusion'

		Returns:
			None (saves matplotlib.pyplot figure)

		"""
		attributions_array = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [(i+j)/2 for i, j in zip(occlusion, gradxinput)]

			elif method == 'gradientxinput':
				attribution = self.gradientxinput(input_tensor)

			else:
				attribution = self.occlusion(input_tensor)

			attributions_array.append(attributions)

		plt.imshow(attributions_array)
		plt.savefig('attributions.png')
		plt.close()
		return














