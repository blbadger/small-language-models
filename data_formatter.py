# data_formatter.py

# import standard libraries
import random
import string

# import third-party libraries
import torch
import pandas as pd 
import numpy as np 


class Format:
	"""
	Formats the 'linear_historical.csv' file to replicate experiments
	in arXiv:2211.02941

	Not intended for general use.
	"""

	def __init__(self, file, prediction_feature, ints_only=False):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower()[0] == 'n' else x)
		df = df[:][:10000]
		self.prediction_feature = prediction_feature
		self.places_dict = self._init_places_dict(ints_only)
		self.embedding_dim = len(self.places_dict) 

		# 80/20 training/test split and formatting
		length = len(df[:])
		split_i = int(length * 0.8)
		training = df[:][:split_i]
		self.training_inputs = training[[i for i in training.columns if i != prediction_feature]]
		self.training_outputs = training[[prediction_feature]]

		validation_size = length - split_i
		validation = df[:][split_i:split_i + validation_size]

		self.val_inputs = validation[[i for i in validation.columns if i != prediction_feature]]
		self.val_outputs = validation[[prediction_feature]]
		self.val_inputs.reset_index(inplace=True, drop=True)
		self.val_outputs.reset_index(inplace=True, drop=True)


	def _init_places_dict(self, ints_only=False):
		"""
		Initialize the dictionary storing character to embedding dim tensor map.

		kwargs:
			ints_only: bool, if True then a numerical input is expected.

		returns:
			places_dict: dictionary
		"""
		if ints_only:
			places_dict = {s:i for i, s in enumerate('0123456789. -:_')}
		else:
			chars = string.printable
			places_dict = {s:i for i, s in enumerate(chars)}

		return places_dict


	def stringify_input(self, input_type='training', short=True, n_taken=4, remove_spaces=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values, with n_taken characters per field.

		kwargs:
			input_type: str, type of data input requested ('training' or 'test' or 'validation')
			short: bool, if True then at most n_taken letters per feature is encoded
			n_taken: int, number of letters taken per feature
			remove_spaces: bool, if True then input feature spaces are removed before encoding

		Returns:
			array: string: str of values in the row of interest

		"""
		n = n_taken

		if input_type == 'training':
			inputs = self.training_inputs

		elif input_type == 'validation':
			inputs = self.val_inputs

		else:
			inputs = self.test_inputs

		if short == True:
			inputs = inputs.applymap(lambda x: '_'*n_taken if str(x) in ['', '(null)'] else str(x)[:n])
		else:
			inputs = inputs.applymap(lambda x:'_'*n_taken if str(x) in ['', '(null)'] else str(x))

		inputs = inputs.applymap(lambda x: '_'*(n_taken - len(x)) + x)
		string_arr = inputs.apply(lambda x: '_'.join(x.astype(str)), axis=1)

		return string_arr


	def string_to_tensor(self, string, flatten):
		"""
		Convert a string into a tensor

		Args:
			string: arr[str]
			flatten: bool, if True then tensor has dim [1 x length]

		Returns:
			tensor: torch.Tensor

		"""
		places_dict = self.places_dict

		# vocab_size x embedding dimension (ie input length)
		tensor_shape = (len(string), len(places_dict)) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(string):
			tensor[i][places_dict[letter]] = 1.

		if flatten:
			tensor = torch.flatten(tensor)
		return tensor


	def transform_to_tensors(self, training=True, flatten=True):
		"""
		Transform input and outputs to arrays of tensors

		kwargs:
			flatten: bool, if True then tensors are of dim [1 x length]

		"""

		if training:
			string_arr = self.stringify_input(input_type='training')
			outputs = self.training_outputs

		else:
			string_arr = self.stringify_input(input_type='validation')
			outputs = self.val_outputs

		input_arr, output_arr = [], []
		for i in range(len(string_arr)):
			if outputs[self.prediction_feature][i]:
				string = string_arr[i]
				input_arr.append(self.string_to_tensor(string, flatten))
				output_arr.append(torch.tensor(outputs[self.prediction_feature][i]))
		
		return input_arr, output_arr


	def generate_test_inputs(self):
		"""
		Generate tensor inputs from a test dataset

		"""
		inputs = []
		for i in range(len(self.test_inputs)):
			input_tensor =self.string_to_tensor(input_string)
			inputs.append(input_tensor)

		return inputs



class FormatDeliveries:
	"""
	Formats the 'linear_historical.csv' file to replicate experiments
	in arXiv:2211.02941

	Not intended for general use.
	"""

	def __init__(self, file, prediction_feature, training=True, n_per_field=False):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower() == 'nan' else x)
		df = df[:10000]
		length = len(df['Elapsed Time'])
		self.input_fields = ['Store Number', 
							'Market', 
							'Order Made',
							'Cost',
							'Total Deliverers', 
							'Busy Deliverers', 
							'Total Orders',
							'Estimated Transit Time',
							'Linear Estimation']

		if n_per_field:
			self.taken_ls = [4 for i in self.input_fields] # somewhat arbitrary size per field
		else:
			self.taken_ls = [4, 1, 8, 5, 3, 3, 3, 4, 4]

		if training:
			# df = shuffle(df)
			df.reset_index(inplace=True)
 
			# 80/20 training/validation split
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[self.input_fields]
			self.training_outputs = [i for i in training[prediction_feature][:]]

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.validation_inputs = validation[self.input_fields]
			self.validation_outputs = [i for i in validation[prediction_feature][:]]
			self.validation_inputs = self.validation_inputs.reset_index()

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify


	def stringify_input(self, index, training=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistent structure to inputs regardless of missing values.

		Args:
			index: int, position of input n 

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

	def unstructured_stringify(self, index, training=True, pad=True, length=50):
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


	@classmethod
	def string_to_tensor(self, input_string):
		"""
		Convert a string into a tensor

		Args:
			string: str, input as a string

		Returns:
			tensor: torch.Tensor() object
		"""

		places_dict = {s:int(s) for s in '0123456789'}
		for i, char in enumerate('. -:_'):
			places_dict[char] = i + 10

		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(input_string), 1, 15) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(input_string):
			tensor[i][0][places_dict[letter]] = 1.

		tensor = tensor.flatten()
		return tensor 


	def sequential_tensors(self, training=True):
		"""
		kwargs:
			training: bool

		Returns:
			input_tensors: torch.Tensor objects
			output_tensors: torch.Tensor objects
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
			output_tensors.append(torch.Tensor([outputs[i]]))

		return input_tensors, output_tensors

 
