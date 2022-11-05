# linear_models.py

import pandas as pd 
import numpy as np 

import sklearn 
from sklearn.utils import shuffle
from datetime import datetime
from statsmodels.formula.api import ols

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

def format_data():
	"""
	Reformats date data into datetime objects

	Args:
		None

	Returns:
		None (saves 'formatted_historical.csv')

	"""
	file = 'historical_data.csv'
	df = pd.read_csv(file)
	df = shuffle(df)

	# remove rows with null times
	df.reset_index(inplace=True)

	arr = []
	# convert datetime strings to datetime objects
	for i in range(len(df['actual_delivery_time'])):

		# Bottleneck here: strptime string matching is rather slow. 
		# Expect for prod data datetimes to already be formatted if pulled from an SQL database
		deldate = datetime.strptime(str(df['actual_delivery_time'][i]), '%Y-%m-%d %H:%M:%S')
		credate = datetime.strptime(str(df['created_at'][i]), '%Y-%m-%d %H:%M:%S')

		df['actual_delivery_time'][i] = deldate
		df['created_at'][i] = credate
		elapsed_time = df['actual_delivery_time'][i] - df['created_at'][i]
		elapsed_seconds = elapsed_time.total_seconds()

		arr.append(elapsed_seconds)

	# create column of actual wait time from start and end datetimes
	df['etime'] = arr

	return df

# format_data()

df = pd.read_csv('data/formatted_historical.csv')


def linear_regression(df):
	"""
	Perform a multiple linear regression to predict duration

	Args:
		df: pd.dataframe

	Returns:
		fit: statsmodel.formula.api.ols.fit() object
		None (prints error measurements)

	"""
	df = df.dropna(axis=0)

	length = len(df['etime'])

	# 80/20 training/test split
	split_i = int(length * 0.8)

	training = df[:][:split_i]
	validation_size = length - split_i

	validation = df[:][split_i:split_i + validation_size]

	val_inputs = validation[['store_id', 
							'market_id', 
							'total_busy_dashers', 
							'total_onshift_dashers', 
							'total_outstanding_orders',
							'estimated_store_to_consumer_driving_duration']]

	val_outputs = [i for i in validation['etime']]

	# R-like syntax
	fit = ols('etime ~ \
			   C(market_id) + \
			   total_onshift_dashers + \
			   total_busy_dashers + \
			   total_outstanding_orders + \
			   estimated_store_to_consumer_driving_duration', data=training).fit() 


	print (fit.summary())
	predictions = fit.predict(val_inputs)

	# weighted MSE and MAE accuracy calculations
	count = 0
	loss = 0
	ab_loss = 0
	weighted_loss = 0
	for i, pred in enumerate(predictions):	
		target = val_outputs[i]
		output = pred
		ab_loss += abs(output - target)
		if output < target:
			weighted_loss += 2*abs(output - target)
		else:
			weighted_loss += abs(output - target)
		if output < target:
			loss += (2*(output - target))**2
		else:
			loss += (output - target)**2
		count += 1

	weighted_mse = loss / count
	print (f'weighted absolute error: {weighted_loss / count}')
	print (f'Mean Abolute Error: {ab_loss / count}')
	print (weighted_mse)
	print ('weighted RMS: {}'.format(weighted_mse**0.5))

	mse = sum([(i-j)**2 for i, j in zip(predictions, val_outputs)]) / validation_size
	print (mse)
	print ('Linear model RMS error: ', (mse**0.5))

	return fit


fit = linear_regression(df)


def linear_fit(fit, file_name):
	"""
	Predict values with the trained linear model

	Args:
		fit: statsmodel.formula.api.ols.fit() object

	Returns:
		df: pd.dataframe

	"""

	df = pd.read_csv('predict_data.csv')
	length = len(df['market_id'])

	estimations = fit.predict(df)
	print (estimations)

	df['linear_ests'] = estimations
	return df


file_name = 'data/predict_data.csv'
df = linear_fit(fit, file_name)


def convert_to_int(df, field, file_name):
	"""
	Convert a float field to ints

	Args:
		df: pandas dataframe
		field: str

	Returns:
		none (modifies df and saves to storage)

	"""


	for i in range(len(df[field])):
		if str(df[field][i]) not in ['', 'NaN', 'nan']:
			df[field][i] = int(df[field][i])

	df.to_csv(file_name)


file_name = 'data_to_predict.csv'
field = 'linear_ests'
convert_to_int(df, field, file_name)


