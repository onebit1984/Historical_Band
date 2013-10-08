#!/bin/env python

import tempfile
import os
import subprocess
import sys
import datetime
import traceback

import common_rudin.utils as utils
from common_rudin.weather_covariates import Weather_Covariates
from common_rudin.time_covariates import Time_Covariates
#from common_rudin.historical_covariates import Historical_Covariates
from historical_covariates_steam import Historical_Covariates_Steam
from weather_covariates_steam import Weather_Covariates_Steam

BUFFER_FLUSH_THRESHOLD = 1000

class Compile_Covariates:
	""" scale compile covariates
	"""

	def __init__(self, floor, quadrant, options, lgr, hour_idx, keys,
		space_temp_obj, weather_obj, model_type, is_train=True, forecast_start_ts=None,
		forecast_length=24, scaled_params_file=None):

		self.options = options
		self.lgr = lgr
		self.floor = floor
		self.quadrant = quadrant

		self.scaled_params_file = scaled_params_file
		self.hour_idx = hour_idx

		self.weather_obj = weather_obj
		self.space_temp_obj = space_temp_obj
		self.is_train = is_train

		self.forecast_start_ts = forecast_start_ts
		self.forecast_length = forecast_length
		self.model_type = model_type

		self.keys = None
		if is_train:
			self.keys = self.filter_keys(keys)
		else:
			# generate test keys
			self.keys = self.filter_keys(self.gen_test_keys(forecast_start_ts,
				forecast_length))
			weather_obj.compute_similar_weather_day_cache(self.keys)

		# generate time-related covariates
		self.time_covr_obj = Time_Covariates(self.keys, self.options, self.lgr)

		# generate weather-derived covariates
		# compute historical-observation-derived covariates
		# if model_type != "steam_demand":
		self.weather_covr_obj = Weather_Covariates_Steam(self.keys,
			self.weather_obj, self.options, self.lgr)

		self.historical_covr_obj = Historical_Covariates_Steam(self.keys,
			self.space_temp_obj, self.weather_obj, self.hour_idx, self.model_type,
			self.options, self.lgr)

		self.actual_keys = []
		self.covariates_file, self.scaled_params_file, self.covr_file = \
			self.compile_covariates()
		# else:
			# self.weather_covr_obj = Weather_Covariates_Steam(self.keys,
				# self.weather_obj, self.options, self.lgr)

			# self.historical_covr_obj = Historical_Covariates_Steam(self.keys,
				# self.space_temp_obj, self.weather_obj, self.hour_idx,
				# self.model_type, self.options, self.lgr)

		# actual keys: used for test covariates to indicate missing data


	def __del__(self):
		""" cleanup """
		# remove temp files
		if not (self.options.debug is not None and self.options.debug == 1):
			try:
				os.remove(self.covariates_file)
				if self.is_train:
					os.remove(self.scaled_params_file)
				else:
					os.remove(self.covr_file)
			except Exception, e:
				self.lgr.warning('cleanup failed: %s' % traceback.format_exc())


	def filter_keys(self, keys):
		""" filter keys for hourly model """

		filtered_keys = []
 
		for key in keys:
			if key.hour == self.hour_idx:
				filtered_keys.append(key)

		# for k in filtered_keys:
			# self.lgr.info(k)
		return filtered_keys


	def gen_test_keys(self, forecast_start_ts, forecast_length):
		""" generate keys for test data """

		if forecast_start_ts is None:
			self.lgr.critical('forecast start ts must be set')
			sys.exit(1)

		tmp_ts = forecast_start_ts
		gap_td = datetime.timedelta(minutes=self.options.forecast_granularity)
		end_ts = forecast_start_ts + datetime.timedelta(hours=forecast_length)

		keys = []
		while tmp_ts < end_ts:
			keys.append(tmp_ts)
			tmp_ts += gap_td

		return keys


	def compile_covariates(self):
		""" compile covariates into libsvm format """

		# compile covariates
		compiled_covariates = []
		y = []

		for i, key in enumerate(self.keys):

			# append covariates
			row = []
			try:

				row.append(self.weather_obj.humidex_regularized[key])
				row.extend(self.weather_covr_obj.covariates[i])
				row.extend(self.time_covr_obj.covariates[i])
				row.extend(self.historical_covr_obj.covariates[i])

			except KeyError, e:
				#self.lgr.info('%s' % e)
				if self.is_train:
					if self.options.debug:
						self.lgr.info('covariates missing for %s: %s' % (key, row))
				else:
					self.lgr.critical('covariates missing for %s: %s' % (key, row))
				continue

			# skip rows with missing data
			if None in row:
				#self.lgr.info('row with missing data: %s: %s' % (key, row))
				if not self.is_train:
					self.lgr.debug('covariates missing for key; %s: %s' % (key, row))
				continue

			# append y
			if self.is_train:
				try:
					y.append(self.space_temp_obj.data[key])
				except KeyError, e:
					self.lgr.debug('observation data missing for %s' % key)
					continue
			else:
				y.append(0.0)

			self.actual_keys.append(key)
			compiled_covariates.append(row)

		#self.lgr.info(compiled_covariates[0])
		# save covariates file; it is used by the cross validation module
		#covariates_file = None
		#if self.options.debug is not None and self.options.debug == 1:
		covariates_file = self.save_covariates(y, compiled_covariates)
		self.lgr.info('covariates file: %s' % covariates_file)

		# generate temp file for scaling
		f_name = self.gen_scale_input(y, compiled_covariates)

		# perform scaling
		scaled_file, scale_params_file = self.scale(f_name)

		# remove temporary files
		if not (self.options.debug is not None and self.options.debug == 1):
			os.remove(f_name)

		return [scaled_file, scale_params_file, covariates_file]


	def save_covariates(self, y, compiled_covariates):
		""" save covariates to a csv file """

		# create out file name
		f_name, _ = self.create_temp_file('_%s_%s_%s_hour_%d.csv' % (
			self.model_type, self.options.building_id, self.get_covr_typ(),
			self.hour_idx))
		self.lgr.debug('covariates file: %s' % f_name)

		with open(f_name, 'w+') as f_out:

			# write header
			buffer = '%s,%s,%s,%s,%s,%s\n' % ('timestamp',
				','.join(self.space_temp_obj.columns),
				','.join(self.weather_obj.columns),
				','.join(self.weather_covr_obj.columns),
				','.join(self.time_covr_obj.columns),
				','.join(self.historical_covr_obj.columns))

			for i, key in enumerate(self.actual_keys):

				row = '%s,%g,' % (key, y[i])
				row += '%s\n' % ','.join(map(str, compiled_covariates[i]))

				buffer += row

				if i % BUFFER_FLUSH_THRESHOLD == 0:
					f_out.write(buffer)
					buffer = ''

			# empty buffer
			f_out.write(buffer)

		return f_name
	

	def create_temp_file(self, file_suffix='', close_fd=True):
		""" create temporary file; caller must delete the file after use """

		fd, f_name = tempfile.mkstemp(suffix=file_suffix,
			dir=self.options.temp_dir, text=True)
		if close_fd:
			os.close(fd)
		return [f_name, fd]



	def gen_scale_input(self, y, compiled_covariates):
		""" generate input file to scale utility """

		# create out file name
		f_name, _ = self.create_temp_file('_%s.pre_scale' % self.get_covr_typ())
		if self.options.debug is not None and self.options.debug == 1:
			self.lgr.info('temp covariates file: %s' % f_name)

		with open(f_name, 'w+') as f_out:
			buffer = ''
			for i, y_val in enumerate(y):

				line = '%g' % y_val
				for j, covariate in enumerate(compiled_covariates[i]):
					line += ' %d:%g' % (j+1, covariate)

				buffer += (line + '\n')

				if i % BUFFER_FLUSH_THRESHOLD == 0:
					f_out.write(buffer)
					buffer = ''

			# empty buffer
			f_out.write(buffer)

		return f_name



	def get_covr_typ(self):
		""" get type of covariates being compiled: whether test or train """

		building_info = ''
		# floor and quadrant are not relevant for models like steam and electric
		# demand
		if self.floor and self.quadrant:
			building_info = '_fl_%s_%s' % (self.floor,
				self.quadrant)

		if self.is_train:
			return 'train%s' % building_info
		return 'test%s' % building_info



	def scale(self, data_file):
		""" scale data """

		scale_params_arg = None
		if not self.is_train:
			scale_params_arg = '-r'
		else:
			scale_params_arg = '-s'

		# create scale params file
		scale_params_file = self.scaled_params_file
		if self.is_train:
			scale_params_file, _ = self.create_temp_file(
				'_%s.scale_params' % self.get_covr_typ())
			if self.options.debug is not None and self.options.debug == 1:
				self.lgr.info('scale params file: %s' % scale_params_file)

		# generate temp scale output file
		out_f_name, _ = self.create_temp_file('_%s.scaled' % self.get_covr_typ())
		if self.options.debug is not None and self.options.debug == 1:
			self.lgr.info('scale output file: %s' % out_f_name)

		scaling_args = [os.path.abspath(self.options.svm_scale), '-l', '0',
			scale_params_arg, os.path.abspath(scale_params_file), data_file]
		if self.options.debug is not None and self.options.debug == 1:
			self.lgr.info('scale args: %s' % scaling_args)

		try:
			utils.create_child_proc(scaling_args, out_f_name)
		except subprocess.CalledProcessError, e:
			self.lgr.critical('scale failed: %s' % e)
			sys.exit(1)

		return [out_f_name, scale_params_file]


