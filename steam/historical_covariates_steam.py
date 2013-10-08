#!/bin/env python

import sys
import datetime


class Historical_Covariates_Steam:

	def __init__(self, keys, obs_data_obj, weather_obj, hour, model_type,
			options, lgr):

		self.lgr = lgr
		self.options = options
		self.keys = keys
		self.obs_data_obj = obs_data_obj
		self.weather_obj = weather_obj

		self.hour = hour
		self.model_type = model_type

		self.ONE_DAY  = datetime.timedelta(days=1)

		self.columns = ['sim_day_%s' % model_type, 'sim_day2_%s' % model_type,
			#'sim_day_avg_%s' % model_type, 'sim_day2_avg_%s' % model_type,
			'sim_day_hour_avg_%s' % model_type,
			'sim_day2_hour_avg_%s' % model_type]
		self.covariates = self.gen_covariates()


	def compute_average(self, start_dt, end_dt):
		""" compute average space temp over [start_dt, end_dt)"""
		# generate time stamp from date
		midnight_tm = datetime.time(0, 0, 0, 0)
		start_ts = datetime.datetime.combine(start_dt, midnight_tm)
		end_ts   = datetime.datetime.combine(end_dt, midnight_tm)

		# try:
			# start_idx = self.obs_data_obj.data.index(start_ts)
		# except ValueError, e:
			# if self.options.debug is not None and self.options.debug == 1:
				# self.lgr.warning('key not found: %s' % e)
			# return None

		sum, count = 0.0, 0 
		# for key in self.obs_data_obj.keys[start_idx:]:
			# if key >= end_ts:
				# break

			# sum   += self.obs_data_obj.data[key]
			# count += 1
			
		tmp_ts = start_ts
		gap_td = datetime.timedelta(minutes=self.options.forecast_granularity)
		while tmp_ts < end_ts:
			if tmp_ts in self.obs_data_obj.data:
				sum   += self.obs_data_obj.data[tmp_ts]
				count += 1

			tmp_ts += gap_td

		# compute average
		if count:
			return sum/count
		return None


	def compute_average_hour(self, start_dt, end_dt):
		""" compute average space temp over [start_dt, end_dt)"""
		# generate time stamp from date
		start_tm = datetime.time(self.hour, 0, 0, 0)
		start_ts = datetime.datetime.combine(start_dt, start_tm)
		end_ts   = datetime.datetime.combine(end_dt, start_tm)

		# try:
			# start_idx = self.obs_data_obj.keys.index(start_ts)
		# except ValueError, e:
			# if self.options.debug is not None and self.options.debug == 1:
				# self.lgr.warning('key not found: %s' % e)
			# return None

		sum, count = 0.0, 0 
		# for key in self.obs_data_obj.keys[start_idx:]:
			# if key >= end_ts:
				# break

			##filter: look at the data for model hour only 
			# if key.hour != self.hour:
				# continue

			# sum   += self.obs_data_obj.data[key]
			# count += 1

		tmp_ts = start_ts
		gap_td = datetime.timedelta(minutes=self.options.forecast_granularity)
		while tmp_ts < end_ts:

			if tmp_ts.hour > self.hour or tmp_ts >= end_ts:
				break

			if tmp_ts.hour != self.hour:
				tmp_ts += gap_td
				continue

			if tmp_ts in self.obs_data_obj.data:
				sum   += self.obs_data_obj.data[tmp_ts]
				count += 1
			tmp_ts += gap_td

		# compute average
		if count:
			return sum/count
		return None


	def gen_covariates(self):
		""" compute covariates """

		covariates = []
		debug_dt = datetime.date(2013, 5, 14)

		for ts in self.keys:

			dt = ts.date()
			tm = ts.time()

			# similar-weather days
			sim_dt = self.weather_obj.similar_humidex_day_cache[dt][0][0]
			sim_dt2 = self.weather_obj.similar_humidex_day_cache[dt][1][0]

			iso_weekday = ts.isoweekday()

			# compute similar-weather-day2 observed data
			sim_day2_obs = None
			sim_day2_ts = datetime.datetime.combine(sim_dt2, tm)
			if sim_day2_ts in self.obs_data_obj.data:
				sim_day2_obs = self.obs_data_obj.data[sim_day2_ts]

			# compute similar observation
			sim_day_obs = None
			sim_day_ts = datetime.datetime.combine(sim_dt, tm)
			if sim_day_ts in self.obs_data_obj.data:
				sim_day_obs = self.obs_data_obj.data[sim_day_ts]

			# previous day average space temp
			#sim_day_avg_obs = None
			# sim_day_avg_obs = self.compute_average(sim_dt,
				# sim_dt + self.ONE_DAY)

			# similar_day_hour_avg_spc_temp
			sim_day_hour_avg_obs = self.compute_average_hour(sim_dt,
				sim_dt + self.ONE_DAY)

			# previous week average space temp
			# sim_day2_avg_obs = None
			interval_end_dt = sim_dt2 + self.ONE_DAY
			# sim_day2_avg_obs = self.compute_average(
				# sim_dt2, interval_end_dt)

			# prev_wk_hour_avg_spc_temp
			sim_day2_hour_avg_obs = self.compute_average_hour(
				sim_dt2, interval_end_dt)

			if dt == debug_dt:
				self.lgr.info('ts: %s,sim dt: %s, sim dt2: %s' % (
					ts, sim_dt, sim_dt2))
				#sys.exit(0)

			covariates.append([sim_day_obs, sim_day2_obs,
				#sim_day_avg_obs, sim_day2_avg_obs,
				sim_day_hour_avg_obs, sim_day2_hour_avg_obs])

		return covariates
			