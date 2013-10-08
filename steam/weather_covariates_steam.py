#!/bin/env python

import sys
import datetime
import traceback


class Weather_Covariates_Steam:

	def __init__(self, keys, weather_obj, options, lgr):

		self.lgr = lgr
		self.options = options
		self.keys = keys
		self.weather_obj = weather_obj

		self.columns = ['humidex_sim_day_delta', 'humidex_sim_day2_delta',
			'bldg_oper_min_temp', 'bldg_oper_max_temp',
			'bldg_oper_avg_temp', 'bldg_oper_min_temp_sim_day_delta',
			'bldg_oper_max_temp_sim_day_delta',
			'bldg_oper_avg_temp_sim_day_delta',
			'bldg_oper_min_temp_sim_day2_delta',
			'bldg_oper_max_temp_sim_day2_delta',
			'bldg_oper_avg_temp_sim_day2_delta']

		self.covariates = self.gen_covariates()


	def _get_stats(self, data):
		""" compute min max and average """
		_min, _max, _avg = sys.maxint, -sys.maxint - 1, None
		length = len(data)
		if length:
			_avg = sum(data)/length
			_min, _max = min(data), max(data)

		return [_min, _max, _avg]



	def gen_covariates(self):
		""" generate weather covariates """

		covariates = []
		# weather covariates are stay the same over a given day
		# so we use a cache to avoid recomputing them
		#covariate_cache = {}

		for ts in self.keys:
			dt = ts.date()
			tm = ts.time()

			# look in cache first
			# if dt in covariate_cache:
				# covariates.append(covariate_cache[dt])
				# continue

			# cache miss: iterate weather data/forecast for
			# buidling operation hours
			start_tm = datetime.time(self.options.building_open_hour)
			ts = datetime.datetime.combine(dt, start_tm)
			
			end_tm = datetime.time(self.options.building_close_hour)
			#end_ts = datetime.datetime.combine(dt, end_tm)
			gap_td = datetime.timedelta(
				minutes=self.options.forecast_granularity)

			# most similar days
			sim_dt = self.weather_obj.similar_humidex_day_cache[dt][0][0]
			sim_dt2 = self.weather_obj.similar_humidex_day_cache[dt][1][0]

			sim_day_ts  = datetime.datetime.combine(sim_dt, tm)
			sim_day2_ts = datetime.datetime.combine(sim_dt2, tm)

			ts_humidx       = self.weather_obj.humidex_regularized[ts] 
			humidx_sim_day  = self.weather_obj.humidex_regularized[sim_day_ts]
			humidx_sim_day2 = self.weather_obj.humidex_regularized[sim_day2_ts]

			idx = start_tm
			temp_list, sim_day_temp_list, sim_day2_temp_list = [], [], []
			while idx < end_tm:

				tmp_ts = datetime.datetime.combine(dt, idx)
				tmp_sim_day_ts = datetime.datetime.combine(sim_dt, idx)
				tmp_sim_day2_ts = datetime.datetime.combine(sim_dt2, idx)

				try:
					temp = self.weather_obj.temp_regularized[tmp_ts]
					tmp_sim_day = self.weather_obj.temp_regularized[tmp_sim_day_ts]
					tmp_sim_day2 = self.weather_obj.temp_regularized[tmp_sim_day2_ts]
				except KeyError, e:
					self.lgr.critical('weather data missing for %s\n%s' % (idx,
						traceback.format_exc()))
					sys.exit(1)

				temp_list.append(temp)
				sim_day_temp_list.append(tmp_sim_day)
				sim_day2_temp_list.append(tmp_sim_day2)

				idx = (tmp_ts + gap_td).time()


			min_temp, max_temp, avg_temp = self._get_stats(temp_list)
			min_temp_sim_day, max_temp_sim_day, avg_temp_sim_day = \
				self._get_stats(sim_day_temp_list)
			min_temp_sim_day2, max_temp_sim_day2, avg_temp_sim_day2 = \
				self._get_stats(sim_day2_temp_list)

			data = [humidx_sim_day - ts_humidx, humidx_sim_day2 - ts_humidx,
				min_temp, max_temp, avg_temp, min_temp_sim_day - min_temp,
				max_temp_sim_day - max_temp, avg_temp_sim_day - avg_temp,
				min_temp_sim_day2 - min_temp, max_temp_sim_day2 - max_temp,
				avg_temp_sim_day2 - avg_temp]

			covariates.append(data)
			#covariate_cache[dt] = data

		return covariates