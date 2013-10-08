#!/bin/env python

""" predict space temperature in a given quadrant of a given floor based on
	past observed load and other covariates
"""

__version__ = '$Id'
__author__ = 'agagneja@ccls.columbia.edu'
_module = 'steam'

from collections import OrderedDict
import datetime
import os

from common_rudin.db_utils import connect
import common_rudin.utils as utils

steam_demand_data_query = """
	SELECT TIMESTAMP, VALUE
	FROM [%s].dbo.[%s]
	WHERE TIMESTAMP > ? AND TIMESTAMP < ?
	ORDER BY TIMESTAMP
"""



class Steam:
	""" process space temp data """

	def __init__(self, lgr, options, forecast_start_ts):

		self.lgr = lgr
		self.options = options
		self.forecast_start_ts = forecast_start_ts

		self.columns = ['steam_demand']
		self.data, self.keys = self.get_data()

		self.data, self.keys = utils.interpolate_data2(self.data,
			options, lgr)

		#self.radius = 30 # days
		self.data, self.keys = self.filter_data(forecast_start_ts)
		
		if self.options.debug is not None and self.options.debug == 1:
			utils.write_dict_to_csv(self.data, os.path.join(
				self.options.temp_dir, 'steam_demand.csv'),
				['timestamp'].extend(self.columns))

		self.MAX_OBS_GAP = datetime.timedelta(minutes=30)
		self.validate_data()



	def filter_data(self, forecast_start_ts):
		""" filter out data data from other seasons and/or irrelevant data
			the start and end markers (month, day, hour and minute combination) specify
			the boundaries for the data to keep
		"""

		data_end_ts = self.keys[-1]
		radius_td = datetime.timedelta(days=30)
		ONE_YEAR = datetime.timedelta(days=365)
		year_ago_lower_bound_ts = forecast_start_ts - ONE_YEAR - radius_td
		year_ago_upper_bound_ts = forecast_start_ts - ONE_YEAR + radius_td
		two_years_ago_lower_bound_ts = forecast_start_ts - 2*ONE_YEAR - radius_td
		two_years_ago_upper_bound_ts = forecast_start_ts - 2*ONE_YEAR + radius_td

		#filtered_data = OrderedDict([])
		filtered_keys = []
		for ts, val in self.data.iteritems():
			# if ts.month  >= start_month and ts.month  <= end_month and \
			   # ts.day    >= start_day   and ts.day    <= end_day and \
			   # ts.hour   >= start_hr    and ts.hour   <= end_hour and \
			   # ts.minute >= start_min   and ts.minute <= end_minute:
			if ts <= forecast_start_ts and \
			  (ts >= forecast_start_ts - radius_td or \
               (ts >= year_ago_lower_bound_ts and ts <= year_ago_upper_bound_ts) or
			   (ts >= two_years_ago_lower_bound_ts and ts <= two_years_ago_upper_bound_ts)):
				#filtered_data[ts] = val
				filtered_keys.append(ts)
		
		return [self.data, filtered_keys]



	def validate_data(self):
		""" validate raw data """

		prev_ts = None
		for ts, _ in self.data.items():
			if prev_ts and ts - prev_ts > self.MAX_OBS_GAP:
				self.lgr.warning('steam demand observations missing between %s and %s'
					% (prev_ts, ts))
			prev_ts = ts


	def get_data_csv(self):
		""" get steam demand data from csv 
			assumption: data is chronologically ordered
		"""

		data = OrderedDict([])
		keys = []

		# use csv if one is available
		if self.options.steam_data_file and \
		   len(self.options.steam_data_file) > 0:
			with open(self.options.steam_data_file, 'r') as f_in:

				# skip first two lines
				for line in f_in.readlines()[2:]:
					cols = line.split('\t')

					# strip whitespace
					cols = map(str.strip, cols)

					# parse timestamp
					ts = datetime.datetime.strptime(cols[0], '%m/%d/%Y %H:%M')

					# filter based on required granularity
					if ts.minute % self.options.forecast_granularity != 0:
						continue

					# ignore seconds and microseconds
					ts -= datetime.timedelta(seconds=ts.second,
						microseconds=ts.microsecond)

					# parse load value: may have commas, may end with a 
					# fullstop (.) and be surrounded by double quotes
					#self.lgr.info(cols[1])
					val_str = cols[1].strip('".').replace(',', '')
					load = 0.0
					if len(val_str):
						load = float()

					data[ts] = load
					keys.append(ts)

		return [data, keys]



	def get_data(self):
		""" get steam demand data from database
		"""

		# get data from MACH Energy CSV
		data, keys = self.get_data_csv()

		most_recent_csv_ts = datetime.datetime(1970, 1, 1, 0, 0)
		if len(keys):
			most_recent_csv_ts = keys[-1]

		# connect
		cnxn, cursr = connect(self.options.db_driver, self.options.db_user,
			self.options.db_pwd, self.options.building_db,
			self.options.building_db_server)

		query = steam_demand_data_query % (self.options.building_db,
			self.options.steam_demand_table)

		if self.options.debug is not None and self.options.debug == 1:
			self.lgr.info('executing %s' % query)

		cursr.execute(query, most_recent_csv_ts, self.forecast_start_ts)

		data = OrderedDict([])
		keys = []

		for row in cursr.fetchall():
			ts, steam_demand = row

			# filter based on required granularity
			if ts.minute % self.options.forecast_granularity != 0:
				continue

			# ignore seconds and microseconds
			ts -= datetime.timedelta(seconds=ts.second,
				microseconds=ts.microsecond)

			data[ts] = steam_demand
			keys.append(ts)

		#lgr.info('first entry = %s' % self.steam_demand_data.items())

		# close connection
		cursr.close()
		cnxn.close()

		return [data, keys]