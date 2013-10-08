#!/bin/env python

""" predict space temperature in a given quadrant of a given floor based on
	past observed load and other covariates
"""

__version__ = '$Id'
__author__ = 'agagneja@ccls.columbia.edu'
_module = 'predict_steam_usage'

import os
import sys
from sets import Set
from collections import OrderedDict, deque
import datetime
import math
import optparse
import cfgparse
import traceback

from common_rudin.common import log_from_config, setup, setup_cparser
from compile_covariates import Compile_Covariates
import common_rudin.utils as utils


from steam import Steam
from common_rudin.weather import Weather
from common_rudin.build_model import Build_Model


def main(argv):
	""" driver function """

	if argv is None:
		argv = sys.argv

	version = "%prog 0.1"
	usage = "usage: %prog [options]"

	oparser = optparse.OptionParser(usage=usage, version=version,
		description=__doc__)
	cparser = cfgparse.ConfigParser()
	setup(oparser, cparser, _module)

	options, args = cparser.parse(oparser, argv)

	lgr = log_from_config(options, _module)
	lgr.info('*** %s starting up' % _module)
	
	arg_count = len(args)

	run_ts = datetime.datetime.now()
	lgr.info('run time: %s' % run_ts)

	# determine forecast start time
	forecast_start_ts = None
	if arg_count == 1:
		forecast_start_ts = run_ts
	elif arg_count == 6:
		int_args = map(int, args[1:])
		forecast_start_ts = datetime.datetime(int_args[0], int_args[1], int_args[2],
			int_args[3], int_args[4])
	else:
		lgr.critical(args)
		oparser.error("incorrect number of arguments")

	forecast_start_ts = utils.adjust_ts(forecast_start_ts,
			options.forecast_granularity, options, lgr)
	lgr.info('forecast start time: %s' % forecast_start_ts)

	# get building ids
	building_ids = utils.parse_value_list(options.building_ids)
	weather_station_ids = utils.parse_value_list(options.weather_station_ids)

	reinstantiate_cparser = False
	for bldg_idx, building_id in enumerate(building_ids):

		if bldg_idx != 0:
			reinstantiate_cparser = True
		options = utils.load_bldg_config(building_id, oparser, cparser, argv,
			options, lgr, reinstantiate_cparser)

		# check if building has steam supply
		if not options.has_steam_supply:
			lgr.info("building %s doen't have steam suply" % building_id)
			continue

		#lgr.info('building id: %s, options: %s' % (building_id, options))

		#bldg_floors, floor_quadrants = utils.get_floors_quadrants(options, lgr)

		# read weather data; weather is assumed to be same for all floors of a building
		weather_obj = Weather(lgr, options, forecast_start_ts)

		# read inputs
		# read space temperature
		steam_obj = Steam(lgr, options, forecast_start_ts)

		weather_obj.compute_similar_weather_day_cache(steam_obj.keys)

		models = []
		model_type = 'steam_demand'
		for hour_idx in range(0, 24):

			try:

				# compile train covariates
				train_covr_obj = Compile_Covariates(None, None, options, lgr,
					hour_idx, steam_obj.keys, steam_obj, weather_obj, model_type)

				# generate test covariates
				test_covr_obj = Compile_Covariates(None, None, options, lgr,
					hour_idx, None, steam_obj, weather_obj, model_type, False,
					forecast_start_ts, options.forecast_length,
					train_covr_obj.scaled_params_file)

				# fetch cross-validation results
				opt_c, opt_gamma = utils.get_optimal_model_params_bldg(hour_idx,
					forecast_start_ts, options.xval_results_table_steam,
					options, lgr)
				lgr.info('optimal C: %g, optimal gamma: %g' % (opt_c, opt_gamma))

				# build model
				model_obj = Build_Model(train_covr_obj.covariates_file, opt_c,
					opt_gamma, options, lgr)

				# apply model
				p_labels = model_obj.apply_model(test_covr_obj.covariates_file)
				p_labels = utils.validate_labels(p_labels)

				utils.save_prediction_steam_electric(run_ts, hour_idx, p_labels,
					test_covr_obj.actual_keys, options.results_table_steam,
					model_obj.model.get_svr_probability(),options, lgr)

			except Exception, e:
				lgr.info('Prediction failure for hour %d: %s' % (hour_idx,
					traceback.format_exc()))



if __name__ == '__main__':
	main(sys.argv)