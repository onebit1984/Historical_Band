#!/bin/env python

""" find optimal C and gamma for SVM model.
	Cleanup old covariate files
"""

__version__ = '$Id'
__author__ = 'agagneja@ccls.columbia.edu'
_module = 'cross_validate_steam'

import sys
import os
from collections import OrderedDict
import datetime
import re

import os
import time
import optparse
import cfgparse

from sets import Set

from common_rudin.db_utils import connect
from common_rudin.common import log_from_config, setup
import common_rudin.utils as utils

# LIBSVM
import python.svmutil as svmutil

C_SET = [50, 100, 200, 300, 400, 500, 1000, 5000, 10000]
GAMMA_SET = [0.0005, 0.001, 0.00125, 0.0015, 0.001625, 0.00175, 0.002, 0.00225, 0.0025]

model_params_shortlist = {
0:{
'C_SET': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700],
'GAMMA_SET': [0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015]
},

1:{
'C_SET': [],
'GAMMA_SET': []
},

2:{
'C_SET': [],
'GAMMA_SET': []
},
}


ONE_DAY = datetime.timedelta(days=1)
ONE_WEEK = datetime.timedelta(days=7)
ONE_MONTH = datetime.timedelta(days=30)
ARCHIVE_THRESHOLD = datetime.timedelta(days=7)


def gen_model_params_subset(hour, trng_end_ts, options, lgr):
	""" generate c and gamma range based on best c and gamma
		observed in the last 30 days
	"""

	# get best (c, gamma) values from database
	select_qry = """
		SELECT T1.Run_Datetime, T1.c, T1.gamma
		FROM (SELECT *
			  FROM [%s].dbo.[%s]
			  WHERE HOUR = ? AND Run_Datetime >= ?) T1
			INNER JOIN
			 (SELECT Run_Datetime, MIN(avg_error) min_avg_error
			  FROM [%s].dbo.[%s]
			  WHERE HOUR = ? AND Run_Datetime >= ?
			  GROUP BY RUN_DATETIME) T2
		ON T1.RUN_DATETIME = T2.RUN_DATETIME
			AND T1.avg_error = T2.min_avg_error
		ORDER BY T1.Run_Datetime
	""" % (options.results_db, options.xval_results_table_steam,
	options.results_db, options.xval_results_table_steam)

	# connect to the database
	cnxn, cursr = connect(options.db_driver, options.results_db_user, options.results_db_pwd,
		options.results_db, options.results_db_server)

	start_ts = trng_end_ts - ONE_MONTH
	cursr.execute(select_qry, hour, start_ts, hour, start_ts)
	best_Cs, best_gammas = Set([]), Set([])
	for row in cursr.fetchall():
		ts, C, gamma = row
		if ts:
			best_Cs.add(C)
			best_gammas.add(gamma)

	cnxn.close()

	# generate subsets to consider
	c_shortlist = compute_shortlist(sorted(list(best_Cs)))
	gamma_shortlist = compute_shortlist(sorted(list(best_gammas)))

	# if shortlists are empty, use defaults
	if not len(c_shortlist):
		c_shortlist = C_SET
	if not len(gamma_shortlist):
		gamma_shortlist = GAMMA_SET

	lgr.info('best c set: %s' % sorted(list(best_Cs)))
	lgr.info('new c shortlist: %s' % c_shortlist)
	lgr.info('best gamma set: %s' % sorted(list(best_gammas)))
	lgr.info('new gamma shortlist: %s' % gamma_shortlist)
	
	return [c_shortlist, gamma_shortlist]


def compute_shortlist(best_param_list_sorted):
	""" compute shortlist of model parameter """

	shortlisted_param_vals = []

	if len(best_param_list_sorted):

		#prev_val = None
		for param_val in best_param_list_sorted:

			partial_sum = 0
			# if prev_val:
				# partial_sum = prev_val
			step_size = param_val * 0.10

			shortlisted_param_vals.extend([param_val - step_size, param_val,
				param_val + step_size])
			#prev_val = param_val

		#shortlisted_param_vals.add(prev_val*2)

	return shortlisted_param_vals


def load_covariates_csv(file):
	""" load covariates from CSV file
		The first column is assumed to contain timestamp YYYY-MM-DD HH:MM:SS
	"""

	covariates = OrderedDict([])

	with open(file, 'r') as f_in:
		# skip header
		for line in f_in.readlines()[1:]:
			cols = line.split(',')
			#print cols
			cols_clean = []
			for col in cols[1:]:
				if col == 'None':
					cols_clean.append(None)
				else:
					cols_clean.append(float(col))

			covariates[datetime.datetime.strptime(cols[0],
				'%Y-%m-%d %H:%M:%S')] = cols_clean

	return covariates



def save_grid_srch_results(results, run_ts, options, lgr):
	""" save grid search results """

	if not len(results):
		lgr.info('no results to save')
		return

	# connect to the database
	cnxn, cursr = connect(options.db_driver, options.results_db_user, options.results_db_pwd,
		options.results_db, options.results_db_server)

	insert_stmt = """
		INSERT INTO [%s].dbo.[%s]
		(Run_DateTime, Hour, c, gamma, avg_error)
		VALUES (?, ?, ?, ?, ?)
	""" % (options.results_db, options.xval_results_table_steam)

	insert_seq = []
	for hour, result in enumerate(results):
		for result_row in result.iteritems():
			params, err_vals = result_row
			c, gamma = params
			insert_seq.append((run_ts, hour, c, gamma,
				sum(err_vals)/len(err_vals)))

	if len(insert_seq):
		cursr.executemany(insert_stmt, insert_seq)
		cnxn.commit()
	cnxn.close()



def find_latest_training_file(hour, options, lgr, directory='.'):
	""" find most recent training covariate csv files in the directory"""

	TRAIN_FILENAME_FORMAT = re.compile(
		r'%s' % options.train_covariate_filename_fmt % (options.building_floor,
			options.floor_quadrant, hour))

	max_mtime, newest_train_file = None, None
	for dirname, _, filenames in os.walk(directory):

		for filename in filenames:

			if TRAIN_FILENAME_FORMAT.match(filename):

				# get file path and last modify time
				file_t = os.path.join(dirname, filename)
				time_t = os.path.getmtime(file_t)

				# if current file is more recent, save its path
				if max_mtime:
					if max_mtime < time_t:
						max_mtime, newest_train_file = time_t, file_t
				else:
					max_mtime, newest_train_file = time_t, file_t

	file_mtime = ''
	if max_mtime:
		file_mtime = time.ctime(max_mtime)
	return newest_train_file, file_mtime


def find_latest_training_files(options, lgr, directory='.'):
	""" find most recent training covariate csv files in the directory
		This is done for each hourly model. Also cleanup directory of
		old covariate files
	"""

	TRAIN_FILENAME_FORMAT = re.compile(
		r'%s' % options.train_covariate_filename_fmt_steam % options.building_id)

	covariate_files = [[None, None]]*24 # one entry for each hour's covariates
	files_to_del = []

	max_mtime, newest_train_file = None, None
	for dirname, _, filenames in os.walk(directory):

		for filename in filenames:

			match_obj = TRAIN_FILENAME_FORMAT.match(filename)
			if match_obj:

				hour = int(match_obj.group(1))

				if hour not in range(0, 24):
					continue

				# get file path and last modify time
				file_t = os.path.join(dirname, filename)
				time_t = datetime.datetime.fromtimestamp(os.path.getmtime(file_t))

				files_to_del.append([time_t, file_t])

				# get last seen file name and modify time
				max_mtime, _ = covariate_files[hour]

				# if current file is more recent, save its path
				if max_mtime:
					if max_mtime < time_t:
						covariate_files[hour] = [time_t, file_t]
				else:
					covariate_files[hour] = [time_t, file_t]

	# do not delete most recent coavariate files
	for mtime, file in covariate_files:
		if file:
			files_to_del.remove([mtime, file])

	lgr.info('cleaning up old covariate files')
	# do not delete very recent files
	now = datetime.datetime.now()
	for mtime, file in files_to_del:
		if now - mtime > ARCHIVE_THRESHOLD:
			lgr.info('removing file %s' % file)
			os.remove(file)

	return covariate_files



MAX_DATA_GAP = 92
def validate_test_data(data_dict, start_idx, end_idx, data_granularity):
	""" find gaps in data[start_idx:end_idx+1]
		Boundary data must be present
	"""

	# TODO : check for missing covariate values
	# check boundary data
	boundary_data_missing = False
	if start_idx not in data_dict or end_idx not in data_dict:
		boundary_data_missing = True

	curr_idx = start_idx
	max_gap, curr_gap = 0, 0
	prev_idx = None
	in_gap = False

	while curr_idx <= end_idx:
		if (curr_idx not in data_dict) or None in data_dict[curr_idx]:

			if in_gap == True:
				curr_gap += 1
			else:
				curr_gap = 1
				in_gap = True

		elif in_gap == True:
			in_gap = False
			max_gap = max(max_gap, curr_gap)
			curr_gap = 0

		curr_idx += data_granularity

	max_gap = max(max_gap, curr_gap)
	return (max_gap, boundary_data_missing)



def gen_xval_data(data_dict, test_start_idx, test_end_idx, options, lgr):
	""" generate test and train data using the test data range provided
		The data leftover after generating test data is used for training
	"""

	train_data, test_data = OrderedDict([]), OrderedDict([])
	for ts, row in data_dict.iteritems():
		if ts < test_start_idx or ts > test_end_idx:
			train_data[ts] = row
		else:
			test_data[ts] = row

	return (train_data, test_data)



def gen_scaled_svm_file(data, out_file, out_file_scaled, options, lgr,
						scale_params_file=None):
	""" generate file for regression in libsvm format """

	skipped_rows = 0
	missing_observations = 0

	with open(out_file, 'w') as f_out:
		row_idx = -1
		for (ts, row) in data.iteritems():
			row_idx += 1

			line = []
			line.append(row[0]) # target/Y value

			missing_cols = 0
			for (i, col) in enumerate(row[1:]):
				if col is None:
					missing_cols += 1

				if col == 0: # 0 is the default value for svm-light
					continue

				line.append('%d:%s' % (i+1, col))

			if missing_cols == 0:
				f_out.write(' '.join(map(str, line)) + '\n')
			else:
				skipped_rows += 1

	lgr.warning('rows with missing observations = %d' % skipped_rows)

	# if scaling parameters are available, use them
	scale_params_arg = None
	if scale_params_file:
		scale_params_arg = '-r'
	else:
		scale_params_arg = '-s'
		scale_params_file, _ = utils.create_temp_file(options, lgr,
			'_scale_params')

	scaling_args = [os.path.abspath(options.svm_scale), '-l', '0', scale_params_arg,
		scale_params_file, out_file]

	utils.create_child_proc(scaling_args, out_file_scaled)
	return scale_params_file



def cleanup(tmp_files, options, lgr):
	""" cleanup: removes temporaries """

	# remove temp files
	if not (options.debug is not None and options.debug == 1):
		for tmp_file in tmp_files:
			os.remove(tmp_file)



def find_optimal_params(train_covariates, data_granularity, test_set_size,
						step_size, hour_idx, options, lgr):
	""" data_granularity: interval, in minutes, between each data reading
		test_set_size: timedelta object of the size of test set to use for
			cross-validation
		step_size: timedelta object of the size of step to take on moving
			from one test set to the next

		1. Divide training data in training and testing set
			Test set: 1 week of contiguous data
			Training set: rest of the data set
			Assumption: The total data set is at least 1 month worth of data
				with no missing covariates
		2. For each unique (C, gamma) pair from the set, train on training set and test on test set. Save test set scores.
		3. Find (C, gamma) pair with best average performance
	performs cross-validation using the given subset of C, gamma
	"""

	opt_c, opt_gamma = None, None
	ts_keys = train_covariates.keys()
	data_start_ts, data_end_ts = ts_keys[0], ts_keys[-1]

	if options.debug is not None and options.debug == 1:
		lgr.info('performing cross validation')
		lgr.info('training data: %s - %s' % (data_start_ts, data_end_ts))

	data_gran_td = datetime.timedelta(minutes=data_granularity)
	results = {}

	# set C and gamma shortlist based on recent observed best C and gamma
	c_set, gamma_set = gen_model_params_subset(hour_idx, data_end_ts,
			options, lgr)

	# set initial test data interval
	curr_test_start_ts, curr_test_end_ts = data_start_ts, data_start_ts + test_set_size

	while curr_test_end_ts <= data_end_ts:

		lgr.info('examining test interval (%s, %s)' % (curr_test_start_ts,
			curr_test_end_ts))

		# check test interval for gaps
		(max_gap, bounds_missing) = validate_test_data(train_covariates,
				curr_test_start_ts, curr_test_end_ts, data_gran_td)

		if bounds_missing or max_gap > MAX_DATA_GAP:
			lgr.info('max gap = %d, bound missing = %s' % (max_gap, bounds_missing))
			curr_test_start_ts, curr_test_end_ts = curr_test_start_ts + \
									step_size, curr_test_end_ts + step_size
			continue

		lgr.info('using test interval (%s, %s)' % (str(curr_test_start_ts),
				str(curr_test_end_ts)))

		train_data, test_data = gen_xval_data(train_covariates,
			curr_test_start_ts, curr_test_end_ts, options, lgr)

		# generate scaled train and test data files
		train_file_prefix = '_train_xval_%s_%s' % (
			curr_test_start_ts.strftime('%Y%m%d%H%M'),
			curr_test_end_ts.strftime('%Y%m%d%H%M'))

		scale_input_train_file, _ = utils.create_temp_file(options, lgr,
			train_file_prefix + '.pre_scale')
		scaled_train_file, _ = utils.create_temp_file(options, lgr,
			train_file_prefix + '.scaled')
		scale_params_file = gen_scaled_svm_file(train_data,
			scale_input_train_file, scaled_train_file, options, lgr, None)

		test_file_prefix = '_test_xval_%s_%s' % (
			curr_test_start_ts.strftime('%Y%m%d%H%M'),
			curr_test_end_ts.strftime('%Y%m%d%H%M'))
		scale_input_test_file, _ = utils.create_temp_file(options, lgr,
			test_file_prefix + '.pre_scale')
		scaled_test_file, _ = utils.create_temp_file(options, lgr,
			test_file_prefix + '.scaled')
		gen_scaled_svm_file(test_data, scale_input_test_file, scaled_test_file,
			options, lgr, scale_params_file)

		y_train, x_train = svmutil.svm_read_problem(scaled_train_file)
		y_test_actual, x_test = svmutil.svm_read_problem(scaled_test_file)

		for c in c_set:
			for gamma in gamma_set:

				if options.debug is not None and options.debug == 1:
					lgr.info('C = %g, gamma = %g' % (c, gamma))

				model = svmutil.svm_train(y_train, x_train,
							'-s 3 -t 2 -c %g -g %s -h 0 -q' % (c, gamma))
				p_label, p_acc, p_val = svmutil.svm_predict(y_test_actual, x_test,
															model)

				scores = svmutil.evaluations(y_test_actual, p_label)
				if (c, gamma) in results:
					results[(c, gamma)].append(scores[1]) # Acc, MSE, SCC
					#results[(c, gamma)].append(compute_mae(y_test_actual, p_label))
				else:
					results[(c, gamma)] = [scores[1]] # Acc, MSE, SCC
					#results[(c, gamma)] = [compute_mae(y_test_actual, p_label)]

		curr_test_start_ts, curr_test_end_ts = curr_test_start_ts + step_size, \
						curr_test_end_ts + step_size

		# clean up
		cleanup([scale_input_train_file, scaled_train_file,
			scale_input_test_file, scaled_test_file, scale_params_file],
			options, lgr)
			

	best_score = 99999999.0
	for (p, l) in results.iteritems():
		avg_score = sum(l)/len(l)
		if avg_score < best_score: # for SCC, we do a MAX
			opt_c, opt_gamma = p 
			best_score = avg_score

	if opt_c and opt_gamma:
		lgr.info('optimal C = %g, optimal gamma = %g' % (opt_c, opt_gamma))
	lgr.info('best score = %g' % best_score)
	return results



def main(argv):
	""" driver function """

	if argv is None:
		argv = sys.argv

	version = '%prog 0.1'
	usage = 'usage: %prog [options]'

	oparser = optparse.OptionParser(usage=usage, version=version,
		description=__doc__)
	cparser = cfgparse.ConfigParser()
	setup(oparser, cparser, _module)

	options, args = cparser.parse(oparser, argv)

	lgr = log_from_config(options, _module)
	lgr.info('*** %s starting up ***' % _module)

	run_ts = datetime.datetime.now()
	lgr.info('run time: %s' % run_ts)

	# get building ids
	building_ids = utils.parse_value_list(options.building_ids)

	reinstantiate_cparser = False
	for bldg_idx, building_id in enumerate(building_ids):

		if bldg_idx != 0:
			reinstantiate_cparser = True
		options = utils.load_bldg_config(building_id, oparser, cparser, argv,
			options, lgr, reinstantiate_cparser)

		# find newest training covariates file
		latest_trn_files = find_latest_training_files(options, lgr,
			options.temp_dir)

		# lgr.info(latest_trn_files)
		# continue

		all_results = []
		for hour_idx in range(0, 24):

			# retrieve latest covariate file name for hour_idx
			file_mtime, latest_trn_file = latest_trn_files[hour_idx]

			if not latest_trn_file:
				lgr.critical('Training covariates file not found for hour %d' % hour_idx)
				continue

			lgr.info('Using training covariates file %s with last modify time %s' % (
				latest_trn_file, file_mtime))

			# load train covariates
			train_covariates = load_covariates_csv(latest_trn_file)

			# perform grid search
			all_results.append(find_optimal_params(train_covariates,
				options.forecast_granularity, ONE_WEEK, ONE_WEEK, hour_idx,
				options, lgr))

		lgr.info('grid search finished in %s' % (
			datetime.datetime.now() - run_ts))

		# save results to db
		save_grid_srch_results(all_results, run_ts, options, lgr)



if __name__ == '__main__':
	main(sys.argv)

