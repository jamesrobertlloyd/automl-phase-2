#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August-November 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
from sys import argv
import libscores
# FIXME - only needed to make eval statements happy
from libscores import *

import util

# Default I/O directories:
import os
root_dir = os.path.dirname(__file__)
# default_data_dir = os.path.join(root_dir, '..', 'data', 'phase_1_cv_fold_01')

# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'automl_example')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'automl_example_only_rf')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'data_doubling_rf_msl_5')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'data_doubling_rf_msl_1')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'data_doubling_rf_msl_10')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'data_doubling_rf_msl_20')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'cv_rf_msl')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'cv_rf_gbm')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'ft_rf_gbm')

# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'automl')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'automl_rf')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf_gbm')
# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf_gbm_v2')

# default_pred_dir = os.path.join(root_dir, '..', 'predictions', 'test', 'bac_test')


default_data_dir = os.path.join(root_dir, '..', 'data', 'phase_1_cv_fold_05')
default_pred_dir = os.path.join(root_dir, '..', 'predictions', '2015-05-19-5-1', 'madeline', 'fold_05', 'FT')
default_score_dir = os.path.join(root_dir, '..', 'scores', '2015-05-19-5-1', 'fold_05')

# Constants
TEST = 'test'  # TODO - think critically about valid vs test

# Constant used for a missing score
missing_score = -0.999999  # FIXME - why this number exactly?

# Version number
# Change in 0.81: James Robert Lloyd taking control of code
# Change in 0.9 : Very much rewritten - more pythonic and generally nicer
scoring_version = 0.9
    
# =============================== MAIN ========================================
    
if __name__ == '__main__':

    # Get directories from program arguments if available
    if len(argv) > 1:
        data_dir = argv[1]
    else:
        data_dir = default_data_dir
    if len(argv) > 2:
        pred_dir = argv[2]
    else:
        pred_dir = default_pred_dir
    if len(argv) > 3:
        score_dir = argv[3]
    else:
        score_dir = default_score_dir

    # Get names of datasets
    dataset_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    # Name of predictions
    algo_name = os.path.split(pred_dir)[-1]

    # For each dataset
    for dataset_name in dataset_names:
        print('Dataset : %s' % dataset_name)
        # Prepare list of times and performances
        time_score_list = []
        # Is the first post file available?
        firstpost_name = os.path.join(pred_dir, '%s.firstpost' % dataset_name)
        if os.path.isfile(firstpost_name):
            # Record time of firstpost file creation
            start_time = os.path.getmtime(firstpost_name)
            # Makedir if necessary
            util.mkdir(os.path.join(score_dir, dataset_name, algo_name))
            # Load info about this dataset
            info_file = os.path.join(data_dir, dataset_name, '%s_public.info' % dataset_name)
            info = libscores.get_info(info_file)
            # FIXME HACK
            info['metric'] = 'auc_metric'
            # END FIXME HACK
            # Load solution for this dataset
            solution_file = os.path.join(data_dir, dataset_name, '%s_%s.solution' % (dataset_name, TEST))
            solution = libscores.read_array(solution_file)
            # For each set of predictions
            prediction_files = util.ls(os.path.join(pred_dir, '%s_%s_*.predict' % (dataset_name, TEST)))
            for prediction_file in prediction_files:
                # Time of file creation since algorithm start
                file_time = os.path.getmtime(prediction_file) - start_time
                # Open predictions
                prediction = libscores.read_array(prediction_file)
                # Check predictions match shape of solution
                if solution.shape != prediction.shape:
                    raise ValueError("Mismatched prediction shape {} vs. {}".format(prediction.shape, solution.shape))
                # Score
                if info['metric'] == 'r2_metric' or info['metric'] == 'a_metric':
                    # Remove NaN and Inf for regression
                    solution = libscores.sanitize_array(solution)
                    prediction = libscores.sanitize_array(prediction)
                    # TODO - remove eval
                    score = eval(info['metric'] + '(solution, prediction, "' + info['task'] + '")')
                else:
                    # Compute version that is normalized (for classification scores).
                    # This does nothing if all values are already in [0, 1]
                    [csolution, cprediction] = libscores.normalize_array(solution, prediction)
                    # TODO - remove eval
                    score = eval(info['metric'] + '(csolution, cprediction, "' + info['task'] + '")')
                    # score = eval('bac_metric' + '(csolution, cprediction, "' + info['task'] + '")')
                time_score_list.append((file_time, score))
                print('Time  : %7.1f' % file_time + '  Score : %7.5f' % score)
        # Save results
        save_folder = os.path.join(score_dir, dataset_name, algo_name)
        util.mkdir(save_folder)
        with open(os.path.join(save_folder, 'learning_curve.csv'), 'w') as score_file:
            score_file.write('Time,Score\n')
            for (time, score) in time_score_list:
                score_file.write('%s,%s\n' % (time, score))