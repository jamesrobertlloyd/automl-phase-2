from __future__ import division

__author__ = 'James Robert Lloyd'
__description__ = 'Various functions allowing automatic algorithm selection and parameter tuning'

import pybo
import pygp
from pybo.bayesopt import bayesopt
from mwhutils.random import rstate

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.interactive(True)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_hastie_10_2
from sklearn import cross_validation
from sklearn import metrics

import scipy.stats

import cPickle as pickle

import os
import sys
sys.path.append(os.path.dirname(__file__))
import shutil
import time
import gc
import copy
import multiprocessing

from automl_lib import data_io                       # general purpose input/output functions
from automl_lib.data_io import vprint           # print only in verbose mode
from automl_lib.data_manager import DataManager # load/save data and get info about them
from automl_lib.models import MyAutoML, MyAutoML_RF_Only          # example model from scikit learn
import libscores

import freezethaw as ft
import managers

import util


def competition_example(input_dir, output_dir, basename, time_budget):
    verbose = True
    debug_mode = 0
    max_cycle = 50
    # ======== Learning on a time budget:
    # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
    start = time.time()
    print('#########\nStarting!\n#########')

    # ======== Creating a data object with data, informations about it
    vprint(verbose,  "======== Reading and converting data ==========")
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
    print D

    # ======== Keeping track of time
    time_spent = time.time() - start
    vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        execution_success = False
        return

    # ========= Creating a model, knowing its assigned task from D.info['task'].
    # The model can also select its hyper-parameters based on other elements of info.
    vprint( verbose,  "======== Creating model ==========")
    M = MyAutoML(D.info, verbose, debug_mode)
    print M

    # ========= Iterating over learning cycles and keeping track of time
    # Preferably use a method that iteratively improves the model and
    # regularly saves predictions results gradually getting better
    # until the time budget is exceeded.
    # The example model we provide we use just votes on an increasingly
    # large number of "base estimators".
    time_spent = time.time() - start
    vprint( verbose,  "[+] Remaining time after building model %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    time_budget = time_budget - time_spent # Remove time spent so far
    start = time.time()              # Reset the counter
    time_spent = 0                   # Initialize time spent learning
    cycle = 0

    while True:
        vprint( verbose,  "=========== " + basename.capitalize() +" Training cycle " + str(cycle) +" ================")
        # Exponentially scale the number of base estimators
        M.model.n_estimators = int(np.exp2(cycle))
        # Fit base estimators
        M.fit(D.data['X_train'], D.data['Y_train'])
        vprint( verbose,  "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
        # Make predictions
        if 'X_valid' in D.data:
            Y_valid = M.predict(D.data['X_valid'])
        if 'X_test' in D.data:
            Y_test = M.predict(D.data['X_test'])
        vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        # Write results
        if 'X_valid' in D.data:
            filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
        if 'X_test' in D.data:
            filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_test), Y_test)
        vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
        time_spent = time.time() - start
        vprint( verbose,  "[+] End cycle, remaining time %5.2f sec" % (time_budget-time_spent))
        cycle += 1


def competition_example_only_rf(input_dir, output_dir, basename, time_budget):
    verbose = True
    debug_mode = 0
    max_cycle = 50
    # ======== Learning on a time budget:
    # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
    start = time.time()
    print('#########\nStarting!\n#########')

    # ======== Creating a data object with data, informations about it
    vprint(verbose,  "======== Reading and converting data ==========")
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
    print D

    # ======== Keeping track of time
    time_spent = time.time() - start
    vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        execution_success = False
        return

    # ========= Creating a model, knowing its assigned task from D.info['task'].
    # The model can also select its hyper-parameters based on other elements of info.
    vprint( verbose,  "======== Creating model ==========")
    M = MyAutoML_RF_Only(D.info, verbose, debug_mode)
    print M

    # ========= Iterating over learning cycles and keeping track of time
    # Preferably use a method that iteratively improves the model and
    # regularly saves predictions results gradually getting better
    # until the time budget is exceeded.
    # The example model we provide we use just votes on an increasingly
    # large number of "base estimators".
    time_spent = time.time() - start
    vprint( verbose,  "[+] Remaining time after building model %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    time_budget = time_budget - time_spent # Remove time spent so far
    start = time.time()              # Reset the counter
    time_spent = 0                   # Initialize time spent learning
    cycle = 0

    while time_spent <= time_budget * 0.75 and cycle <= max_cycle:
        vprint( verbose,  "=========== " + basename.capitalize() +" Training cycle " + str(cycle) +" ================")
        # Exponentially scale the number of base estimators
        M.model.n_estimators = int(np.exp2(cycle))
        # Fit base estimators
        M.fit(D.data['X_train'], D.data['Y_train'])
        vprint( verbose,  "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
        # Make predictions
        if 'X_valid' in D.data:
            Y_valid = M.predict(D.data['X_valid'])
        if 'X_test' in D.data:
            Y_test = M.predict(D.data['X_test'])
        vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        # Write results
        if 'X_valid' in D.data:
            filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
        if 'X_test' in D.data:
            filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_test), Y_test)
        vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
        time_spent = time.time() - start
        vprint( verbose,  "[+] End cycle, remaining time %5.2f sec" % (time_budget-time_spent))
        cycle += 1


def data_doubling_rf(input_dir, output_dir, basename, time_budget, msl=1):
    """Random forest with fixed parameters but increasing data"""
    verbose = True
    # Start timing
    start = time.time()
    # Load data
    print('Loading data')
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
    print(D)

    # ======== Keeping track of time
    time_spent = time.time() - start
    vprint(verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint(verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    time_budget = time_budget - time_spent # Remove time spent so far
    start = time.time()              # Reset the counter
    time_spent = 0                   # Initialize time spent learning
    cycle = 0

    quit_next_time = False

    while time_spent <= time_budget * 0.75:
        if quit_next_time:
            break
        # Exponentially scale the amount of data
        n_data = int(np.exp2(cycle + 4))
        if n_data >= D.data['X_train'].shape[0]:
            n_data = D.data['X_train'].shape[0]
            quit_next_time = True
        print('n_data = %d' % n_data)
        # Fit estimator to subset of data
        M = RandomForestClassifier(n_estimators=1000, min_samples_leaf=msl, n_jobs=1)
        M.fit(D.data['X_train'][:n_data, :], D.data['Y_train'][:n_data])
        M.fit(D.data['X_train'], D.data['Y_train'])
        vprint( verbose,  "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
        # Make predictions
        if 'X_valid' in D.data:
            Y_valid = M.predict_proba(D.data['X_valid'])[:, 1]
        if 'X_test' in D.data:
            Y_test = M.predict_proba(D.data['X_test'])[:, 1]
        vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        # Write results
        if 'X_valid' in D.data:
            filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
        if 'X_test' in D.data:
            filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_test), Y_test)
        vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
        time_spent = time.time() - start
        vprint( verbose,  "[+] End cycle, remaining time %5.2f sec" % (time_budget-time_spent))
        cycle += 1


def cv_growing_rf(input_dir, output_dir, basename, time_budget, msl=(1,2,4,8,16,32), n_folds=5):
    """Using cross validation to select between different versions of random forest """
    verbose = True
    start = time.time()
    # Load data
    print('Loading data')
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
    print D
    # Splitting data into folds
    folds = cross_validation.KFold(D.data['X_train'].shape[0], n_folds=n_folds)

    # ======== Keeping track of time
    time_spent = time.time() - start
    vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    print('Building models')
    models = [RandomForestClassifier(n_estimators=1, min_samples_leaf=n) for n in msl]

    time_spent = time.time() - start
    vprint( verbose,  "[+] Remaining time after building models %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    time_budget = time_budget - time_spent # Remove time spent so far
    start = time.time()              # Reset the counter
    time_spent = 0                   # Initialize time spent learning
    cycle = 0

    while time_spent <= time_budget * 0.75:
        vprint( verbose,  "=========== " + basename.capitalize() +" Training cycle " + str(cycle) +" ================")
        # Exponentially scale the number of base estimators
        for model in models:
            model.n_estimators = int(np.exp2(cycle))
        # Cross validate models
        cv_scores = np.zeros(len(models))
        for (train, test) in folds:
            for (i, model) in enumerate(models):
                model.fit(D.data['X_train'][train], D.data['Y_train'][train])
                predictions = model.predict_proba(D.data['X_train'][test])[:, -1]
                try:
                    score = metrics.roc_auc_score(D.data['Y_train'][test], predictions)
                except:
                    score = 0
                cv_scores[i] += score
        # Best index is?
        print(cv_scores)
        best_index = np.argmax(cv_scores)
        # Train best model on full data
        model = models[best_index]
        model.fit(D.data['X_train'], D.data['Y_train'])
        # Make predictions
        if 'X_valid' in D.data:
            Y_valid = model.predict_proba(D.data['X_valid'])[:, 1]
        if 'X_test' in D.data:
            Y_test = model.predict_proba(D.data['X_test'])[:, 1]
        vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        # Write results
        if 'X_valid' in D.data:
            filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
        if 'X_test' in D.data:
            filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_test), Y_test)
        vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
        time_spent = time.time() - start
        vprint( verbose,  "[+] End cycle, remaining time %5.2f sec" % (time_budget-time_spent))
        cycle += 1


def cv_growing_rf_gbm(input_dir, output_dir, basename, time_budget, msl=(1, 3, 9, 27), n_folds=5):
    """Using cross validation to select between different versions of random forest """
    verbose = True
    start = time.time()
    # Load data
    print('Loading data')
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
    print D
    # Splitting data into folds
    folds = cross_validation.KFold(D.data['X_train'].shape[0], n_folds=n_folds)

    # ======== Keeping track of time
    time_spent = time.time() - start
    vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    print('Building models')
    models = [RandomForestClassifier(n_estimators=1, min_samples_leaf=n) for n in msl]
    models.append(GradientBoostingClassifier(min_samples_split=2, warm_start=True))
    models.append(GradientBoostingClassifier(min_samples_split=5, warm_start=True))
    models.append(GradientBoostingClassifier(min_samples_split=10, warm_start=True))

    time_spent = time.time() - start
    vprint( verbose,  "[+] Remaining time after building models %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    time_budget = time_budget - time_spent # Remove time spent so far
    start = time.time()              # Reset the counter
    time_spent = 0                   # Initialize time spent learning
    cycle = 0

    while time_spent <= time_budget * 0.75:
        vprint( verbose,  "=========== " + basename.capitalize() +" Training cycle " + str(cycle) +" ================")
        # Exponentially scale the number of base estimators
        for model in models:
            model.n_estimators = int(np.exp2(cycle))
        # Cross validate models
        cv_scores = np.zeros(len(models))
        for (train, test) in folds:
            for (i, model) in enumerate(models):
                model.fit(D.data['X_train'][train], D.data['Y_train'][train])
                predictions = model.predict_proba(D.data['X_train'][test])[:, -1]
                try:
                    score = metrics.roc_auc_score(D.data['Y_train'][test], predictions)
                except:
                    score = 0
                cv_scores[i] += score
        # Best index is?
        print(cv_scores)
        best_index = np.argmax(cv_scores)
        # Train best model on full data
        model = models[best_index]
        model.fit(D.data['X_train'], D.data['Y_train'])
        # Make predictions
        if 'X_valid' in D.data:
            Y_valid = model.predict_proba(D.data['X_valid'])[:, 1]
        if 'X_test' in D.data:
            Y_test = model.predict_proba(D.data['X_test'])[:, 1]
        vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        # Write results
        if 'X_valid' in D.data:
            filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
        if 'X_test' in D.data:
            filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(output_dir,filename_test), Y_test)
        vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
        time_spent = time.time() - start
        vprint( verbose,  "[+] End cycle, remaining time %5.2f sec" % (time_budget-time_spent))
        cycle += 1


def freeze_thaw_cv_rf(input_dir, output_dir, basename, time_budget, msl=(1, 3, 9, 27), n_folds=3, compute_quantum=5,
                      plot=False):
    """Freeze thaw on cross validated random forest with varying min samples leaf"""
    verbose = True
    start = time.time()
    temp_dir = 'temp'
    util.mkdir(temp_dir)
    util.mkdir(output_dir)
    # Load data
    print('Loading data')
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
    print D
    # Splitting data into folds
    folds = cross_validation.KFold(D.data['X_train'].shape[0], n_folds=n_folds)
    # Report time remaining
    time_spent = time.time() - start
    vprint(verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint(verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    print('Building models')
    # other_models = [[GradientBoostingClassifier(n_estimators=1, min_samples_split=n, warm_start=True)
    #            for _ in range(n_folds + 1)]
    #           for n in msl]
    models = [[ft.WarmLearner(RandomForestClassifier(n_estimators=1, min_samples_leaf=n))
               for _ in range(n_folds + 1)]
              for n in msl]
    # models = models + other_models
    pickles = [[os.path.join(temp_dir, util.mkstemp_safe(temp_dir, '.pk'))
                for _ in range(n_folds + 1)]
               for _ in range(len(models))]

    try:

        print('Marshalling models')
        for (model_list, pickle_list) in zip(models, pickles):
            for (model, pickle_filename) in zip(model_list, pickle_list):
                with open(pickle_filename, 'wb') as pickle_file:
                    pickle.dump(model, pickle_file)
        del models
        gc.collect()

        time_spent = time.time() - start
        vprint(verbose,  "[+] Remaining time after building models %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint(verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            return

        time_budget = time_budget - time_spent # Remove time spent so far
        start = time.time()              # Reset the counter
        scores = [[] for _ in range(len(pickles))]
        times = [[] for _ in range(len(pickles))]
        prediction_times = [[] for _ in range(len(pickles))]
        prediction_files = [[] for _ in range(len(pickles))]
        predict_counter = 0
        models_to_be_run = range(len(pickles))

        # Set up freeze thaw parameters
        alpha = 1
        beta = 1
        scale = 1
        log_noise = np.log(0.001)
        x_scale = 1
        x_ell = 0.001
        a = 1
        b = 1
        bounds = [[0.01, 5],
                  [0.01, 5],
                  [0.1, 5],
                  [np.log(0.0000001), np.log(0.01)],
                  [0.1, 10],
                  [0.1, 10],
                  [0.33, 3],
                  [0.33, 3]]

        while time.time() - start + compute_quantum <= time_budget:
            # For the moment just run all algorithms
            for (i, pickle_list) in enumerate(pickles):
                if not i in models_to_be_run:
                    continue
                print(i)
                local_start = time.time()
                # Read models into memory
                print('Reading into memory')
                model_copies = []
                for pickle_filename in pickle_list:
                    with open(pickle_filename, 'rb') as pickle_file:
                        model_copies.append(pickle.load(pickle_file))
                print('Training')
                model_times, model_scores = increase_n_estimators_cv(model_copies,
                                                                     D.data['X_train'], D.data['Y_train'],
                                                                     folds, compute_quantum)
                scores[i] += model_scores
                # Make predictions with this model and save to temp file
                model = model_copies[-1]
                if 'X_valid' in D.data:
                    Y_valid = model.predict_proba(D.data['X_valid'])[:, 1]
                if 'X_test' in D.data:
                    Y_test = model.predict_proba(D.data['X_test'])[:, 1]
                random_string = util.random_string()
                if 'X_valid' in D.data:
                    filename_valid = basename + '_valid_' + random_string + '.predict'
                    data_io.write(os.path.join(temp_dir, filename_valid), Y_valid)
                if 'X_test' in D.data:
                    filename_test = basename + '_test_' + random_string + '.predict'
                    data_io.write(os.path.join(temp_dir, filename_test), Y_test)
                # Put models back onto disk
                print('Saving to disk')
                for (model, pickle_filename) in zip(model_copies, pickle_list):
                    with open(pickle_filename, 'wb') as pickle_file:
                        pickle.dump(model, pickle_file)
                del model_copies
                gc.collect()
                # Save time
                time_taken = time.time() - local_start
                model_total_time = model_times[-1]
                if len(times[i]) == 0:
                    offset = 0
                else:
                    offset = times[i][-1]
                adjusted_times = [offset + a_time * time_taken / model_total_time for a_time in model_times]
                times[i] += adjusted_times
                # Save adjusted time corresponding to prediction
                prediction_times[i].append(times[i][-1])
                prediction_files[i].append(random_string)
            # Print data in MATLAB format
            print('N = %d;' % len(times))
            print('''
t = cell(N, 1);
t_star = cell(N, 1);
y = cell(N, 1);
x = nan(N, 1);
true_mu = nan(N, 1);
''')
            t_star = [range(1, 50, 1)] * len(pickles)
            # for i in range(len(scores)):
            #     print('x(%d) = %f;' % (i + 1, msl[i]))
            #     print('t{%d} = [' % (i + 1) + ';'.join(['%f' % t for t in times[i]]) + '];')
            #     print('t_star{%d} = [' % (i + 1) + ';'.join(['%f' % t for t in t_star[i]]) + '];')
            #     # print("t_star{%d} = linspace(max(t{%d}), t_max, 50)';" % (i + 1, i + 1))
            #     print('y{%d} = [' % (i + 1) + ';'.join(['%f' % s for s in scores[i]]) + '];')
            # Run freeze thaw on data
            print('Thinking')
            t_kernel = ft.ft_K_t_t_plus_noise
            x_kernel = ft.cov_iid
            x = 0 * np.ones((len(pickles), 1))
            m = np.zeros((len(pickles), 1))
            t_star = []
            for i in range(len(pickles)):
                t_star.append(np.linspace(times[i][-1], times[i][-1] + time_budget - (time.time() - start), 50))
            # Sample parameters
            # xx = [alpha, beta, scale, noise, x_scale, x_ell, a, b]
            xx = [alpha, beta, scale, log_noise, x_scale]
            logdist = lambda xx: ft.ft_ll(m, times, scores, x, x_kernel, dict(scale=xx[4]), t_kernel,
                                          dict(scale=xx[2], alpha=xx[0], beta=xx[1], log_noise=xx[3]))
            xx = ft.slice_sample_bounded_max(1, 10, logdist, xx, 0.5, True, 10, bounds)[0]
            alpha = xx[0]
            beta = xx[1]
            scale = xx[2]
            log_noise = xx[3]
            x_scale = xx[4]
            # print(log_noise)
            logdist(xx)
            # Setup params
            x_kernel_params = dict(scale=x_scale)
            t_kernel_params = dict(scale=scale, alpha=alpha, beta=beta, log_noise=log_noise)
            # TODO - Subset data and do parameter inference!
            y_mean, y_covar = ft.ft_posterior(m, times, scores, t_star, x, x_kernel, x_kernel_params, t_kernel, t_kernel_params)
            # Also compute posterior for already computed predictions
            # FIXME - what if prediction times has empty lists
            predict_mean, _ = ft.ft_posterior(m, times, scores, prediction_times, x, x_kernel, x_kernel_params, t_kernel, t_kernel_params)
            # Identify predictions thought to be the best currently
            best_mean = -np.inf
            best_model_index = None
            best_time_index = None
            for i in range(len(pickles)):
                if max(predict_mean[i]) > best_mean:
                    best_mean = max(predict_mean[i])
                    best_model_index = i
                    best_time_index = np.argmax(np.array(predict_mean[i]))
            print('Best model index : %d' % best_model_index)
            print('Best time : %f' % prediction_times[best_model_index][best_time_index])
            print('Estimated performance : %f' % predict_mean[best_model_index][best_time_index])
            # Save these predictions to the output dir
            print('Saving predictions')
            predict_counter += 1
            if 'X_valid' in D.data:
                shutil.copy(os.path.join(temp_dir,
                                         basename + '_valid_' + \
                                         prediction_files[best_model_index][best_time_index] + '.predict'),
                            os.path.join(output_dir,
                                         basename + '_valid_' + str(predict_counter).zfill(3) + '.predict'))
            if 'X_test' in D.data:
                shutil.copy(os.path.join(temp_dir,
                                         basename + '_test_' + \
                                         prediction_files[best_model_index][best_time_index] + '.predict'),
                            os.path.join(output_dir,
                                         basename + '_test_' + str(predict_counter).zfill(3) + '.predict'))
            # Pick best candidate to run next
            best_model_index = -1
            best_current_value = -np.inf
            for i in range(len(pickles)):
                if y_mean[i][0] > best_current_value:
                    best_current_value = y_mean[i][0]
                    best_model_index = i
            best_model_index = -1
            best_acq_fn = -np.inf
            for i in range(len(pickles)):
                mean = y_mean[i][-1]
                std = np.sqrt(y_covar[i][-1, -1] - np.exp(log_noise))
                acq_fn = ft.trunc_norm_mean_upper_tail(a=best_current_value, mean=mean, std=std) - best_current_value
                print(acq_fn)
                if acq_fn > best_acq_fn:
                    best_acq_fn = acq_fn
                    best_model_index = i
            models_to_be_run = [best_model_index]
            print('Selecting model index : %d' % best_model_index)
            # Plot curves
            if plot:
                # TODO - Make this save to temp directory
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title('Learning curves')
                ax.set_xlabel('Time (seconds)')
                ax.set_xscale('log')
                ax.set_ylabel('Score')
                for i in range(len(scores)):
                    ax.plot(times[i], scores[i],
                            color=util.colorbrew(i),
                            linestyle='dashed', marker='o',
                            label=str(i))
                    ax.plot(t_star[i], y_mean[i],
                            color=util.colorbrew(i),
                            linestyle='-', marker='',
                            label=str(i))
                    ax.fill_between(t_star[i], y_mean[i].ravel() - np.sqrt(np.diag(y_covar[i]) - np.exp(log_noise)),
                                               y_mean[i].ravel() + np.sqrt(np.diag(y_covar[i]) - np.exp(log_noise)),
                                    color=util.colorbrew(i),
                                    alpha=0.2)
                leg = ax.legend(loc='best')
                leg.get_frame().set_alpha(0.5)
                plt.show()

    finally:
        for pickle_list in pickles:
            for pickle_filename in pickle_list:
                os.remove(pickle_filename)
        for prediction_file_list in prediction_files:
            for random_string in prediction_file_list:
                if 'X_valid' in D.data:
                    filename_valid = basename + '_valid_' + random_string + '.predict'
                    os.remove(os.path.join(temp_dir, filename_valid))
                if 'X_test' in D.data:
                    filename_test = basename + '_test_' + random_string + '.predict'
                    os.remove(os.path.join(temp_dir, filename_test))


def freeze_thaw_cv_rf_gbm(input_dir, output_dir, basename, time_budget,
                          msl=(1, 3, 9, 27), mss=(1, 3, 9, 27), n_folds=3, compute_quantum=5,
                          plot=False, n_cpu=1, trees_per_compute=1):
    """Freeze thaw on cross validated random forest and gbm"""
    verbose = True
    start = time.time()
    temp_dir = 'temp'
    util.mkdir(temp_dir)
    util.mkdir(output_dir)
    # Load data
    print('Loading data')
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
    # Convert sparse data to dense
    try:
        D.data['X_train'] = D.data['X_train'].toarray()
        D.data['X_valid'] = D.data['X_valid'].toarray()
        D.data['X_test'] = D.data['X_test'].toarray()
    except:
        # FIXME
        print('This should really be an if statement')
    print D
    # Splitting data into folds
    folds = cross_validation.KFold(D.data['X_train'].shape[0], n_folds=n_folds)
    # Report time remaining
    time_spent = time.time() - start
    vprint(verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
    if time_spent >= time_budget:
        vprint(verbose,  "[-] Sorry, time budget exceeded, skipping this task")
        return

    print('Building models')
    gbm_models = [[GradientBoostingClassifier(n_estimators=1, min_samples_split=n, warm_start=True)
                   for _ in range(n_folds + 1)]
                  for n in mss]
    rf_models = [[ft.WarmLearner(RandomForestClassifier(n_estimators=n_cpu*trees_per_compute, min_samples_leaf=n,
                                                        n_jobs=n_cpu))
                  for _ in range(n_folds + 1)]
                 for n in msl]
    all_models = [gbm_models, rf_models]
    gbm_pickles = [[os.path.join(temp_dir, util.mkstemp_safe(temp_dir, '.pk'))
                    for _ in range(n_folds + 1)]
                   for _ in range(len(gbm_models))]
    rf_pickles = [[os.path.join(temp_dir, util.mkstemp_safe(temp_dir, '.pk'))
                   for _ in range(n_folds + 1)]
                  for _ in range(len(rf_models))]
    all_pickles = [gbm_pickles, rf_pickles]
    # Parameters on log scale feels sensible
    rf_x = list(np.log(value) for value in msl)
    gbm_x = list(np.log(value) for value in mss)
    x = [gbm_x, rf_x]

    try:

        print('Marshalling models')
        for models, pickles in zip(all_models, all_pickles):
            for (model_list, pickle_list) in zip(models, pickles):
                for (model, pickle_filename) in zip(model_list, pickle_list):
                    with open(pickle_filename, 'wb') as pickle_file:
                        pickle.dump(model, pickle_file)
            del models
        gc.collect()

        time_spent = time.time() - start
        vprint(verbose,  "[+] Remaining time after building models %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint(verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            return

        time_budget = time_budget - time_spent # Remove time spent so far
        start = time.time()              # Reset the counter

        predict_counter = 0

        scores = []
        times = []
        prediction_times = []
        prediction_files = []
        models_to_be_run = []

        alpha = []
        beta = []
        scale = []
        log_noise = []
        x_scale = []
        x_ell = []
        a = []
        b = []

        for pickles in all_pickles:
            scores.append([[] for _ in range(len(pickles))])
            times.append([[] for _ in range(len(pickles))])
            prediction_times.append([[] for _ in range(len(pickles))])
            prediction_files.append([[] for _ in range(len(pickles))])
            models_to_be_run.append(range(len(pickles)))

            # Set up freeze thaw parameters
            alpha.append(3)
            beta.append(1)
            scale.append(1)
            log_noise.append(np.log(0.0001))
            x_scale.append(1)
            x_ell.append(0.001)
            a.append(1)
            b.append(1)

        bounds = [[2, 4],
                  [0.01, 5],
                  [0.1, 5],
                  [np.log(0.0000001), np.log(0.001)],
                  [0.1, 10],
                  [0.1, 10],
                  [0.33, 3],
                  [0.33, 3]]

        while time.time() - start + compute_quantum <= time_budget:
            for (j, pickles) in enumerate(all_pickles):
                for (i, pickle_list) in enumerate(pickles):
                    if not i in models_to_be_run[j]:
                        continue
                    print(j)
                    print(i)
                    local_start = time.time()
                    # Read models into memory
                    print('Reading into memory')
                    model_copies = []
                    for pickle_filename in pickle_list:
                        with open(pickle_filename, 'rb') as pickle_file:
                            model_copies.append(pickle.load(pickle_file))
                    print('Training')
                    # model_times, model_scores = increase_n_estimators_cv(model_copies,
                    #                                                      D.data['X_train'], D.data['Y_train'],
                    #                                                      folds, compute_quantum)
                    model_times, model_scores, cut_off = increase_n_estimators_cv(model_copies,
                                                                         D.data['X_train'], D.data['Y_train'],
                                                                         folds, compute_quantum)
                    # Add some jitter to make the GPs happier
                    # FIXME - this can be fixed with better modelling assumptions
                    for k in range(len(model_scores)):
                        model_scores[k] += 0.0005 * np.random.normal()
                    scores[j][i] += model_scores
                    # Make predictions with this model and save to temp file
                    model = model_copies[-1]
                    if 'X_valid' in D.data:
                        Y_valid = model.predict_proba(D.data['X_valid'])[:, 1]
                        Y_valid = scaled_cut_off(Y_valid, cut_off)
                    if 'X_test' in D.data:
                        Y_test = model.predict_proba(D.data['X_test'])[:, 1]
                        Y_test = scaled_cut_off(Y_test, cut_off)
                    random_string = util.random_string()
                    if 'X_valid' in D.data:
                        filename_valid = basename + '_valid_' + random_string + '.predict'
                        data_io.write(os.path.join(temp_dir, filename_valid), Y_valid)
                    if 'X_test' in D.data:
                        filename_test = basename + '_test_' + random_string + '.predict'
                        data_io.write(os.path.join(temp_dir, filename_test), Y_test)
                    # Put models back onto disk
                    print('Saving to disk')
                    for (model, pickle_filename) in zip(model_copies, pickle_list):
                        with open(pickle_filename, 'wb') as pickle_file:
                            pickle.dump(model, pickle_file)
                    del model_copies
                    gc.collect()
                    # Save time
                    time_taken = time.time() - local_start
                    model_total_time = model_times[-1]
                    if len(times[j][i]) == 0:
                        offset = 0
                    else:
                        offset = times[j][i][-1]
                    adjusted_times = [offset + a_time * time_taken / model_total_time for a_time in model_times]
                    times[j][i] += adjusted_times
                    # Save adjusted time corresponding to prediction
                    prediction_times[j][i].append(times[j][i][-1])
                    prediction_files[j][i].append(random_string)
                for i in range(len(scores[j])):
                    print('t{%d} = [' % (i + 1) + ';'.join(['%f' % t for t in times[j][i]]) + '];')
                    # print("t_star{%d} = linspace(max(t{%d}), t_max, 50)';" % (i + 1, i + 1))
                    print('y{%d} = [' % (i + 1) + ';'.join(['%f' % s for s in scores[j][i]]) + '];')
            print('Thinking')
            y_mean = [None] * len(all_pickles)
            y_covar = [None] * len(all_pickles)
            predict_mean = [None] * len(all_pickles)
            t_star = [None] * len(all_pickles)
            remaining_time = time_budget - (time.time() - start)
            for (j, pickles) in enumerate(all_pickles):
                # Run freeze thaw on data
                t_kernel = ft.ft_K_t_t_plus_noise
                # x_kernel = ft.cov_iid
                x_kernel = ft.cov_matern_5_2
                # m = np.zeros((len(pickles), 1))
                m = 0.5 * np.ones((len(pickles), 1))
                t_star[j] = []
                # Subsetting data
                times_subset = copy.deepcopy(times[j])
                scores_subset = copy.deepcopy(scores[j])
                for i in range(len(pickles)):
                    if len(times_subset[i]) > 50:
                        times_subset[i] = list(np.array(times_subset[i])[[int(np.floor(k))
                                                                  for k in np.linspace(0, len(times_subset[i]) - 1, 50)[1:]]])
                        scores_subset[i] = list(np.array(scores_subset[i])[[int(np.floor(k))
                                                                    for k in np.linspace(0, len(scores_subset[i]) - 1, 50)[1:]]])
                    # print(len(times_subset[i]))
                for i in range(len(pickles)):
                    t_star[j].append(np.linspace(times[j][i][-1], times[j][i][-1] + remaining_time, 50))
                # Sample parameters
                xx = [alpha[j], beta[j], scale[j], log_noise[j], x_scale[j], x_ell[j]]
                # logdist = lambda xx: ft.ft_ll(m, times_subset, scores_subset, x[j], x_kernel, dict(scale=xx[4]), t_kernel,
                #                               dict(scale=xx[2], alpha=xx[0], beta=xx[1], log_noise=xx[3]))
                logdist = lambda xx: ft.ft_ll(m, times_subset, scores_subset, x[j], x_kernel, dict(scale=xx[4], ell=xx[5]), t_kernel,
                                              dict(scale=xx[2], alpha=xx[0], beta=xx[1], log_noise=xx[3]))
                xx = ft.slice_sample_bounded_max(1, 10, logdist, xx, 0.5, True, 10, bounds)[0]
                alpha[j] = xx[0]
                beta[j] = xx[1]
                scale[j] = xx[2]
                log_noise[j] = xx[3]
                x_scale[j] = xx[4]
                x_ell[j] = xx[5]
                print(xx)
                # Setup params
                x_kernel_params = dict(scale=x_scale[j], ell=x_ell)
                t_kernel_params = dict(scale=scale[j], alpha=alpha[j], beta=beta[j], log_noise=log_noise[j])
                y_mean[j], y_covar[j] = ft.ft_posterior(m, times_subset, scores_subset, t_star[j], x[j], x_kernel, x_kernel_params, t_kernel, t_kernel_params)
                # Also compute posterior for already computed predictions
                # FIXME - what if prediction times has empty lists
                predict_mean[j], _ = ft.ft_posterior(m, times_subset, scores_subset, prediction_times[j], x[j], x_kernel, x_kernel_params, t_kernel, t_kernel_params)
            # Identify predictions thought to be the best currently
            best_mean = -np.inf
            best_model_index = None
            best_time_index = None
            best_pickle_index = None
            for (j, pickles) in enumerate(all_pickles):
                for i in range(len(pickles)):
                    if max(predict_mean[j][i]) >= best_mean:
                        best_mean = max(predict_mean[j][i])
                        best_model_index = i
                        best_pickle_index = j
                        best_time_index = np.argmax(np.array(predict_mean[j][i]))
            print('Best pickle index : %d' % best_pickle_index)
            print('Best model index : %d' % best_model_index)
            print('Best time : %f' % prediction_times[best_pickle_index][best_model_index][best_time_index])
            print('Estimated performance : %f' % predict_mean[best_pickle_index][best_model_index][best_time_index])
            # Save these predictions to the output dir
            print('Saving predictions')
            if 'X_valid' in D.data:
                shutil.copy(os.path.join(temp_dir,
                                         basename + '_valid_' + \
                                         prediction_files[best_pickle_index][best_model_index][best_time_index] + '.predict'),
                            os.path.join(output_dir,
                                         basename + '_valid_' + str(predict_counter).zfill(3) + '.predict'))
            if 'X_test' in D.data:
                shutil.copy(os.path.join(temp_dir,
                                         basename + '_test_' + \
                                         prediction_files[best_pickle_index][best_model_index][best_time_index] + '.predict'),
                            os.path.join(output_dir,
                                         basename + '_test_' + str(predict_counter).zfill(3) + '.predict'))
            predict_counter += 1
            # Pick best candidate to run next
            best_current_value = best_mean
            best_pickle_index = None
            best_model_index = None
            best_acq_fn = -np.inf
            for (j, pickles) in enumerate(all_pickles):
                for i in range(len(pickles)):
                    mean = y_mean[j][i][-1]
                    std = np.sqrt(y_covar[j][i][-1, -1] - np.exp(log_noise[j]))
                    acq_fn = ft.trunc_norm_mean_upper_tail(a=best_current_value, mean=mean, std=std) - best_current_value
                    if acq_fn >= best_acq_fn:
                        best_acq_fn = acq_fn
                        best_model_index = i
                        best_pickle_index = j
            models_to_be_run = []
            for _ in range(len(all_pickles)):
                models_to_be_run.append([])
            models_to_be_run[best_pickle_index] = [best_model_index]
            print('Selecting pickle index : %d' % best_pickle_index)
            print('Selecting model index : %d' % best_model_index)
            # Plot curves
            if plot:
                # TODO - Make this save to temp directory
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title('Learning curves')
                ax.set_xlabel('Time (seconds)')
                ax.set_xscale('log')
                ax.set_ylabel('Score')
                label_count = 0
                for j in range(len(all_pickles)):
                    for i in range(len(scores[j])):
                        ax.plot(times[j][i], scores[j][i],
                                color=util.colorbrew(label_count),
                                linestyle='dashed', marker='o',
                                label=str(label_count))
                        ax.plot(t_star[j][i], y_mean[j][i],
                                color=util.colorbrew(label_count),
                                linestyle='-', marker='')
                        ax.fill_between(t_star[j][i], y_mean[j][i].ravel() - np.sqrt(np.diag(y_covar[j][i]) - np.exp(log_noise[j])),
                                                   y_mean[j][i].ravel() + np.sqrt(np.diag(y_covar[j][i]) - np.exp(log_noise[j])),
                                        color=util.colorbrew(label_count),
                                        alpha=0.2)
                        label_count += 1
                leg = ax.legend(loc='best')
                leg.get_frame().set_alpha(0.5)
                plt.show()

    finally:
        for (j, pickles) in enumerate(all_pickles):
            for pickle_list in pickles:
                for pickle_filename in pickle_list:
                    os.remove(pickle_filename)
            for prediction_file_list in prediction_files[j]:
                for random_string in prediction_file_list:
                    if 'X_valid' in D.data:
                        filename_valid = basename + '_valid_' + random_string + '.predict'
                        os.remove(os.path.join(temp_dir, filename_valid))
                    if 'X_test' in D.data:
                        filename_test = basename + '_test_' + random_string + '.predict'
                        os.remove(os.path.join(temp_dir, filename_test))


def scaled_cut_off(probs, cut_off):
    """Scale probabilities to be balanced around 0 - 0.5 - 1 using cut_off"""
    scaled = np.copy(probs)
    scaled[probs <= cut_off] = probs[probs <= cut_off] * 0.5 / cut_off
    scaled[probs > cut_off] = 1 - (1 - probs[probs > cut_off]) * 0.5 / (1 - cut_off)
    return scaled


def linspace_sublist(a_list, n):
    """Linearly spaced sub list"""
    return list(np.array(a_list)[[int(np.floor(k)) for k in np.linspace(0, len(a_list) - 1, n)]])


def increase_n_estimators_cv(model_copies, X, y, folds, time_budget, n_samples=25):
    """Cross validate models while increasing n_estimators as much as possible within time budget"""
    start = time.time()
    predict_time = time_budget / n_samples
    # print('pt = %s' % predict_time)
    times = []
    scores = []
    # Increase number of estimators until time limit reached
    while time.time() - start < time_budget:
        for (i, (train, _)) in enumerate(folds):
            model = model_copies[i]
            model.n_estimators += 1
            model.fit(X[train, :], y[train])
        model = model_copies[-1]
        model.n_estimators += 1
        model.fit(X, y)
        # print(time.time() - start)
        if time.time() - start > predict_time:
            # Cross validate
            score = 0
            n = 0
            predictions_list = []
            y_test_list = []
            for (i, (_, test)) in enumerate(folds):
                predictions = model_copies[i].predict_proba(X[test, :])[:, -1]
                predictions_list.append(predictions)
                y_test_list.append(y[test])
                try:
                    # metrics.roc_auc_score(y[test], predictions)
                    score += metrics.roc_auc_score(y[test], predictions)
                    # # Determine optimal cut-off for BAC metric
                    # # FIXME - only do this when necessary
                    # bac_time = time.time()
                    # cut_offs = linspace_sublist(predictions, 10)
                    # best_bac = -np.inf
                    # best_cut_off = None
                    # for cut_off in cut_offs:
                    #     bac = libscores.bac_metric(y[test], scaled_cut_off(predictions, cut_off))
                    #     if bac > best_bac:
                    #         best_bac = bac
                    #         best_cut_off = cut_off
                    # print('Best cut off = %f' % best_cut_off)
                    # bac_time = time.time() - bac_time
                    # print('Time to optimise cut off = %f' % bac_time)
                    # score += best_bac
                except:
                    score += 0
                n += 1

            score = score / n
            times.append(time.time() - start)
            scores.append(score)
            if len(scores) == n_samples:
                break
            # Next time at which to make a prediction
            predict_time = time.time() - start + (time_budget - (time.time() - start)) / (n_samples - len(scores))
            # print('pt = %s' % predict_time)

    bac_time = time.time()
    cut_offs = linspace_sublist(sorted(predictions), 20)
    cut_offs.append(0.44)
    cut_offs.append(0.46)
    cut_offs.append(0.48)
    cut_offs.append(0.50)
    cut_offs.append(0.52)
    cut_offs.append(0.54)
    cut_offs.append(0.56)
    best_bac = -np.inf
    best_cut_off = None
    for cut_off in cut_offs:
        if cut_off > 0 and cut_off < 1:
            bac = 0
            for (y_test, predictions) in zip(y_test_list, predictions_list):
                bac += libscores.bac_metric(y_test, scaled_cut_off(predictions, cut_off))
            bac = bac / len(predictions_list)
            if bac > best_bac:
                best_bac = bac
                best_cut_off = cut_off
    print('Best cut off = %f' % best_cut_off)
    bac_time = time.time() - bac_time
    print('Time to optimise cut off = %f' % bac_time)
    # score = best_bac

    # return times, scores
    return times, scores, best_cut_off


def automl_phase_0(input_dir, output_dir, basename, time_budget):
    # Check type of data
    D = DataManager(basename, input_dir, verbose=True, only_info=True)
    if D.info['task'] == 'binary.classification':
        compute_quantum = D.info['time_budget'] / (10 * 8)
        freeze_thaw_cv_rf_gbm(input_dir, output_dir, basename, time_budget,
                              msl=(1, 3, 9, 27), mss=(1, 3, 9, 27), n_folds=3, compute_quantum=compute_quantum,
                              plot=False, n_cpu=1, trees_per_compute=1)
    else:
        competition_example(input_dir, output_dir, basename, time_budget)


def automl_phase_1(input_dir, output_dir, basename, time_budget):
    manager = managers.FreezeThawManager(input_dir, output_dir, basename, time_budget)
    manager.init()
    manager.run()


if __name__ == '__main__':
    # input_dir = "/scratch/home/Research/auto-stat/automl-2015/src/../data/dsss_bin_class_fold_01"
    output_dir = "/scratch/home/Research/auto-stat/automl-2015/src/../predictions/test/dorothea_test"
    input_dir = "/scratch/home/Research/auto-stat/automl-2015/src/../data/phase_0"
    data_name = "dorothea"
    # data_name = "coil2000"
    # data_name = "magic"
    # time_budget = 299.999944925
    # input_dir = "/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/AutoStat/automl-2015/src/../data/dss_bin_class_fold_01"
    # output_dir = "/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/AutoStat/automl-2015/src/../predictions/test/ft_cv_rf_gbm"
    # data_name = "arcene"
    # data_name = "banana"
    # data_name = "ring"
    time_budget = 60 * 60 * 3
    # time_budget = 50
    # cv_growing_rf(input_dir, output_dir, data_name, time_budget)
    # freeze_thaw_cv_rf(input_dir, output_dir, data_name, time_budget, plot=True, compute_quantum=5)
    # freeze_thaw_cv_rf_gbm(input_dir, output_dir, data_name, time_budget, plot=True, compute_quantum=15, n_cpu=1)
    # freeze_thaw_cv_rf_gbm(input_dir, output_dir, data_name, time_budget, plot=True, compute_quantum=60,
    #                       n_cpu=8, trees_per_compute=10)
    freeze_thaw_cv_rf_gbm(input_dir, output_dir, data_name, time_budget, plot=True, compute_quantum=15,
                          n_cpu=1, trees_per_compute=1, mss=(1,3,9), msl=(1,3,9))