from __future__ import division
import functools
import util

__author__ = 'James Robert Lloyd, Emma Smith'
__description__ = 'Objects that model learner performance and make decisions'

# import time
import copy
# import os
# import cPickle as pickle
from collections import defaultdict
import psutil
# import multiprocessing
import logging
import random
import time as time_module

# set up logging for metalearner module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from agent import Agent
# import util
import freezethaw as ft
import constants

from stackcombiner import StackCombiner
import libscores


class ForgetfulGreedy(Agent):
    """Recommends the current best performing algorithm"""

    def __init__(self, **kwargs):
        super(ForgetfulGreedy, self).__init__(**kwargs)

        self.learner_score_values = dict()
        self.learner_score_times = dict()
        self.learner_prediction_times = dict()

        self.communication_sleep = 1

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
                self.standard_responses(message)
                if message['subject'] == 'scores':
                    # Update local copies of scores
                    self.learner_score_values = message['learner_score_values']
                    self.learner_score_times = message['learner_score_times']
                    # FIXME - assumes test data always available
                    self.learner_prediction_times = message['learner_test_prediction_times']
            except (IndexError, AttributeError):
                pass

    def first_action(self):
        # No reason to start paused at the moment
        self.state = 'running'
        p = psutil.Process()
        current_cpus = p.cpu_affinity()
        if len(current_cpus) > 1:
            p.cpu_affinity([current_cpus[1]])  # meta learning only happens on the second CPU

    def next_action(self):
        # Check mail
        self.read_messages()
        # If running, form an opinion of the best learner and tell parent
        if self.state == 'running':
            names_scores = []
            data = False
            for name in self.learner_score_times.iterkeys():
                if len(self.learner_score_values[name]) > 0:
                    score = self.learner_score_values[name][-1]
                    data = True
                else:
                    score = -np.inf
                names_scores.append((name, score))
            names_scores.sort(key=lambda x: x[1])
            sorted_names = [name for (name, score) in names_scores]
            if data:
                self.send_to_parent(dict(subject='preference', sender=self.name, preference=sorted_names))


class IndependentFreezeThaw(Agent):
    """Assumes exponential mixture decays of all learning curves modelled separately"""

    def __init__(self, **kwargs):
        super(IndependentFreezeThaw, self).__init__(**kwargs)

        self.remaining_time = None

        self.learner_score_values = dict()
        self.learner_score_times = dict()
        self.learner_prediction_times = dict()

        self.communication_sleep = 1
        # self.learner_names = []

        self.scores = defaultdict(list)
        self.times = defaultdict(list)
        self.times_subset = defaultdict(list)
        self.scores_subset = defaultdict(list)
        self.t_star = defaultdict(list)

        self.alpha = defaultdict(functools.partial(util.identity, 3))
        self.beta = defaultdict(functools.partial(util.identity, 1))
        self.scale = defaultdict(functools.partial(util.identity, 1))
        self.log_noise = defaultdict(functools.partial(util.identity, np.log(0.1)))
        self.x_scale = defaultdict(functools.partial(util.identity, 2))
        self.x_ell = defaultdict(functools.partial(util.identity, 0.001))  # These are currently unused
        self.a = defaultdict(functools.partial(util.identity, 1))  # These are currently unused
        self.b = defaultdict(functools.partial(util.identity, 1))  # These are currently unused

        self.y_mean = defaultdict(list)
        self.y_covar = defaultdict(list)
        self.predict_mean = defaultdict(list)
        self.y_samples = defaultdict(list)

        self.compute_quantum = None

        # TODO - think about the bounds on alpha and beta more critically! Transform them to make them scale free?
        self.bounds = [[1, 5],
                       [0.1, 5],
                       [0, np.inf],
                       [np.log(0.0000001), np.inf],
                       [0.1, 10],
                       [0.1, 10],
                       [0.33, 3],
                       [0.33, 3]]

        # self.t_star = dict()
        self.waiting = True

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
            except (IndexError, AttributeError):
                break
            else:
                self.standard_responses(message)
                if message['subject'] == 'scores':
                    self.waiting = False
                    # Update local copies of scores
                    self.learner_score_values = message['learner_score_values']
                    self.learner_score_times = message['learner_score_times']
                    # FIXME - assumes test data always available
                    self.learner_prediction_times = message['learner_test_prediction_times']
                    self.remaining_time = message['remaining_time']
                    self.compute_quantum = message['compute_quantum']

                    # print(self.remaining_time)
                    # FIXME - this is a bit of a hack to initialise everything here
                    # FIXME - and to use lists where dictionaries would be more appropriate
                    # if len(self.learner_names) == 0:
                    #     # Initialise various things
                    #     self.learner_names = sorted(list(self.learner_score_values.iterkeys()))

    def first_action(self):
        # No reason to start paused at the moment
        self.state = 'running'
        p = psutil.Process()
        current_cpus = p.cpu_affinity()
        if len(current_cpus) > 1:
            p.cpu_affinity([current_cpus[1]])  # meta learning only happens on the second CPU
        self.waiting = False

    def next_action(self):
        start_time = time_module.clock()
        # Check mail
        self.read_messages()
        # If running, form an opinion of the best learner and tell parent
        if self.state == 'running':
            if not self.waiting:
                # Check to see if we have sufficient data yet
                # data = False
                # for name in self.learner_score_times.iterkeys():
                #     if len(self.learner_score_values[name]) > 0:
                #         data = True
                learning_curve_data = []
                all_data = []
                for name in self.learner_score_times.iterkeys():
                    if len(self.learner_score_values[name]) > 1:
                        learning_curve_data += self.learner_score_values[name]
                    all_data += self.learner_score_values[name]
                if len(learning_curve_data) > 5:
                    names_scores = []
                    grand_mean = sum(all_data) / len(all_data)
                    grand_scale = max(all_data) - min(all_data)
                    if grand_scale == 0:
                        grand_mean = 1
                        grand_scale = 1
                    # Scale all the data
                    for name in self.learner_score_times.iterkeys():
                        self.learner_score_values[name] = [(value - grand_mean) / grand_scale
                                                           for value in self.learner_score_values[name]]
                    for name in self.learner_score_times.iterkeys():
                        # Need more than one data point to start predicting anything useful
                        if len(self.learner_score_values[name]) > 1:
                            # Set param defaults based on data
                            data_range = np.max(self.learner_score_values[name]) -\
                                         np.min(self.learner_score_values[name])
                            if data_range <= 0 or len(self.learner_score_values[name]) == 0:
                                # Limited data for this child - get data from all other children
                                max_value = -np.inf
                                min_value = np.inf
                                for child in self.learner_score_times.iterkeys():
                                    if len(self.learner_score_values[child]) > 0:
                                        max_value = max(max_value, np.max(self.learner_score_values[child]))
                                        min_value = min(min_value, np.min(self.learner_score_values[child]))
                                data_range = max_value - min_value
                            self.scale.default_factory = functools.partial(util.identity, data_range * data_range)
                            # Scale / 10 std deviation heuristic
                            self.log_noise.default_factory = functools.partial(util.identity,
                                                                               np.log(data_range * data_range / 100))
                            self.bounds[2][-1] = data_range * data_range * 9
                            self.bounds[3][-1] = np.log(data_range * data_range)
                            self.bounds[3][0] = np.log(data_range * data_range / 40000)
                            # Also set the default factories
                            self.scale.default_factory = functools.partial(util.identity, data_range * data_range)
                            self.log_noise.default_factory = functools.partial(util.identity,
                                                                               np.log(0.1 * data_range * data_range))
                            # print(data_range)
                            # print(np.log(data_range * data_range / 100))
                            # print(data_range / 10)
                            # Exponential mixture kernel inference
                            t_kernel = ft.ft_K_t_t_plus_noise
                            x_kernel = ft.cov_matern_5_2
                            m = [np.mean(self.learner_score_values[name])]
                            x = [0]
                            # Subsetting data
                            self.times_subset[name] = list(copy.deepcopy(self.learner_score_times[name]))
                            self.scores_subset[name] = list(copy.deepcopy(self.learner_score_values[name]))
                            # Add some jitter to make the GPs happier
                            # FIXME - should not need this really
                            for i in range(len(self.scores_subset[name])):
                                # self.scores_subset[name][i] += 0.001 * np.random.normal()
                                self.scores_subset[name][i] += (data_range / 200) * np.random.normal()
                            if len(self.times_subset[name]) > 50:
                                indices = [int(np.floor(k))
                                           for k in np.linspace(0, len(self.times_subset[name]) - 1, 50)[1:]]
                                self.times_subset[name] = list(np.array(self.times_subset[name])[indices])
                                self.scores_subset[name] = list(np.array(self.scores_subset[name])[indices])
                            # self.t_star[name] = np.linspace(self.learner_score_times[name][-1],
                            #                                 self.learner_score_times[name][-1] + self.remaining_time, 50)
                            self.t_star[name] = copy.deepcopy(self.learner_prediction_times[name])
                            if len(self.learner_prediction_times[name]) > 0:
                                current_time = self.learner_prediction_times[name][-1]
                            else:
                                current_time = 0
                            additional_time = self.compute_quantum
                            added_points = 0
                            while additional_time < self.remaining_time:
                                self.t_star[name].append(additional_time + current_time)
                                # additional_time *= 2
                                additional_time += self.compute_quantum
                                added_points += 1
                                if added_points >= 10:
                                    break
                            self.t_star[name] = np.array(self.t_star[name])
                            # print(util.is_sorted(self.learner_score_times[name]))
                            # print(self.learner_score_times[name])
                            # print(self.t_star[name])
                            # print(self.remaining_time)
                            # Sample parameters
                            self.y_samples[name] = []
                            for _ in range(1):
                                # print('\nSampling\n')
                                xx = [self.alpha[name], self.beta[name], self.scale[name], self.log_noise[name],
                                      self.x_scale[name], self.x_ell[name]]
                                logdist = lambda xx: ft.ft_ll(m, [self.times_subset[name]], [self.scores_subset[name]],
                                                              x, x_kernel,
                                                              dict(scale=xx[4], ell=xx[5]), t_kernel,
                                                              dict(scale=xx[2], alpha=xx[0], beta=xx[1], log_noise=xx[3]))
                                try:
                                    xx = ft.slice_sample_bounded_max(1, 1, logdist, xx, 2, True, 10, self.bounds)[0]
                                except:
                                    logger.error('Slice sampling failed - continuing')
                                # print('\nFinished sampling\n')
                                # xx = ft.slice_sample_bounded_max(1, 1, logdist, xx, 0.5, False, 10, self.bounds)[0]
                                # print xx
                                self.alpha[name] = xx[0]
                                self.beta[name] = xx[1]
                                self.scale[name] = xx[2]
                                self.log_noise[name] = xx[3]
                                self.x_scale[name] = xx[4]
                                self.x_ell[name] = xx[5]
                                # Setup params
                                x_kernel_params = dict(scale=self.x_scale[name], ell=self.x_ell[name])
                                t_kernel_params = dict(scale=self.scale[name], alpha=self.alpha[name], beta=self.beta[name],
                                                       log_noise=self.log_noise[name])
                                post_m, post_v = ft.ft_posterior(m, [self.times_subset[name]], [self.scores_subset[name]],
                                                                 [self.t_star[name]], x, x_kernel, x_kernel_params,
                                                                 t_kernel, t_kernel_params)
                                # Remove excess noise
                                for i in range(post_v[0].shape[0]):
                                    post_v[0][i, i] -= np.exp(self.log_noise[name])
                                self.y_mean[name], self.y_covar[name] = post_m[0], post_v[0]
                                # Rescale
                                self.y_mean[name] = self.y_mean[name] * grand_scale + grand_mean
                                self.y_covar[name] = self.y_covar[name] * grand_scale
                                # self.y_samples[name] = []
                                for _ in range(5):
                                    # print(post_m[0].shape)
                                    # print(post_v[0].shape)
                                    # print(np.sqrt(np.diag(post_v[0])).shape)
                                    # sample = post_m[0].ravel() +\
                                    #          np.sqrt(np.diag(post_v[0])).ravel() * np.random.randn(*post_m[0].ravel().shape)

                                    # print(sample.shape)
                                    try:
                                        # TODO - test this function when things getting close to singular
                                        sample = np.random.multivariate_normal(post_m[0].ravel(),
                                                                               post_v[0],
                                                                               size=(1,))
                                    except:
                                        # Might have been singular
                                        sample = post_m[0].ravel()
                                    # Rescale
                                    sample = sample * grand_scale + grand_mean
                                    self.y_samples[name].append(sample)
                                # Send an update home
                                # self.send_to_parent(dict(subject='meta predictions', sender=self.name,
                                #                        t=self.times_subset, y=self.scores_subset,
                                #                        t_star=self.t_star,
                                #                        y_mean=self.y_mean,
                                #                        y_covar=self.y_covar))

                            # Remove old samples
                            while len(self.y_samples[name]) > 10:
                                self.y_samples[name].pop(0)
                            # Also compute posterior for already computed predictions
                            # FIXME - what if prediction times has empty lists
                            post_m, _ = ft.ft_posterior(m, [self.times_subset[name]], [self.scores_subset[name]],
                                                        [self.learner_prediction_times[name]], x,
                                                        x_kernel, x_kernel_params,
                                                        t_kernel, t_kernel_params)
                            self.predict_mean[name] = post_m[0]
                            # Rescale
                            self.predict_mean[name] = self.predict_mean[name] * grand_scale + grand_mean
                            # Rescale something else
                            self.scores_subset[name] = np.array(self.scores_subset[name]) * grand_scale + grand_mean
                            self.scores_subset[name] = list(self.scores_subset[name])

                            # Send an update home
                            self.send_to_parent(dict(subject='meta predictions', sender=self.name,
                                                     t=self.times_subset, y=self.scores_subset,
                                                     t_star=self.t_star,
                                                     y_mean=self.y_mean,
                                                     y_covar=self.y_covar,
                                                     y_samples=self.y_samples))

                            # Identify predictions thought to be the best currently
                            best_mean = -np.inf
                            best_learner = None
                            best_time_index = None
                            for name in self.learner_score_times.iterkeys():
                                if len(self.predict_mean[name]) > 0 and max(self.predict_mean[name]) >= best_mean:
                                    best_mean = max(self.predict_mean[name])
                                    best_learner = name
                                    best_time_index = np.argmax(np.array(self.predict_mean[name]))
                            # print('Best learner : %s' % best_learner)
                            # print('Best time : %f' % self.learner_prediction_times[best_learner][best_time_index])
                            # print('Estimated performance : %f' %self. predict_mean[best_learner][best_time_index])
                            # Report home
                            self.send_to_parent(dict(subject='prediction selection', sender=self.name,
                                                   learner=best_learner,
                                                   time_index=best_time_index,
                                                   value=best_mean))

                            # Pick best candidate to run next
                            best_current_value = best_mean
                            best_learner = None
                            best_acq_fn = -np.inf
                            for name in self.learner_score_times.iterkeys():
                                if len(self.y_mean[name]) > 0:
                                    mean = self.y_mean[name][-1]
                                    # std = np.sqrt(self.y_covar[name][-1, -1] - np.exp(self.log_noise[name]))
                                    std = np.sqrt(self.y_covar[name][-1, -1])
                                    acq_fn = ft.trunc_norm_mean_upper_tail(a=best_current_value, mean=mean, std=std) -\
                                             best_current_value
                                    if acq_fn >= best_acq_fn:
                                        best_acq_fn = acq_fn
                                        best_learner = name
                                    names_scores.append((name, acq_fn))
                                else:
                                    names_scores.append((name, -np.inf))
                            # print('Selecting learner : %s' % best_learner)
                            # Collate result and send to parent
                            names_scores.sort(key=lambda x: x[1])
                            sorted_names = [name for (name, score) in names_scores]
                            self.send_to_parent(dict(subject='computation preference', sender=self.name, preference=sorted_names))
                # Ask for more data
                # if constants.DEBUG:
                #     print('\n\n\n\nAsking for scores\n\n\n\n')
                self.send_to_parent(dict(subject='scores please'))
                self.waiting = True
        time_taken = time_module.clock() - start_time
        # # Do not ask for the scores too often!
        # if not self.compute_quantum is None:
        #     self.communication_sleep = max(1, self.compute_quantum / 3 - time_taken)


class StackerV1(Agent):
    """First experiment at stacking"""

    def __init__(self, data_info, **kwargs):
        super(StackerV1, self).__init__(**kwargs)

        self.data_info = data_info

        self.remaining_time = None

        self.learners = []
        self.learner_score_values = dict()
        self.learner_score_times = dict()
        self.learner_held_out_pred_times = dict()
        self.learner_held_out_pred_files = dict()
        self.learner_valid_pred_times = dict()
        self.learner_valid_pred_files = dict()
        self.learner_test_pred_times = dict()
        self.learner_test_pred_files = dict()
        self.learner_order = list()

        self.stack_times = list()
        self.stack_scores = list()
        self.improvement_amounts = defaultdict(list)
        self.improvement_times = defaultdict(list)

        self.stack_test_files = []

        self.communication_sleep = 1

        self.waiting = True

        self.valid_data = None
        self.test_data = None

        # Meta learner predictions
        self.meta_pred_times_past = defaultdict(list)
        self.meta_pred_times = defaultdict(list)
        self.meta_pred_means = defaultdict(list)
        self.meta_pred_covar = defaultdict(list)
        self.meta_pred_samples = defaultdict(list)

        # Weights in stacking and blacklist
        self.meta_data_set_order = []
        self.stacking_weights = []
        self.stacking_variances = []
        self.stacking_importances = []
        self.blacklist = []  # A list of algorithms not to include in stacking

        # Meta data set
        self.meta_X = None
        self.meta_X_test = None
        self.meta_X_valid = None
        self.meta_y = None
        self.targets = None

        # Record data in different ways
        self.time_ordered_held_out_files = dict()  # Lists held out filenames in order of creation
        self.time_ordered_valid_files = dict()  # Sim for validation set
        self.time_ordered_test_files = dict()  # Sim for test set
        self.predict_times = defaultdict(list)   # The times of these file creations - relative to the learners
        self.scores_at_predict_time = defaultdict(list)  # Saving the scores at these times TODO - get from FT
        self.best_scores = defaultdict(functools.partial(util.identity, -np.inf))  # The best scores for each learner TODO - get from FT

        self.stacking_feature_data = defaultdict(list)

        # Misc state
        self.data_count = 0
        self.total_time = 0
        self.updated_child = None
        self.saved_test_files = None
        self.time_checkpoint = None
        self.original_compute_quantum = None

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
            except (IndexError, AttributeError):
                break
            else:
                self.standard_responses(message)
                if message['subject'] == 'scores':
                    self.waiting = False
                    # Update local copies of scores
                    self.learners = message['learners']
                    self.learner_score_values = message['learner_score_values']
                    self.learner_score_times = message['learner_score_times']
                    self.learner_held_out_pred_times = message['learner_held_out_prediction_times']
                    self.learner_held_out_pred_files = message['learner_held_out_prediction_files']
                    # FIXME - hax
                    self.meta_pred_times_past = copy.deepcopy(self.learner_held_out_pred_times)
                    self.learner_valid_pred_times = message['learner_valid_prediction_times']
                    self.learner_valid_pred_files = message['learner_valid_prediction_files']
                    self.learner_test_pred_times = message['learner_test_prediction_times']
                    self.learner_test_pred_files = message['learner_test_prediction_files']
                    self.learner_order = message['learner_order']
                    self.remaining_time = message['remaining_time']
                if message['subject'] == 'predictions':
                    # print('Received predictions')
                    self.meta_pred_times = message['times']
                    self.meta_pred_means = message['means']
                    self.meta_pred_covar = message['covar']
                    self.meta_pred_samples = message['samples']
                if message['subject'] == 'original compute quantum':
                    self.original_compute_quantum = message['compute_quantum']

    def first_action(self):
        self.state = 'running'
        p = psutil.Process()
        current_cpus = p.cpu_affinity()
        if len(current_cpus) > 1:
            p.cpu_affinity([current_cpus[1]])  # meta learning only happens on the second CPU
        if constants.DEBUG:
            with open(constants.STACK_DATA_FL, 'w') as stacking_data_file:
                stacking_data_file.write('ID,current,imp,imp_over_best,imp_over_stack,corr,current_stack,stack_imp\n')
        # Set a few flags
        self.valid_data = 'X_valid' in self.data
        self.test_data = 'X_test' in self.data
        # Ask for data to get things started
        self.send_to_parent(dict(subject='scores please'))
        self.send_to_parent(dict(subject='predictions please'))
        self.waiting = True

    @property
    def n_predictions(self):
        return len(self.learner_order)

    def perform_stacking(self):
        """Outer loop of stacking - data management and calling of routines"""
        self.time_checkpoint = time_module.clock()
        # FIXME - Dirty hax
        self.saved_test_files = copy.deepcopy(self.learner_test_pred_files)
        # Setup before constructing data
        # TODO - no need to repeat this all the time - can save the information!
        self.time_ordered_held_out_files = dict()  # Lists held out filenames in order of creation
        self.time_ordered_valid_files = dict()  # Sim for validation set
        self.time_ordered_test_files = dict()  # Sim for test set
        self.predict_times = dict()  # The times of these file creations - relative to the learners
        self.scores_at_predict_time = defaultdict(list)  # Saving the scores at these times
        self.best_scores = defaultdict(functools.partial(util.identity, -np.inf))  # The best scores for each learner
        # Count through the data
        for n in range(self.n_predictions):
            # Which child last made predictions?
            self.updated_child = self.learner_order[n]
            # Update time
            self.predict_times[self.updated_child] = self.learner_held_out_pred_times[self.updated_child].pop(0)
            # Determine most recent score
            child_score = None
            # Do we have access to a smoothed score?
            for time, score in zip(self.meta_pred_times[self.updated_child],
                                   self.meta_pred_means[self.updated_child]):
                if np.allclose([time], [self.predict_times[self.updated_child]]):
                    child_score = score
                    # print('Used smoothed score %f' % score)
            # If not - take from data
            if child_score is None:
                for time, score in zip(self.learner_score_times[self.updated_child],
                                       self.learner_score_values[self.updated_child]):
                    if time <= self.predict_times[self.updated_child]:
                        child_score = score
            self.scores_at_predict_time[self.updated_child].append(child_score)
            if child_score > self.best_scores[self.updated_child]:
                self.best_scores[self.updated_child] = child_score
                update_files = True
            else:
                update_files = False
            # Update files
            if update_files:
                self.time_ordered_held_out_files[self.updated_child] = self.learner_held_out_pred_files[self.updated_child].pop(0)
                if self.valid_data:
                    self.time_ordered_valid_files[self.updated_child] = self.learner_valid_pred_files[self.updated_child].pop(0)
                if self.test_data:
                    self.time_ordered_test_files[self.updated_child] = self.learner_test_pred_files[self.updated_child].pop(0)
            else:
                # Individual score did not improve - do not update file
                self.learner_held_out_pred_files[self.updated_child].pop(0)
                if self.valid_data:
                    self.learner_valid_pred_files[self.updated_child].pop(0)
                if self.test_data:
                    self.learner_test_pred_files[self.updated_child].pop(0)
            # Should we update the stacking performance and blame?
            if n + 1 > len(self.stack_scores):
                stacking_time_checkpoint = time_module.clock()
                # Construct meta data set
                self.construct_meta_data_set()
                # Learn meta model
                self.learn_stack()
                # Has this taken a while
                stacking_time_taken = time_module.clock() - stacking_time_checkpoint
                # print('Original CQ = %f' % self.original_compute_quantum)
                # print('Stacking learning time = %f' % stacking_time_taken)
                if ((not self.original_compute_quantum is None) and
                    (stacking_time_taken > self.original_compute_quantum)):
                    self.blacklist.append(self.stacking_importances[-1][0][0])  # Most recent, least important, name
                    # print(self.blacklist)
                    # print(self.stacking_importances[-1])
                self.send_to_parent(dict(subject='time taken', sender=self.name, time=stacking_time_taken))
                # Record data about algorithm performance and stacking performance
                self.record_stacking_data()
                # Decide preferences for learners if enough data
                if n >= 1:
                    self.recommend_learners()
                # Recommend which predictions to use
                self.select_predictions()
        time_taken = time_module.clock() - self.time_checkpoint
        # print('Total stacking time = %f' % time_taken)
        # self.send_to_parent(dict(subject='time taken', sender=self.name, time=time_taken))
        self.send_to_parent(dict(subject='finished stacking', send=self.name))

    def construct_meta_data_set(self):
        """Assemble latest predictions into a meta data set for the purposes of stacking"""
        if self.data_info['task'] == 'binary.classification':
            targets = 1
        elif self.data_info['task'] == 'multiclass.classification':
            targets = self.data_info['target_num']
        else:
            raise Exception('I do not know how to set the number of targets for %s' % self.data_info['task'])
        # Set number of base learners
        count = 0
        learners_on_the_guestlist = []
        for name in self.time_ordered_held_out_files.iterkeys():
            if not name in self.blacklist:
                learners_on_the_guestlist.append(name)
        if len(learners_on_the_guestlist) == 0:
            learners_on_the_guestlist.append(self.learner_order[-1])
        meta_X = np.zeros((self.data['Y_train'].shape[0], len(learners_on_the_guestlist) * targets))
        if self.valid_data:
            meta_X_valid = np.zeros((self.data['X_valid'].shape[0],
                                     len(learners_on_the_guestlist) * targets))
        else:
            meta_X_valid = None
        if self.test_data:
            meta_X_test = np.zeros((self.data['X_test'].shape[0],
                                    len(learners_on_the_guestlist) * targets))
        else:
            meta_X_test = None
        self.data_count = 0
        self.total_time = 0
        self.meta_data_set_order = []
        self.stacking_variances = []
        for name in learners_on_the_guestlist:
            self.meta_data_set_order.append(name)
            # TODO - these temporary files could be destroyed here
            filename = self.time_ordered_held_out_files[name]
            predictions = np.load(filename)
            predictions = util.ensure_2d(predictions)
            if targets == 1:
                self.stacking_variances.append(np.var(predictions.ravel()))
            else:
                var = 0
                for i in range(targets):
                    var += np.var(predictions[:,i].ravel())
                var = var / targets
                self.stacking_variances.append(var)
            meta_X[:, (self.data_count * targets):((self.data_count + 1) * targets)] = predictions
            if self.valid_data:
                filename = self.time_ordered_valid_files[name]
                predictions = np.load(filename)
                predictions = util.ensure_2d(predictions)
                meta_X_valid[:, (self.data_count * targets):((self.data_count + 1) * targets)] = predictions
            if self.test_data:
                filename = self.time_ordered_test_files[name]
                predictions = np.load(filename)
                predictions = util.ensure_2d(predictions)
                meta_X_test[:, (self.data_count * targets):((self.data_count + 1) * targets)] = predictions
            self.data_count += 1
            self.total_time += self.predict_times[name]
        if self.data_info['task'] == 'multiclass.classification':
            meta_y = self.data['Y_train_1_of_k']
        else:
            meta_y = self.data['Y_train']
        self.meta_X = meta_X
        self.meta_X_test = meta_X_test
        self.meta_X_valid = meta_X_valid
        self.meta_y = meta_y
        self.targets = targets
        # if constants.DEBUG:
            # np.savetxt('../stacking-data/meta_X.csv', meta_X, delimiter=',')
            # np.savetxt('../stacking-data/meta_y.csv', meta_y, delimiter=',')
            # if np.all(meta_X_test == 0):
            #     raise Exception('Meta X test is just zeros')
            # if np.all(meta_X == 0):
            #     raise Exception('Meta X is just zeros')

    def learn_stack(self):
        """Form meta model, make predictions and update manager"""
        folds = KFold(n=self.meta_X.shape[0], n_folds=5)
        # Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        # Cs = [0.01, 1, 100]
        # Cs = [0.01, 100]
        Cs = [100]
        scores = []
        best_score = -np.inf
        best_C = None
        for C in Cs:
            sum_score = 0
            n_score = 0
            sum_score_test = 0
            n_score_test = 0
            for train, test in folds:

                if self.data_info['task'] == 'multiclass.classification':
                    meta_model = StackCombiner(num_classes=self.targets, C=C, combine_method="tied_ovr")
                else:
                    meta_model = LogisticRegression(C=C, penalty='l2')
                meta_model.fit(self.meta_X[train], self.meta_y[train])
                # FIXME - get rid of this if / else
                if self.data_info['task'] == 'multiclass.classification':
                    pred = meta_model.predict_proba(self.meta_X[test])
                else:
                    pred = meta_model.predict_proba(self.meta_X[test])[:, 1]
                score = libscores.eval_metric(metric=self.data_info['eval_metric'],
                                              truth=self.meta_y[test],
                                              predictions=pred,
                                              task=self.data_info['task'])
                if not np.isnan(score):
                    sum_score += score
                    n_score += 1
            score = sum_score / n_score
            scores.append(score)
            if score > best_score:
                best_score = score
                best_C = C
        # Make predictions with best model
        if self.data_info['task'] == 'multiclass.classification':
            meta_model = StackCombiner(num_classes=self.targets, C=best_C, combine_method="tied_ovr")
        else:
            meta_model = LogisticRegression(C=best_C, penalty='l2')
        meta_model.fit(self.meta_X, self.meta_y)
        if self.valid_data:
            if self.data_info['task'] == 'multiclass.classification':
                pred_valid = meta_model.predict_proba(self.meta_X_valid)
            else:
                pred_valid = meta_model.predict_proba(self.meta_X_valid)[:, 1]
            tmp_valid = util.random_temp_file_name('.npy')
            np.save(tmp_valid, pred_valid)
        else:
            pred_valid = None
            tmp_valid = None
        if self.test_data:
            if self.data_info['task'] == 'multiclass.classification':
                pred_test = meta_model.predict_proba(self.meta_X_test)
            else:
                pred_test = meta_model.predict_proba(self.meta_X_test)[:, 1]
            tmp_test = util.random_temp_file_name('.npy')
            np.save(tmp_test, pred_test)
        else:
            pred_test = None
            tmp_test = None
        if 'Y_test' in self.data:
            if self.data_info['task'] == 'multiclass.classification':
                meta_y_test = self.data['Y_test_1_of_k']
            else:
                meta_y_test = self.data['Y_test']
            test_score = libscores.eval_metric(metric=self.data_info['eval_metric'],
                                               truth=meta_y_test,
                                               predictions=pred_test,
                                               task=self.data_info['task'])
        else:
            test_score = None
        # Record weights
        coefs = list(np.array(meta_model.coef_).ravel())
        # print(coefs)
        self.stacking_weights.append(list(zip(self.meta_data_set_order, coefs)))
        self.stacking_importances.append([(learner, abs(coef) * np.sqrt(var))
                                          for (learner, coef, var) in zip(self.meta_data_set_order,
                                                                          coefs,
                                                                          self.stacking_variances)])
        self.stacking_weights[-1].sort(key=lambda x: abs(x[-1]))
        self.stacking_importances[-1].sort(key=lambda x: abs(x[-1]))
        # Record times and scores
        self.stack_scores.append(best_score)
        self.stack_times.append(self.total_time)
        self.stack_test_files.append(tmp_test)
        if len(self.stack_scores) >= 2:
            self.improvement_times[self.updated_child].append(self.total_time)
            self.improvement_amounts[self.updated_child].append(self.stack_scores[-1] -
                                                           self.stack_scores[-2])
        # Tell parent about the estimated performance of the stacking
        self.send_to_parent(dict(subject='stacking performance',
                                 time=self.total_time,
                                 held_out_score=best_score,
                                 test_score=test_score,
                                 valid_pred_file=tmp_valid,
                                 test_pred_file=tmp_test))
        # Tell parent who to blame / praise
        self.send_to_parent(dict(subject='stacking blame',
                                 times=self.improvement_times,
                                 amounts=self.improvement_amounts))

    def record_stacking_data(self):
        """Save data about stacking to file"""
        # FIXME - need more object properties
        if len(self.stack_scores) > 1:
            stack_performance_change = self.stack_scores[-1] - self.stack_scores[-2]
            previous_stack_performance = self.stack_scores[-2]
        else:
            stack_performance_change = np.nan
            previous_stack_performance = np.nan
        if len(self.scores_at_predict_time[self.updated_child]) > 1:
            # individual_improvement = self.scores_at_predict_time[self.updated_child][-1] - \
                                     # self.scores_at_predict_time[self.updated_child][-2]
            individual_improvement = self.scores_at_predict_time[self.updated_child][-1] - \
                                     max(self.scores_at_predict_time[self.updated_child][:-1])
            # TODO - is this misleading? Negative improvements are not improvements
            individual_improvement = max(0, individual_improvement)
            previous_learner_performance = self.scores_at_predict_time[self.updated_child][-2]
        else:
            individual_improvement = np.nan
            previous_learner_performance = np.nan
        child_identity = self.updated_child
        # Compute best algorithm
        best_score = -np.inf
        for learner, scores in self.scores_at_predict_time.iteritems():
            if learner == self.updated_child:
                # Do not include most recent score
                best_score = max([best_score] + scores[:-1])
            else:
                best_score = max([best_score] + scores)
        improvement_over_best = self.scores_at_predict_time[self.updated_child][-1] - best_score
        if len(self.stack_scores) > 1:
            improvement_over_stack = self.scores_at_predict_time[self.updated_child][-1] - \
                                     self.stack_scores[-2]
        else:
            improvement_over_stack = np.nan
        # Correlation
        if self.data_info['task'] == 'binary.classification' and \
           len(self.saved_test_files[self.updated_child]) > 1 and \
           len(self.stack_scores) > 1:
            learner_test_predictions = np.load(self.saved_test_files[self.updated_child][-2])
            stack_predictions = np.load(self.stack_test_files[-2])
            correlation = np.corrcoef(learner_test_predictions, stack_predictions)[0, 1]
        else:
            correlation = np.nan

        # print(stack_performance_change, individual_improvement, child_identity, improvement_over_best,
        #       improvement_over_stack, correlation)
        # Save to local structure
        self.stacking_feature_data['learners'].append(child_identity)
        self.stacking_feature_data['previous_performances'].append(previous_learner_performance)
        self.stacking_feature_data['imps'].append(individual_improvement)
        self.stacking_feature_data['imps_over_best'].append(improvement_over_best)
        self.stacking_feature_data['imps_over_stack'].append(improvement_over_stack)
        self.stacking_feature_data['correlations'].append(correlation)
        self.stacking_feature_data['previous_stack_performances'].append(previous_stack_performance)
        self.stacking_feature_data['stack_imps'].append(stack_performance_change)
        # Send to parent
        self.send_to_parent(dict(subject='stacking stats', sender=self.name,
                                 data=[]))
        # Save to file
        if constants.DEBUG:
            with open(constants.STACK_DATA_FL, 'a') as stacking_data_file:
                stacking_data_file.write('%s,%f,%f,%f,%f,%f,%f,%f\n' % (child_identity,
                                                                        previous_learner_performance,
                                                                        individual_improvement,
                                                                        improvement_over_best,
                                                                        improvement_over_stack,
                                                                        correlation,
                                                                        previous_stack_performance,
                                                                        stack_performance_change))

    def recommend_learners(self):
        self.recommend_ft_regression_tree()
        # self.recommend_past_performance()

    def recommend_past_performance(self):
        """Make recommendations based on a really simple model that tracks performance of individual algorithms"""
        # First set empirical mean and variance
        i = 0
        temp_imp_amounts = copy.deepcopy(self.improvement_amounts)
        all_recent_imps = defaultdict(list)
        sum_imps = 0
        sum_sqr_imps = 0
        # prior_window_length = max(10, len(self.learners) * 2)
        prior_window_length = 20
        # FIXME - hax
        n = len(self.predict_times) - 1
        while i < prior_window_length and n - i >= 1:
            imp = temp_imp_amounts[self.learner_order[n-i]].pop(-1)
            all_recent_imps[self.learner_order[n-i]].append(imp)
            sum_imps += imp
            sum_sqr_imps += imp * imp
            i += 1
        if i > 0:  # TODO - work out why this is necessary!
            mean_imp = sum_imps / i
            var_imp = sum_sqr_imps / i - mean_imp * mean_imp
            # Estimate child means and variances
            learner_values = list()
            for name in self.learners:
                # recent_learner_imps = self.improvement_amounts[name][-3:]
                recent_learner_imps = all_recent_imps[name][-3:]
                N = len(recent_learner_imps)
                learner_imp_sum = sum(recent_learner_imps)
                value = (mean_imp + learner_imp_sum) / ( 1 + N) +\
                        np.sqrt((var_imp / (1 + N))) * np.random.normal()
                learner_values.append((name, value))
            learner_values.sort(key=lambda x: x[-1])
            sorted_names = [name for (name, score) in learner_values]
            # Tell parent
            self.send_to_parent(dict(subject='computation preference', sender=self.name,
                                     preference=sorted_names))
            # Remove good algorithms from the blacklist
            best = sorted_names[-1]
            if best in self.blacklist:
                self.blacklist.remove(best)

    def recommend_ft_regression_tree(self):
        """Pass freeze thaw predictions through a regression tree to produce utility estimates"""
        # Collect up data to learn model
        imps = self.stacking_feature_data['imps']
        imps_over_best = self.stacking_feature_data['imps_over_best']
        stack_imps = self.stacking_feature_data['stack_imps']
        features = []
        for imp, imp_over_best, stack_imp in zip(imps, imps_over_best, stack_imps):
            if not np.isnan(imp) and not np.isnan(imp_over_best) and not np.isnan(stack_imp) and not imp <= 0:
                features.append((imp_over_best, stack_imp / imp))
        # Learn model
        # TODO - learn a real model
        features.sort(key=lambda x: x[0])
        # print(features)
        cut_offs = []
        ratios = []
        sum_ratios = 0
        n = 0
        ratio_list = []
        while len(features) > 0:
            imp_over_best, ratio = features.pop()
            # Clip data to avoid outliers to some extent
            if ratio < 0:
                ratio = 0
            if ratio > 2:
                ratio = 2
            # Count
            n += 1
            sum_ratios += ratio
            ratio_list.append(ratio)
            if n >= 10:
                cut_offs.append(imp_over_best)
                # ratios.append(sum_ratios / n)
                ratios.append(copy.deepcopy(ratio_list))
                n = 0
                sum_ratios = 0
                ratio_list = []
        if len(cut_offs) > 0:
            cut_offs[-1] = -np.inf
            ratios[-1] += ratio_list
        # print(cut_offs)
        # print(ratios)
        # Featurise potential actions
        # First find best scores
        best_learner_scores = defaultdict(list)
        for learner in self.meta_pred_times_past.iterkeys():
            if len(self.meta_pred_times_past[learner]) > 0:
                best_learner_score = -np.inf
                for t, m, v in zip(self.meta_pred_times[learner], self.meta_pred_means[learner],
                                   self.meta_pred_covar[learner]):
                    if t <= self.meta_pred_times_past[learner][-1]:
                        if m > best_learner_score:
                            best_learner_score = m
                    else:
                        break
                best_learner_scores[learner] = best_learner_score
        best_score = -np.inf
        for score in best_learner_scores.itervalues():
            if score > best_score:
                best_score = score
        # Now compute featurers
        # TODO - sample features - should we take a joint sample?
        action_features = defaultdict(lambda: defaultdict(list))
        for learner in self.meta_pred_times.iterkeys():
            if len(self.meta_pred_times_past[learner]) > 0:
                for sample in self.meta_pred_samples[learner]:
                    sample_learner_best = best_learner_scores[learner]
                    sample_global_best = best_score
                    imp_sequence = []
                    imp_over_best_sequence = []
                    for t, y in zip(self.meta_pred_times[learner], sample.ravel()):
                        if t <= self.meta_pred_times_past[learner][-1]:
                            last_time = t
                        else:
                            imp = max(0, y - sample_learner_best)
                            imp_over_best = y - sample_global_best
                            imp_sequence.append(imp)
                            imp_over_best_sequence.append(imp_over_best)
                        sample_learner_best = max(sample_learner_best, y)
                        sample_global_best = max(sample_global_best, y)
                    action_features[learner]['imps'].append(imp_sequence)
                    action_features[learner]['imps_over_best'].append(imp_over_best_sequence)
        # Compute utilities of actions by passing through model
        # TODO - take an average of samples - should we take a joint sample of ratios?
        action_utilities = defaultdict(list)
        for learner in action_features.iterkeys():
            for imp_sequence, imp_over_best_sequence in zip(action_features[learner]['imps'],
                                                            action_features[learner]['imps_over_best']):
                utility_sample = []
                # Sample some ratios
                sampled_ratios = []
                for ratio_samples in ratios:
                    sampled_ratio = random.choice(ratio_samples)
                    sampled_ratios.append(sampled_ratio)
                for imp, imp_over_best in zip(imp_sequence,
                                              imp_over_best_sequence):
                    if imp <= 0:
                        utility_sample.append(0)
                    elif len(ratios) < 2:
                        if imp_over_best > 0:
                            utility_sample.append(imp)
                        else:
                            utility_sample.append(imp * 0.2)
                    else:
                        success = False
                        for ratio, cut_off in zip(sampled_ratios, cut_offs):
                            if imp_over_best > cut_off:
                                # print('Using ratio %f' % ratio)
                                utility_sample.append(imp * ratio)
                                success = True
                                break
                                # FIXME - there is a bug that means the break isn't working :(
                        if not success:
                            utility_sample.append(imp * ratio)
                action_utilities[learner].append(utility_sample)
        # Take average
        average_action_utilities = defaultdict(list)
        for learner in action_utilities.iterkeys():
            for utilities in zip(*action_utilities[learner]):
                average_action_utilities[learner].append(sum(utilities) / len(utilities))
        # Compute utilities of multi-step actions
        multi_step_utilities = defaultdict(list)
        for learner in average_action_utilities.iterkeys():
            count = 0
            sum_utility = 0
            for utility in average_action_utilities[learner]:
                count += 1
                sum_utility += utility
                multi_step_utilities[learner].append(float(sum_utility / count))
        # Tell parent about utilities
        self.send_to_parent(dict(subject='utilities', sender=self.name,
                                 utilities=multi_step_utilities,))
        # Tell parent which learner to run
        learner_utility_list = list()
        for learner in multi_step_utilities.iterkeys():
            learner_utility_list.append((learner, max(multi_step_utilities[learner])))
        learner_utility_list.sort(key=lambda x: x[-1])
        sorted_names = [name for (name, score) in learner_utility_list]
        self.send_to_parent(dict(subject='computation preference', sender=self.name,
                                 preference=sorted_names))
        # Remove good algorithms from the blacklist
        if len(sorted_names) > 0:
            best = sorted_names[-1]
            if best in self.blacklist:
                self.blacklist.remove(best)

    def select_predictions(self):
        """Tell the manager which predicitons to use"""
        # Find best index and score for stacker
        best_score = -np.inf
        best_learner = 'stacker'
        best_index = None
        for i, score in enumerate(self.stack_scores):
            if score >= best_score:
                best_score = score
                best_index = i
        # Find best index and score for the learners
        for learner in self.scores_at_predict_time.iterkeys():
            for i, score in enumerate(self.scores_at_predict_time[learner]):
                if score >= best_score:
                    best_score = score
                    best_index = i
                    best_learner = learner
        # Find best one shot algo performance
        # FIXME - this code should now be redundant?
        for learner in self.learner_score_times.iterkeys():
            if len(self.learner_score_times[learner]) == 1 and \
               len(self.scores_at_predict_time[learner]) == 1:  # Is this a one shot algo with a prediction?
                score = self.scores_at_predict_time[learner][0]
                if score >= best_score:
                    best_score = score
                    best_index = 0
                    best_learner = learner
        # Tell manager
        self.send_to_parent(dict(subject='prediction selection', sender=self.name,
                                 learner=best_learner,
                                 time_index=best_index,
                                 value=best_score))

    def next_action(self):
        # Check mail
        self.read_messages()
        # If running, form an opinion of the best learner and tell parent
        if self.state == 'running':
            if not self.waiting:
                self.perform_stacking()
                # Ask for more data
                # FIXME - need to ask for scores first time!!!
                self.send_to_parent(dict(subject='scores please'))
                self.send_to_parent(dict(subject='predictions please'))
                self.waiting = True
                # if constants.DEBUG:
                #     print('\n\n\n\nStacker asking for data\n\n\n\n')