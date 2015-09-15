from __future__ import division

__author__ = 'James Robert Lloyd, Emma Smith'
__description__ = 'Objects that learn from data and predict things'

import time
import copy
import os
# import cPickle as pickle
from collections import defaultdict

import numpy as np
# from sklearn import metrics

import libscores

from agent import Agent, TerminationEx
import util
import constants
import logging

import global_data

# Set up logging for learners module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO - WarmLearner and OneShotLearner should derive from a common base class


class LearnerAgent(Agent):
    """Base class for agent wrappers around learners"""

    def __init__(self, learner, learner_kwargs, train_idx, test_idx, data_info, feature_subset, **kwargs):
        super(LearnerAgent, self).__init__(**kwargs)

        self.learner = learner(**learner_kwargs)
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.data_info = data_info
        self.feature_subset = feature_subset

        self.time_before_checkpoint = 0
        self.time_checkpoint = None
        self.score_times = []
        self.score_values = []
        self.held_out_prediction_times = []
        self.held_out_prediction_files = []
        self.valid_prediction_times = []
        self.valid_prediction_files = []
        self.test_prediction_times = []
        self.test_prediction_files = []
        self.all_class_labels = []
        self.training_class_labels = []
        self.training_to_all_class_labels = []
        self.test_truth = None

        self.data_source = None  # Record how we should access data

    def first_action(self):
        # Record the observed class labels if doing multiclass classification - in case training data does not have
        # examples of all classes
        if self.data_info['task'] == 'multiclass.classification':
            self.all_class_labels = np.unique(self.data['Y_train'])
            self.training_class_labels = np.unique(self.data['Y_train'][self.train_idx])
            self.training_to_all_class_labels = []
            for class_label in self.training_class_labels:
                location = np.where(class_label == self.all_class_labels)[0][0]
                self.training_to_all_class_labels.append(location)
        # Record the truth
        if self.data_info['task'] == 'multiclass.classification':
            self.test_truth = self.data['Y_train_1_of_k'][self.test_idx]
        else:
            self.test_truth = self.data['Y_train'][self.test_idx]
        # Set up feature subset if none or too large
        if self.feature_subset is None or self.feature_subset > self.data['X_train'].shape[1]:
            self.feature_subset = self.data['X_train'].shape[1]

    def get_data(self, name, rows, max_cols):
        """Gets data whilst dealing with sparse / dense stuff"""
        if self.data_source == constants.ORIGINAL:
            if rows == 'all':
                return self.data[name][:, :max_cols]
            else:
                return self.data[name][rows, :max_cols]
        elif self.data_source == constants.DENSE:
            if rows == 'all':
                return self.data[name + '_dense'][:, :max_cols]
            else:
                return self.data[name + '_dense'][rows, :max_cols]
        elif self.data_source == constants.CONVERT_TO_DENSE:
            if rows == 'all':
                return self.data[name][:, :max_cols].toarray()
            else:
                return self.data[name][rows, :max_cols].toarray()
        else:
            raise Exception('Unrecognised data source = %s' % self.data_source)

    def fit(self, rows, max_cols):
        """Deals with sparse / dense stuff"""
        if self.data_source is None:
            # Need to determine appropriate data source
            try:
                self.learner.fit(X=self.data['X_train'][rows, :max_cols],
                                 y=self.data['Y_train'][rows])
                self.data_source = constants.ORIGINAL
            except TypeError:
                # Failed to use sparse data
                if 'X_train_dense' in self.data:
                    self.learner.fit(X=self.data['X_train_dense'][rows, :max_cols],
                                     y=self.data['Y_train'][rows])
                    self.data_source = constants.DENSE
                else:
                    self.learner.fit(X=self.data['X_train'][rows, :max_cols].toarray(),
                                     y=self.data['Y_train'][rows])
                    self.data_source = constants.CONVERT_TO_DENSE
        else:
            self.learner.fit(X=self.get_data(name='X_train', rows=rows, max_cols=max_cols),
                             y=self.data['Y_train'][rows])

    def predict(self, name, rows, max_cols):
        """Deals with different types of task"""
        X_test = self.get_data(name=name, rows=rows, max_cols=max_cols)
        if self.data_info['task'] == 'binary.classification':
            return self.learner.predict_proba(X_test)[:, -1]
        elif self.data_info['task'] == 'multiclass.classification':
            result = np.ones((X_test.shape[0], len(self.all_class_labels)))
            result[:, self.training_to_all_class_labels] = self.learner.predict_proba(X_test)
            return result
        else:
            raise Exception('I do not know how to form predictions for task : %s' % self.data_info['task'])


class WarmLearnerAgent(LearnerAgent):
    """Agent wrapper around warm learner"""

    def __init__(self, time_quantum=30, n_estimators_quantum=1, n_samples=10, **kwargs):
        super(WarmLearnerAgent, self).__init__(**kwargs)

        self.time_quantum = time_quantum
        self.n_estimators_quantum = n_estimators_quantum
        self.learner.n_estimators = self.n_estimators_quantum
        self.n_samples = n_samples

        self.run_one_iteration = False

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
            except (IndexError, AttributeError):
                break
            else:
                self.standard_responses(message)
                # print(message)
                if message['subject'] == 'compute quantum':
                    # print('Warm learner received compute quantum message')
                    self.time_quantum = message['compute_quantum']
                elif message['subject'] == 'run one iteration':
                    self.run_one_iteration = True

    def next_action(self):
        # Read messages
        self.read_messages()
        # Start timing
        self.time_checkpoint = time.clock()
        predict_time = self.time_quantum / self.n_samples
        scores_so_far = 0
        # Increase estimators and learn
        while time.clock() - self.time_checkpoint < self.time_quantum:
            # Read messages - maybe compute quantum has changed?
            self.get_parent_inbox()
            self.read_messages()
            # Do learning
            self.learner.n_estimators += self.n_estimators_quantum
            start_time = time.clock()
            self.fit(self.train_idx, self.feature_subset)
            time_taken = time.clock() - start_time
            if global_data.exp['slowdown_factor'] > 1:
                util.waste_cpu_time(time_taken * (global_data.exp['slowdown_factor'] - 1))
            if time.clock() - self.time_checkpoint > predict_time:
                predictions = self.predict('X_train', self.test_idx, self.feature_subset)
                truth = self.test_truth
                score = libscores.eval_metric(metric=self.data_info['eval_metric'],
                                              truth=truth,
                                              predictions=predictions,
                                              task=self.data_info['task'])

                self.score_times.append(time.clock() - self.time_checkpoint + self.time_before_checkpoint)
                self.score_values.append(score)
                # Send score and time to parent
                self.send_to_parent(dict(subject='score', sender=self.name,
                                         time=self.score_times[-1],
                                         score=self.score_values[-1]))
                scores_so_far += 1
                # Next time at which to make a prediction
                if self.n_samples > scores_so_far:
                    predict_time = time.clock() - self.time_checkpoint + \
                                   (self.time_quantum - (time.clock() - self.time_checkpoint)) / \
                                   (self.n_samples - scores_so_far)
                else:
                    break
        # Save total time taken
        # TODO - this is ignoring the time taken to make valid and test predictions
        self.time_before_checkpoint += time.clock() - self.time_checkpoint
        # Now make predictions

        # FIXME - send all of this data at the same time to prevent gotchas

        if 'X_valid' in self.data:
            predictions = self.predict('X_valid', 'all', self.feature_subset)
            tmp_filename = util.random_temp_file_name('.npy')
            np.save(tmp_filename, predictions)
            self.valid_prediction_files.append(tmp_filename)
            self.valid_prediction_times.append(self.time_before_checkpoint)
            self.send_to_parent(dict(subject='predictions', sender=self.name, partition='valid',
                                     time=self.valid_prediction_times[-1],
                                     filename=self.valid_prediction_files[-1]))
        if 'X_test' in self.data:
            predictions = self.predict('X_test', 'all', self.feature_subset)
            tmp_filename = util.random_temp_file_name('.npy')
            np.save(tmp_filename, predictions)
            self.test_prediction_files.append(tmp_filename)
            self.test_prediction_times.append(self.time_before_checkpoint)
            self.send_to_parent(dict(subject='predictions', sender=self.name, partition='test',
                                     time=self.test_prediction_times[-1],
                                     filename=self.test_prediction_files[-1]))

        predictions = self.predict('X_train', self.test_idx, self.feature_subset)
        # print('Held out')
        # print(predictions[0])
        tmp_filename = util.random_temp_file_name('.npy')
        np.save(tmp_filename, predictions)
        self.held_out_prediction_files.append(tmp_filename)
        self.held_out_prediction_times.append(self.time_before_checkpoint)
        self.send_to_parent(dict(subject='predictions', sender=self.name, partition='held out',
                                 idx=self.test_idx,
                                 time=self.held_out_prediction_times[-1],
                                 filename=self.held_out_prediction_files[-1]))

        if self.run_one_iteration:
            self.pause()


class OneShotLearnerAgent(LearnerAgent):
    """Agent wrapper around learner which learns once"""

    def __init__(self, **kwargs):
        super(OneShotLearnerAgent, self).__init__(**kwargs)

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
            except (IndexError, AttributeError):
                break
            else:
                self.standard_responses(message)

    def next_action(self):
        self.read_messages()
        # Start timing
        self.time_checkpoint = time.clock()
        # Fit learner
        self.fit(self.train_idx, self.feature_subset)
        # Make predictions on held out set and evaluate
        predictions = self.predict('X_train', self.test_idx, self.feature_subset)
        truth = self.test_truth
        score = libscores.eval_metric(metric=self.data_info['eval_metric'],
                                      truth=truth,
                                      predictions=predictions,
                                      task=self.data_info['task'])

        self.score_times.append(time.clock() - self.time_checkpoint + self.time_before_checkpoint)
        self.score_values.append(score)
        # Send score and time to parent
        self.send_to_parent(dict(subject='score', sender=self.name,
                                 time=self.score_times[-1],
                                 score=self.score_values[-1]))
        # Save total time taken
        # TODO - this is ignoring the time taken to make valid and test predictions
        self.time_before_checkpoint += time.clock() - self.time_checkpoint
        # Now make predictions on valid, test and held out sets

        # FIXME - send all of this data at the same time to prevent gotchas

        if 'X_valid' in self.data:
            predictions = self.predict('X_valid', 'all', self.feature_subset)
            tmp_filename = util.random_temp_file_name('.npy')
            np.save(tmp_filename, predictions)
            self.valid_prediction_files.append(tmp_filename)
            self.valid_prediction_times.append(self.time_before_checkpoint)
            self.send_to_parent(dict(subject='predictions', sender=self.name, partition='valid',
                                     time=self.valid_prediction_times[-1],
                                     filename=self.valid_prediction_files[-1]))
        if 'X_test' in self.data:
            predictions = self.predict('X_test', 'all', self.feature_subset)
            tmp_filename = util.random_temp_file_name('.npy')
            np.save(tmp_filename, predictions)
            self.test_prediction_files.append(tmp_filename)
            self.test_prediction_times.append(self.time_before_checkpoint)
            self.send_to_parent(dict(subject='predictions', sender=self.name, partition='test',
                                     time=self.test_prediction_times[-1],
                                     filename=self.test_prediction_files[-1]))

        predictions = self.predict('X_train', self.test_idx, self.feature_subset)
        tmp_filename = util.random_temp_file_name('.npy')
        np.save(tmp_filename, predictions)
        self.held_out_prediction_files.append(tmp_filename)
        self.held_out_prediction_times.append(self.time_before_checkpoint)
        self.send_to_parent(dict(subject='predictions', sender=self.name, partition='held out',
                                 idx=self.test_idx,
                                 time=self.held_out_prediction_times[-1],
                                 filename=self.held_out_prediction_files[-1]))
        # And I'm spent
        raise TerminationEx


class CrossValidationAgent(Agent):
    """Basic cross validation agent"""

    def __init__(self, learner, learner_kwargs, agent_kwargs, folds, data_info,
                 agent=WarmLearnerAgent, subset_prop=1, feature_subset=None, **kwargs):
        super(CrossValidationAgent, self).__init__(**kwargs)

        self.data_info = data_info

        self.child_info = []
        for train, test in folds:
            if subset_prop < 1:
                # noinspection PyUnresolvedReferences
                train = train[:int(np.floor(subset_prop * train.size))]
            self.child_info.append((agent,
                                    util.merge_dicts(agent_kwargs,
                                                     dict(learner=learner, learner_kwargs=learner_kwargs,
                                                          train_idx=train, test_idx=test,
                                                          data_info=data_info,
                                                          feature_subset=feature_subset))))

        self.score_times = []
        self.score_values = []
        self.held_out_prediction_times = []
        self.held_out_prediction_files = []
        self.valid_prediction_times = []
        self.valid_prediction_files = []
        self.test_prediction_times = []
        self.test_prediction_files = []

        self.child_score_times = dict()
        self.child_score_values = dict()
        self.child_held_out_prediction_times = dict()
        self.child_held_out_prediction_files = dict()
        self.child_held_out_idx = dict()
        self.child_valid_prediction_times = dict()
        self.child_valid_prediction_files = dict()
        self.child_test_prediction_times = dict()
        self.child_test_prediction_files = dict()

        self.communication_sleep = 0.1

        # TODO: Improve this hack!
        if agent == WarmLearnerAgent:
            self.immortal_offspring = True
        else:
            self.immortal_offspring = False

    def read_messages(self):
        for child_name, inbox in self.child_inboxes.iteritems():
            while True:
                try:
                    message = inbox.pop(0)
                except (IndexError, AttributeError):
                    break
                else:
                    if message['subject'] == 'score':
                        self.child_score_times[child_name].append(message['time'])
                        self.child_score_values[child_name].append(message['score'])
                    elif message['subject'] == 'predictions':
                        if message['partition'] == 'valid':
                            self.child_valid_prediction_times[child_name].append(message['time'])
                            self.child_valid_prediction_files[child_name].append(message['filename'])
                        elif message['partition'] == 'test':
                            self.child_test_prediction_times[child_name].append(message['time'])
                            self.child_test_prediction_files[child_name].append(message['filename'])
                        elif message['partition'] == 'held out':
                            self.child_held_out_idx[child_name] = message['idx']
                            self.child_held_out_prediction_times[child_name].append(message['time'])
                            self.child_held_out_prediction_files[child_name].append(message['filename'])
        while True:
            try:
                message = self.inbox.pop(0)
            except (IndexError, AttributeError):
                break
            else:
                self.standard_responses(message)
                # print(message)
                if message['subject'] == 'compute quantum':
                    # print('Cross validater received compute quantum message')
                    self.send_to_children(message)

    def first_action(self):
        self.create_children(classes=self.child_info)
        for child_name in self.child_processes.iterkeys():
            self.child_score_times[child_name] = []
            self.child_score_values[child_name] = []
            self.child_held_out_prediction_times[child_name] = []
            self.child_held_out_prediction_files[child_name] = []
            self.child_valid_prediction_times[child_name] = []
            self.child_valid_prediction_files[child_name] = []
            self.child_test_prediction_times[child_name] = []
            self.child_test_prediction_files[child_name] = []
        # self.broadcast_to_children(message=dict(subject='start'))
        self.start_children()

    def next_action(self):
        # Check mail
        self.read_messages()
        # Collect up scores and predictions - even if paused, children may still be finishing tasks
        min_n_scores = min(len(scores) for scores in self.child_score_values.itervalues())
        while len(self.score_values) < min_n_scores:
            n = len(self.score_values)
            num_scores = 0
            sum_scores = 0
            for child_scores in self.child_score_values.itervalues():
                # noinspection PyUnresolvedReferences
                if not np.isnan(child_scores[n]):
                    num_scores += 1
                    sum_scores += child_scores[n]
            score = sum_scores / num_scores
            # score = sum(scores[n] for scores in self.child_score_values.itervalues()) /\
            #         len(self.child_score_values)
            maxtime = max(times[n] for times in self.child_score_times.itervalues())
            self.score_values.append(score)
            self.score_times.append(maxtime)
            self.send_to_parent(dict(subject='score', sender=self.name,
                                   time=self.score_times[-1],
                                   score=self.score_values[-1]))

        # FIXME - send all of this data at the same time to prevent gotchas

        min_n_valid = min(len(times) for times in self.child_valid_prediction_times.itervalues())
        while len(self.valid_prediction_times) < min_n_valid:
            n = len(self.valid_prediction_times)
            predictions = None
            for child_name in self.child_score_times.iterkeys():
                filename = self.child_valid_prediction_files[child_name][n]
                child_predictions = np.load(filename)
                os.remove(filename)
                if predictions is None:
                    predictions = child_predictions
                else:
                    predictions += child_predictions
            predictions /= len(self.child_score_times)
            tmp_filename = util.random_temp_file_name('.npy')
            np.save(tmp_filename, predictions)
            maxtime = max(times[n] for times in self.child_valid_prediction_times.itervalues())
            self.valid_prediction_files.append(tmp_filename)
            self.valid_prediction_times.append(maxtime)
            self.send_to_parent(dict(subject='predictions', sender=self.name, partition='valid',
                                     time=self.valid_prediction_times[-1],
                                     filename=self.valid_prediction_files[-1]))

        min_n_test = min(len(times) for times in self.child_test_prediction_times.itervalues())
        while len(self.test_prediction_times) < min_n_test:
            n = len(self.test_prediction_times)
            predictions = None
            for child_name in self.child_score_times.iterkeys():
                filename = self.child_test_prediction_files[child_name][n]
                child_predictions = np.load(filename)
                os.remove(filename)
                if predictions is None:
                    predictions = child_predictions
                else:
                    predictions += child_predictions
            predictions /= len(self.child_score_times)
            tmp_filename = util.random_temp_file_name('.npy')
            np.save(tmp_filename, predictions)
            maxtime = max(times[n] for times in self.child_test_prediction_times.itervalues())
            self.test_prediction_files.append(tmp_filename)
            self.test_prediction_times.append(maxtime)
            self.send_to_parent(dict(subject='predictions', sender=self.name, partition='test',
                                     time=self.test_prediction_times[-1],
                                     filename=self.test_prediction_files[-1]))

        min_n_held_out = min(len(times) for times in self.child_held_out_prediction_times.itervalues())
        while len(self.held_out_prediction_times) < min_n_held_out:
            n = len(self.held_out_prediction_times)
            # FIXME - get rid of if else here
            if self.data_info['task'] == 'multiclass.classification':
                predictions = np.zeros(self.data['Y_train_1_of_k'].shape)
                # print('Prediction shape')
                # print(predictions.shape)
            else:
                predictions = np.zeros(self.data['Y_train'].shape)
            for child_name in self.child_score_times.iterkeys():
                filename = self.child_held_out_prediction_files[child_name][n]
                child_predictions = np.load(filename)
                os.remove(filename)
                predictions[self.child_held_out_idx[child_name]] = child_predictions
            # print('Combined predictions')
            # print(predictions[0])
            tmp_filename = util.random_temp_file_name('.npy')
            np.save(tmp_filename, predictions)
            maxtime = max(times[n] for times in self.child_held_out_prediction_times.itervalues())
            self.held_out_prediction_files.append(tmp_filename)
            self.held_out_prediction_times.append(maxtime)
            self.send_to_parent(dict(subject='predictions', sender=self.name, partition='held out',
                                     time=self.held_out_prediction_times[-1],
                                     filename=self.held_out_prediction_files[-1]))

        # Check to see if all children have terminated - if so, terminate this agent
        # immortal child dying is failure
        # mortal child dying without sending results is failure
        # any child failure should kill parent
        if self.immortal_offspring is True and len(self.conns_from_children) != len(self.child_states):
            logger.error("%s: Immortal child has died. Dying of grief", self.name)
            raise TerminationEx
        elif self.immortal_offspring is False:
            dead_kids = [x for x in self.child_states if x not in self.conns_from_children]
            for dk in dead_kids:
                if len(self.child_test_prediction_files[dk]) == 0:
                    logger.error("%s: Mortal child %s has died without sending results", self.name, dk)
                    raise TerminationEx
            if len(self.conns_from_children) == 0:
                logger.info("%s: No children remaining. Terminating.", self.name)
                raise TerminationEx


class WarmLearner(object):
    """Wrapper around things like random forest that don't have a warm start method"""
    def __init__(self, base_model, base_model_kwargs):
        self.base_model = base_model(**base_model_kwargs)
        self.model = copy.deepcopy(self.base_model)
        self.n_estimators = self.model.n_estimators

        self.first_fit = True

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def fit(self, X, y):
        if self.first_fit:
            self.model.fit(X, y)
            self.first_fit = False
        # Keep training and appending base estimators to main model
        while self.model.n_estimators < self.n_estimators:
            self.base_model.fit(X, y)
            self.model.estimators_ += self.base_model.estimators_
            self.model.n_estimators = len(self.model.estimators_)
        # Clip any extra models produced
        self.model.estimators_ = self.model.estimators_[:self.n_estimators]
        self.model.n_estimators = self.n_estimators