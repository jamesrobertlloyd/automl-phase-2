from __future__ import division
import glob
import shutil

__author__ = 'James Robert Lloyd, Emma Smith'
__description__ = 'Objects to coordinate stuff'

import constants
import os
import time
from collections import defaultdict
import psutil
# import matplotlib
import matplotlib.pyplot as plt
import functools
import pwd
# import shutil
import random
import copy
import sys
import traceback

import numpy as np
# import scipy.sparse
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os.path
import signal

# from automl_lib.data_manager import DataManager
from data_management import MemoryAwareDataManager as DataManager
from automl_lib import data_io

from agent import Agent, TerminationEx
from learners import CrossValidationAgent, WarmLearner, OneShotLearnerAgent
from metalearners import IndependentFreezeThaw, StackerV1
import util
# import tempfile
import logging
import subprocess

import libscores

import global_data  # Module to make data available to processes on startup

try:
    from pymongo import MongoClient
except ImportError as e:
    print "Import error - pymongo not installed", e.message

import zmq

# Set up logging for managers module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MemoryManager(Agent):
    """Implements utilities to monitor and manage memory of its children"""
    def __init__(self, overhead_memory=2**30, cgroup_soft_limit=6 * 2 ** 30, cgroup_hard_limit=2 * 2 ** 30,
                 **kwargs):
        super(MemoryManager, self).__init__(**kwargs)

        # Create variables that will be set by future managers
        self.meta_learners = []
        # Private memory - physical memory that is only accesible to the process (e.g. no shared libs or arrays)
        self.child_private_memory = dict()
        self.learner_preference = []
        self.pid = os.getpid()
        self.pgid = os.getpgid(0)

        self.cgroup = 'backstreet_bayes_learners'
        # As long as program exits properly control group will be deleted after use
        self.cgroup_mem_limit = None  # limit for cgroup - set during first action
        self.overhead_memory = overhead_memory  # cgroup is limited to available mem - overhead
        self.cgroup_soft_limit = cgroup_soft_limit  # amount required inside the cgroup
        self.cgroup_hard_limit = cgroup_hard_limit  # Point at which emergency pause occurs

        self.password = os.environ.get('PW', '')  # Required to create and manage control groups

    def first_action(self):
        signal.signal(signal.SIGTSTP, util.signal_handler)
        signal.signal(signal.SIGCONT, util.signal_handler)
        signal.signal(signal.SIGTERM, util.signal_handler)  # if termination signal received, tidy up.
        os.setpgid(0, 0)  # makes process and all children run in a separate process group, so orphans easily found
        self.pid = os.getpid()
        self.pgid = os.getpgid(0)

        # sys excepthook to attempt to log exceptions to logger rather than stderr
        def log_uncaught_exceptions(ex_cls, ex, tb):
            logging.error('An error happened and something died', exc_info=(ex_cls, ex, tb))

        sys.excepthook = log_uncaught_exceptions

        # How much memory is there?
        available = psutil.virtual_memory().available  # measured in bytes
        self.cgroup_mem_limit = int(available - self.overhead_memory)  # assume that manager and run.py fit into 0.5GB
        logger.info("Learners memory limit is %.2fGB", self.cgroup_mem_limit/2**30)

        # Create control group to limit resources
        try:
            subprocess.check_call(['which', 'cgexec'])
        except subprocess.CalledProcessError:
            # Install cgroup-bin
            installcg = "echo '{}' | sudo -S apt-get -y install cgroup-bin".format(self.password)
            retcode = subprocess.call(installcg, shell=True)
            if retcode != 0:
                logger.error("Cgroup-bin installation failed")
            else:
                logger.info("Installed cgroup-bin")

        user = pwd.getpwuid(os.getuid())[0]
        makecg = "echo '{}' | sudo -S cgcreate -a {} -g memory:/{}".format(self.password, user, self.cgroup)
        retcode = subprocess.call(makecg, shell=True)
        if retcode != 0:
            logger.error("Cgroup creation failed")

        # Limit memory for group
        with open('/sys/fs/cgroup/memory/{}/memory.limit_in_bytes'.format(self.cgroup), 'wb') as fp:
            fp.write(str(self.cgroup_mem_limit))
        with open('/sys/fs/cgroup/memory/{}/memory.swappiness'.format(self.cgroup), 'wb') as fp:
            fp.write(str(0))
        logger.info("Learners memory limit is %.2fGB", self.cgroup_mem_limit/2**30)

    def next_action(self):
        # Check to see if time limit reached
        time_taken = time.time() - self.start_time
        if time_taken > self.time_budget:
            logger.info('Time budget exceeded - terminating all processes')
            raise TerminationEx

        # Have any of the metalearners died?
        for learner in self.meta_learners:
            if self.child_states[learner] == 'terminated':
                logger.error('Metalearner has died')
                raise TerminationEx

        # Check to see if all children have terminated - if so, terminate the manager
        if not set(self.startable_children).difference(self.meta_learners):
            if self.saving_children:
                time.sleep(1)
            else:
                raise TerminationEx

        # Limit the amount of memory used - save worst processes if excessive
        self.limit_memory()

        # Check for orphans and make sure all children are in the cgroup
        self.orphan_finder()

        # Need to check how much memory is free
        available = psutil.virtual_memory().available  # measured in bytes
        with open('/sys/fs/cgroup/memory/{}/memory.usage_in_bytes'.format(self.cgroup), 'rb') as fp:
            cgroup_used = int(fp.read().strip())
        # Change memory limit for cgroup if current overhead is far from self.overhead
        if not 0.8*self.overhead_memory < available + cgroup_used - self.cgroup_mem_limit < 1.2*self.overhead_memory:
            new_limit = int(available + cgroup_used - self.overhead_memory)
            try:
                with open('/sys/fs/cgroup/memory/{}/memory.limit_in_bytes'.format(self.cgroup), 'wb') as fp:
                    fp.write(str(new_limit))
                    self.cgroup_mem_limit = new_limit
                    logger.info("Learners memory limit is %.2fGB", self.cgroup_mem_limit/2**30)
            except IOError:
                logger.error("Could not access cgroup memory.limit_in_bytes")

        # Housekeeping - kill any slow savers
        self.finish_saving_children()

    def tidy_up(self):
        super(MemoryManager, self).tidy_up()
        # Remove the control group
        delcg = "echo '{}' | sudo -S cgdelete memory:/{}".format(self.password, self.cgroup)
        try:
            subprocess.check_call(delcg, shell=True)
        except subprocess.CalledProcessError:
            logger.error("Removing the control group failed")

    def limit_memory(self, soft_limit=None, hard_limit=None):
        if soft_limit is None:
            soft_limit = self.cgroup_soft_limit
        if hard_limit is None:
            hard_limit = self.cgroup_hard_limit

        # How much memory in the cgroup is free?
        # Annoyingly, this is not real time and so only approximate
        with open('/sys/fs/cgroup/memory/{}/memory.usage_in_bytes'.format(self.cgroup), 'rb') as fp:
            used = int(fp.read().strip())
        available = self.cgroup_mem_limit - used  # in bytes
        logger.info("Available memory inside cgroup is %.1fGB", (available/2**30))
        self.monitor_memory()  # measured in kB

        if available < hard_limit:
            logger.warn("Emergency pause - memory too low")
            self.pause_children()
            # Kill slow savers:
            self.finish_saving_children(save_timeout=120)
            # Kill those that don't respond to save:
            for name, savetime in self.children_told_to_save.items():
                if self.child_flags[name].value == 1 and time.time() - savetime > 10:
                    self.terminate_children(names=[name])

            with open('/sys/fs/cgroup/memory/{}/memory.usage_in_bytes'.format(self.cgroup), 'rb') as fp:
                used = int(fp.read().strip())
            available = self.cgroup_mem_limit - used  # in bytes
            logger.info("Available memory inside cgroup is %.1fGB", (available/2**30))

            # Guess at how much will be freed by saving
            i = 0
            while available < hard_limit:
                if i >= len(self.learner_preference) - 1:
                    logger.warn("Only one learner left!")
                    break
                available += self.child_private_memory[self.learner_preference[i]] * 1024
                i += 1
            if i > 0:
                self.terminate_children(names=self.learner_preference[0:i])
            # Eventually:
            self.resume_children(names=self.meta_learners)
        elif available < soft_limit:
            logger.warn("Memory below soft limit")
            # Save the worst learners
            i = 0
            # Guess at how much will be freed by saving
            while available < soft_limit:
                if i >= len(self.learner_preference) - 1:
                    logger.warn("Only one learner left!")
                    break
                available += self.child_private_memory[self.learner_preference[i]] * 1024
                i += 1
            if i > 0:
                self.start_saving_children(names=self.learner_preference[0:i])

    def monitor_memory(self):
        for name in self.child_states.keys():
            child_private_memory = 0
            try:
                if self.child_processes[name].is_alive() is False:
                    self.child_private_memory[name] = 0
                    continue
                process = psutil.Process(pid=self.child_processes[name].pid)
                # memory for child
                with open('/proc/{}/smaps'.format(process.pid), 'r') as fp:
                    for line in fp.readlines():
                        if line.startswith("Private"):
                            child_private_memory += int(line.split()[1])
                # memory for child's children
                try:
                    for child in process.children(recursive=True):
                        try:
                            with open('/proc/{}/smaps'.format(child.pid), 'r') as fp:
                                for line in fp.readlines():
                                    if line.startswith("Private"):
                                        child_private_memory += int(line.split()[1])
                        except IOError as e:
                            # file not found, probably because we're not running on Linux
                            logger.warn("Proc file not found (error %s) for pid %d, child of %s. "
                                        "Memory usage cannot be calculated", e.strerror, child.pid, name)
                except (psutil.NoSuchProcess, psutil.AccessDenied, IOError) as e:
                    logger.warn("Error %s getting memory usage of children for child %s", e.strerror, name)

                self.child_private_memory[name] = child_private_memory
            except (psutil.NoSuchProcess, psutil.AccessDenied, IOError) as e:
                logger.warn("Error %s getting memory usage for child %s", e.strerror, name)
                self.child_private_memory[name] = 0

        logger.info("Child memory %s", str(self.child_private_memory))

    def orphan_finder(self):
        # Lists all the processes and finds ones in our group with parent 'init'
        # for ps in psutil.process_iter():
        #     if os.getpgid(ps.pid) == self.pgid:
        #         if ps.parent().name() == 'init':
        #             logger.error("Orphaned child with PID %d", ps.pid)
        #             ps.terminate()

        # More efficient - go through pids in tasks list
        with open('/sys/fs/cgroup/memory/{}/memory.usage_in_bytes'.format(self.cgroup), 'rb') as fp:
            for line in fp:
                pid = line.strip()
                if not pid:
                    continue
                pid = int(pid)
                try:
                    ps = psutil.Process(pid=pid)
                    if ps.parent().name() == 'init':
                        logger.error("Orphaned child with PID %d", pid)
                        ps.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
                    pass


class FixedLearnersStackingManager(MemoryManager):
    """Stacking a fixed set of base learners"""
    def __init__(self, input_dir, output_dir, basename, time_budget,
                 n_folds=5, compute_quantum=None, plot=False, debug=True, verbose=True, **kwargs):
        super(FixedLearnersStackingManager, self).__init__(**kwargs)

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.basename = basename
        self.time_budget = time_budget
        self.n_folds = n_folds
        self.compute_quantum = compute_quantum  # How long to run learners when cycling
        self.plot = plot
        self.debug = debug
        self.verbose = verbose

        # Make sure temporary file directory exists
        util.move_make(constants.SAVE_DIR)
        if self.plot and self.exp['movie']:
            util.move_make(constants.MOVIE_TEMP_DIR)

        # Load data
        logger.info('Loading data')
        # TODO - Don't use default methods to replace and filter data
        D = DataManager(basename, input_dir, replace_missing=True, filter_features=True)

        # Load global data variables
        global_data.data = D.data
        self.data = D.data

        global_data.exp = self.exp

        # Remember the size of the data
        self.n_data = D.data['X_train'].shape[0]

        # Record data info
        self.data_info = D.info

        # Should we use an experiment defined error metric?
        if not self.exp['error_metric'] is None:
            self.data_info['metric'] = self.exp['error_metric']
        # Choose which metric to use for model selection
        if self.data_info['metric'] == 'auc_metric':
            # FIXME - work out how to use automl's version of auc
            # self.data_info['eval_metric'] = 'automl auc'
            self.data_info['eval_metric'] = 'sklearn auc'
        elif self.data_info['metric'] == 'bac_metric':
            self.data_info['eval_metric'] = 'automl bac'
        elif self.data_info['metric'] == 'pac_metric':
            self.data_info['eval_metric'] = 'automl pac'
        elif self.data_info['metric'] == 'r2_metric':
            self.data_info['eval_metric'] = 'automl r2'
        elif self.data_info['metric'] == 'f1_metric':
            self.data_info['eval_metric'] = 'automl f1'
        elif self.data_info['metric'] == 'a_metric':
            self.data_info['eval_metric'] = 'automl a'
        else:
            logger.warning('Unrecognised error metric %s. Using AUC', self.data_info['metric'])
            self.data_info['eval_metric'] = 'sklearn auc'

        # For debugging
        # self.data_info['eval_metric'] = 'sklearn auc'
        # self.data_info['eval_metric'] = 'automl f1'

        # Init performance storage
        self.learner_score_values = defaultdict(list)
        self.learner_score_times = defaultdict(list)
        self.learner_held_out_pred_times = defaultdict(list)
        self.learner_held_out_pred_files = defaultdict(list)
        self.learner_valid_pred_times = defaultdict(list)
        self.learner_valid_pred_files = defaultdict(list)
        self.learner_test_pred_times = defaultdict(list)
        self.learner_test_pred_files = defaultdict(list)

        # Init stacking storage
        self.stacking_times = []
        self.stacking_true_times = []
        self.stacking_held_out_scores = []
        self.stacking_test_scores = []
        self.stacking_improvement_amounts = defaultdict(list)
        self.stacking_improvement_times = defaultdict(list)
        self.stacking_valid_files = []
        self.stacking_test_files = []

        # # Init prediction storage
        self.t = defaultdict(list)
        self.y = defaultdict(list)
        self.t_star = defaultdict(list)
        self.y_mean = defaultdict(list)
        self.y_covar = defaultdict(list)
        self.y_samples = defaultdict(list)

        # Utility of actions
        self.utilities = defaultdict(list)

        # # Init best learner storage
        self.best_learner = None
        self.best_time_index = None
        self.best_value = None

        # Record test set performance
        self.test_set_times = []
        self.test_set_compute_quanta = []
        self.test_set_scores = []
        self.compute_quanta = 0

        # Remember names of learners in some order
        self.learners = []
        self.meta_learners = []

        # Various state variables
        self.waiting = False
        self.next_learner_index = None
        self.current_learner_index = 0
        self.learners_asking_for_scores = []
        self.learners_asking_for_predictions = []
        self.broadcast_scores = True
        self.learner_preference = None
        self.current_running_learner = None
        self.learner_paused = False
        self.cycles = 0
        self.update_plots = False
        self.count = 0
        self.pred_count = 0
        self.learner_order = []

        # Other params
        self.communication_sleep = 0.01
        self.save_file = constants.SAVE_DIR + '/managerV1.pk'  # FIXME - what is this for?
        self.name = 'm'
        self.folds = KFold(n=self.n_data, n_folds=self.n_folds)
        self.first_time = True
        self.timeouts = defaultdict(functools.partial(util.identity, self.compute_quantum))
        self.one_shot_names = []
        self.checkpoint = None

    def update_db_common(self):
        """Updates the common collection in the database"""
        if not self.exp['use_db']:
            raise Exception('Database is not in use')
        self.db.common.update(dict(name='start_time'),
                              dict(name='start_time', value=self.start_time),
                              upsert=True)
        self.db.common.update(dict(name='remaining_time'),
                              dict(name='remaining_time', value=self.time_budget - (time.time() - self.start_time)),
                              upsert=True)
        self.db.common.update(dict(name='compute_quantum'),
                              dict(name='compute_quantum', value=self.compute_quantum),
                              upsert=True)

    def first_action(self):
        super(FixedLearnersStackingManager, self).first_action()

        if self.exp['use_db']:
            # Connect to database
            self.db_connection = MongoClient()
            # Wipe the database
            self.db_connection.drop_database('automl')
            self.db = self.db_connection.automl
            # Set common variables
            self.update_db_common()

        if self.exp['strategy'] == 'stack-RNN-train':
            # Set up communication to reinforcement learner
            context = zmq.Context()
            socket = context.socket(zmq.PAIR)
            socket.bind('tcp://*:%d' % constants.ZEROMQ_PORT)  # FIXME - is * safe / correct?
            self.zmq_socket = socket
        
        # Set compute quantum based on how many learners we will have
        compute_quantum_was_none = self.compute_quantum is None
        if self.compute_quantum is None:
            # We set this later
            self.compute_quantum = 1
        self.timeouts = defaultdict(functools.partial(util.identity, self.compute_quantum))

        # Create learners
        if self.data_info['task'] in ('binary.classification', 'multiclass.classification'):
            # Load learner specs
            super_fast_learners = eval(self.exp['super_fast_learners'])
            one_shot_algos = eval(self.exp['one_shot_algos'])
            anytime_algos = eval(self.exp['anytime_algos'])
            # Make super fast learner run on a data subset
            for i in range(len(super_fast_learners)):
                name, agent, kwargs = super_fast_learners[i]
                kwargs['subset_prop'] = min(1, 100 / self.data['X_train'].shape[0])
                super_fast_learners[i] = name, agent, kwargs
            # Add some standard fields to the specifications
            for a_list in [super_fast_learners, one_shot_algos, anytime_algos]:
                for i in range(len(a_list)):
                    name, agent, kwargs = a_list[i]
                    kwargs['folds'] = self.folds
                    kwargs['data_info'] = self.data_info
                    kwargs['exp'] = self.exp
                    a_list[i] = name, agent, kwargs
            # Start building up the learner specificationsa list
            learner_specifications = super_fast_learners
            # Add some small data versions of algorithms TODO - specify in exp
            one_shot_algos_100 = []
            # if self.exp['use_data_subsets']:
            #     if self.data['X_train'].shape[0] > 200:
            #         subset_prop = min(1, 100 / self.data['X_train'].shape[0])
            #         for spec in one_shot_algos:
            #             spec = list(copy.deepcopy(spec))
            #             spec[0] += '_100'
            #             spec[2]['subset_prop'] = subset_prop
            #             spec = tuple(spec)
            #             one_shot_algos_100.append(spec)
            #         learner_specifications += one_shot_algos_100
            one_shot_algos_1000 = []
            if self.exp['use_data_subsets']:
                if self.data['X_train'].shape[0] > 2000:
                    subset_prop = min(1, 1000 / self.data['X_train'].shape[0])
                    for spec in one_shot_algos:
                        spec = list(copy.deepcopy(spec))
                        spec[0] += '_1000'
                        spec[2]['subset_prop'] = subset_prop
                        spec = tuple(spec)
                        one_shot_algos_1000.append(spec)
                    learner_specifications += one_shot_algos_1000
            learner_specifications += one_shot_algos
            n_one_shot_algos = len(learner_specifications)
            anytime_algos_500 = []
            if self.exp['use_data_subsets']:
                if self.data['X_train'].shape[0] > 1000:
                    subset_prop = min(1, 500 / self.data['X_train'].shape[0])
                    for spec in anytime_algos:
                        spec = list(copy.deepcopy(spec))
                        spec[0] += '_500'
                        spec[2]['subset_prop'] = subset_prop
                        spec = tuple(spec)
                        anytime_algos_500.append(spec)
                else:
                    anytime_algos_500 = []
            # Look at how much time we have got and work out maximum number of anytime algos if given 15s each
            # FIXME - this 15s should be replaced with compute quantum
            if self.exp['subset_algos']:
                max_anytime_algos = self.time_budget * 0.1 / 15
                # Always run 4 algos
                if len(anytime_algos_500) > 0:
                    max_anytime_algos /= 2
                    max_anytime_algos = max(2, int(np.floor(max_anytime_algos)))
                else:
                    max_anytime_algos = max(4, int(np.floor(max_anytime_algos)))
            else:
                max_anytime_algos = len(anytime_algos)
            learner_specifications += anytime_algos[:max_anytime_algos]
            learner_specifications += anytime_algos_500[:max_anytime_algos]

            # Create timeout info
            for key in ['super_fast_timeout', 'one_shot_timeout', 'anytime_timeout']:
                if isinstance(self.exp[key], str):
                    self.exp[key] = eval(self.exp[key])
            for learner, _, _ in super_fast_learners:
                self.timeouts[learner] = self.exp['super_fast_timeout'] * self.compute_quantum
            for learner, _, _ in one_shot_algos_100 + one_shot_algos_1000 + one_shot_algos:
                self.timeouts[learner] = self.exp['one_shot_timeout'] * self.compute_quantum
            for learner, _, _ in anytime_algos_500 + anytime_algos:
                self.timeouts[learner] = self.exp['anytime_timeout'] * self.compute_quantum
        else:
            # DEBUG: copy dummy results to output folder
            print "Copying dummy files"
            for fl in glob.glob('./dummy_results/*') + glob.glob('./program/dummy_results/*'):
                shutil.copy(fl, self.output_dir)
                print fl
            raise Exception('I do not know what to do for task = %s' % self.data_info['task'])

        # Create meta learners
        if self.exp['strategy'] in ['FT', 'cycle']:
            meta_learner_names = ['freeze-thaw']
            meta_learners = [(IndependentFreezeThaw, dict(exp=self.exp))]
        else:
            meta_learner_names = ['stacker', 'freeze-thaw']
            meta_learners = [(StackerV1, dict(data_info=self.data_info, exp=self.exp)),
                             (IndependentFreezeThaw, dict(exp=self.exp))]


        # Add children
        learner_names = [spec[0] for spec in learner_specifications]
        learner_classes_and_kwargs = [(spec[1], spec[2]) for spec in learner_specifications]
        self.create_children(new_names=learner_names, classes=learner_classes_and_kwargs)
        self.learners = learner_names

        if compute_quantum_was_none:
            old_compute_quantum = self.compute_quantum
            # Spend 10% of time in the cycling phase on anytime algos - ignore one shot algo time
            time_remaining = self.time_budget - (time.time() - self.start_time)
            self.compute_quantum = time_remaining * 0.1 / len(anytime_algos_500[:max_anytime_algos] +
                                                              anytime_algos[:max_anytime_algos])
            self.compute_quantum = max(15, self.compute_quantum)  # Always give things at least 15 seconds
            self.send_to_children(dict(subject='compute quantum', compute_quantum=self.compute_quantum))
            increase_ratio = self.compute_quantum / old_compute_quantum
            for name in self.timeouts.iterkeys():
                self.timeouts[name] *= increase_ratio

        # FIXME - the following code is clearly in the wrong place
        # Only spend 10% of time on one shot algos
        self.one_shot_names = [name for (name, _, _) in learner_specifications[:n_one_shot_algos]]
        if compute_quantum_was_none:
            time_remaining = self.time_budget - (time.time() - self.start_time)
            timeout = time_remaining * 0.1 / n_one_shot_algos
            for name in self.one_shot_names:
                print(name)
                self.timeouts[name] = timeout
            # Always got time for the first algo
            self.timeouts[self.one_shot_names[0]] = np.inf

        # Create database entries for learners
        if self.exp['use_db']:
            for learner in self.learners:
                self.db.learner_scores.insert_one(dict(learner=learner, times=[], scores=[]))
                self.db.learner_valid_pred.insert_one(dict(learner=learner, times=[], filenames=[]))
                self.db.learner_test_pred.insert_one(dict(learner=learner, times=[], filenames=[]))
                self.db.learner_held_out_pred.insert_one(dict(learner=learner, times=[], filenames=[]))

        self.meta_learners = self.create_children(new_names=meta_learner_names, classes=meta_learners,
                                                  use_cgroup=False)
        # Start the meta learning - it will always run except in a memory panic
        self.start_children(names=self.meta_learners)

        # Tell everyone the original compute quantum TODO - write to db
        self.send_to_children(dict(subject='original compute quantum', compute_quantum=self.compute_quantum))

        # Initially cycle through all of the learners to gather some data
        self.state = 'cycling'
        # self.state = 'random switching'
        self.next_learner_index = 0
        # Initialise the learner preference
        self.learner_preference = self.learners[::-1]
        # We are ready to start a learner running
        self.waiting = False
        # stop tkinter hiding the termination exception
        if self.plot and plt.get_backend() == 'TkAgg':
            import Tkinter as Tk

            def show_error(exc, val, tb):
                if exc == TerminationEx:
                    print "-----Termination in tkinter------"
                    raise TerminationEx

                sys.stderr.write("Exception in Tkinter callback\n")
                sys.last_type = exc
                sys.last_value = val
                sys.last_traceback = tb
                traceback.print_exception(exc, val, tb)

            Tk.Tk.report_callback_exception = show_error

    def read_learning_curve_predictions(self, agent='freeze-thaw'):
        """Read learning curve predictions form the database"""
        cursor = self.db.learning_curve_predictions.find()
        for entry in cursor:
            if entry['agent'] == agent:
                self.t[entry['learner']] = entry['t']
                self.y[entry['learner']] = entry['y']
                self.t_star[entry['learner']] = entry['t_star']
                self.y_mean[entry['learner']] = entry['y_mean']
                self.y_covar[entry['learner']] = entry['y_covar']
                self.y_samples[entry['learner']] = entry['y_samples']

    def read_messages(self):
        for child_name, inbox in self.child_inboxes.iteritems():
            while True:
                try:
                    message = inbox.pop(0)
                except (IndexError, AttributeError):
                    break
                else:
                    logger.debug('Reading message %s from %s', message['subject'], child_name)
                    if message['subject'] == 'score':
                        self.update_plots = True
                        self.learner_score_times[child_name].append(message['time'])
                        self.learner_score_values[child_name].append(message['score'])
                    elif message['subject'] == 'predictions':
                        if message['partition'] == 'valid':
                            self.learner_valid_pred_times[child_name].append(message['time'])
                            self.learner_valid_pred_files[child_name].append(message['filename'])
                        elif message['partition'] == 'test':
                            self.learner_test_pred_times[child_name].append(message['time'])
                            self.learner_test_pred_files[child_name].append(message['filename'])
                            # print(len(self.learner_test_pred_files[child_name]))
                        elif message['partition'] == 'held out':
                            if self.waiting:
                                self.waiting = False
                                if self.exp['strategy'] == 'stack-RNN-train':
                                    # self.pause_children(names=[child_name])
                                    self.pause_children(names=[self.current_running_learner])
                                    self.checkpoint = time.time()
                            self.learner_held_out_pred_times[child_name].append(message['time'])
                            self.learner_held_out_pred_files[child_name].append(message['filename'])
                            self.learner_order.append(child_name)
                            self.compute_quanta += 1
                    elif message['subject'] == 'computation preference':
                        if len(message['preference']) > 0:
                            if (child_name == 'freeze-thaw' and self.exp['strategy'] == 'FT') or \
                               (child_name == 'freeze-thaw' and self.exp['strategy'] == 'stack-FT') or \
                               (child_name == 'stacker' and self.exp['strategy'] == 'stack-meta'):
                                if self.state == 'cycling' and self.cycles > 0:
                                    # We have received info from the meta learner - time to stop cycling
                                    self.state = 'following orders'
                                if self.cycles > 0:
                                    # TODO - is this always going to be the correct behaviour?
                                    self.learner_preference = message['preference']
                    elif message['subject'] == 'meta predictions':
                        # print('Getting predictions from freeze thaw')
                        self.update_plots = True
                        self.t = util.dict_of_lists_to_arrays(message['t'])
                        self.y = util.dict_of_lists_to_arrays(message['y'])
                        self.t_star = util.dict_of_lists_to_arrays(message['t_star'])
                        self.y_mean = util.dict_of_lists_to_arrays(message['y_mean'])
                        self.y_covar = util.dict_of_lists_to_arrays(message['y_covar'])
                        self.y_samples = message['y_samples']
                        # print('Success')
                    elif message['subject'] == 'prediction selection':
                        if (child_name == 'freeze-thaw' and self.exp['strategy'] == 'FT') or \
                           (child_name == 'freeze-thaw' and self.exp['strategy'] == 'cycle') or \
                           (child_name == 'stacker' and self.exp['strategy'] == 'stack-FT') or \
                           (child_name == 'stacker' and self.exp['strategy'] == 'cycle-stack') or \
                           (child_name == 'stacker' and self.exp['strategy'] == 'stack-meta') or \
                           (child_name == 'stacker' and self.exp['strategy'] == 'stack-RNN-train'):
                            updated_opinion = False
                            old_best_learner = self.best_learner
                            old_best_time = self.best_time_index
                            self.best_learner = message['learner']
                            self.best_time_index = message['time_index']
                            if (not self.best_learner == old_best_learner) or (not self.best_time_index == old_best_time):
                                updated_opinion = True
                            self.best_value = message['value']
                            logger.info('New best predictions : %s at %d with value %f',
                                        self.best_learner,
                                        self.best_time_index,
                                        self.best_value)
                            if updated_opinion:
                                # Save best predictions to output dir - remembering to change format
                                if 'X_valid' in self.data:
                                    np_file = self.learner_valid_pred_files[self.best_learner][self.best_time_index]
                                    automl_file = os.path.join(self.output_dir,
                                                               self.basename + '_valid_%03d' % self.pred_count + '.predict')
                                    pred = np.load(np_file)
                                    data_io.write(automl_file, pred)
                                if 'X_test' in self.data:
                                    np_file = self.learner_test_pred_files[self.best_learner][self.best_time_index]
                                    automl_file = os.path.join(self.output_dir,
                                                               self.basename + '_test_%03d' % self.pred_count + '.predict')
                                    pred = np.load(np_file)
                                    data_io.write(automl_file, pred)
                                self.pred_count += 1
                            if 'X_test' in self.data:
                                # Load predictions
                                np_file = self.learner_test_pred_files[self.best_learner][self.best_time_index]
                                pred = np.load(np_file)
                                if (self.data_info['task'] == 'multiclass.classification' and
                                    'Y_test_1_of_k' in self.data) or ('Y_test' in self.data):
                                    # Score
                                    if self.data_info['task'] == 'multiclass.classification':
                                        y_test = self.data['Y_test_1_of_k']
                                    else:
                                        y_test = self.data['Y_test']
                                    test_score = libscores.eval_metric(metric=self.data_info['eval_metric'],
                                                                       truth=y_test,
                                                                       predictions=pred,
                                                                       task=self.data_info['task'])
                                    self.test_set_scores.append(test_score)
                                    self.test_set_times.append(time.time() - self.start_time)
                                    self.test_set_compute_quanta.append(self.compute_quanta)
                                    if self.exp['score_dir'] is not None:
                                        with open(os.path.join(self.exp['score_dir'],
                                                               'learning_curve.csv'), 'a') as score_file:
                                            score_file.write('%f,%f\n' % (self.test_set_times[-1],
                                                                          self.test_set_scores[-1]))
                        else:
                            # Predictions from the wrong person - ignore
                            pass
                    elif message['subject'] == 'scores please':
                        if not child_name in self.learners_asking_for_scores:
                            self.learners_asking_for_scores.append(child_name)
                    elif message['subject'] == 'predictions please':
                        if not child_name in self.learners_asking_for_predictions:
                            # print(child_name + ' asked for predictions')
                            self.learners_asking_for_predictions.append(child_name)
                    elif message['subject'] == 'stacking performance':
                        # print('Received stacking performance')
                        self.update_plots = True
                        self.stacking_times.append(message['time'])
                        self.stacking_held_out_scores.append(message['held_out_score'])
                        self.stacking_test_scores.append(message['test_score'])
                        self.stacking_valid_files.append(message['valid_pred_file'])
                        self.stacking_test_files.append(message['test_pred_file'])

                        # Record the true time taken
                        self.stacking_true_times.append(time.time() - self.start_time)

                        # Save these in an alternative format
                        self.learner_valid_pred_files['stacker'].append(message['valid_pred_file'])
                        self.learner_test_pred_files['stacker'].append(message['test_pred_file'])

                        # # Save latest predictions
                        # updated_opinion = True  # FIXME - make this a real thing
                        # # FIXME - allow the stacker to choose a single algorithm if that looks best
                        # if updated_opinion:
                        #     # Save best predictions to output dir - remember to change format!
                        #     if 'X_valid' in self.data:
                        #         np_file = self.stacking_valid_files[-1]
                        #         automl_file = os.path.join(self.output_dir,
                        #                                    self.basename + '_valid_%03d' % self.pred_count + '.predict')
                        #         pred = np.load(np_file)
                        #         data_io.write(automl_file, pred)
                        #     if 'X_test' in self.data:
                        #         np_file = self.stacking_test_files[-1]
                        #         automl_file = os.path.join(self.output_dir,
                        #                                    self.basename + '_test_%03d' % self.pred_count + '.predict')
                        #         pred = np.load(np_file)
                        #         data_io.write(automl_file, pred)
                        #     self.pred_count += 1
                        # print('Success')
                    elif message['subject'] == 'stacking blame':
                        self.stacking_improvement_amounts = message['amounts']
                        self.stacking_improvement_times = message['times']
                    elif message['subject'] == 'stacking stats':
                        pass
                    elif message['subject'] == 'utilities':
                        self.utilities = message['utilities']
                    elif message['subject'] == 'time taken' and message['sender'] == 'stacker':
                        if message['time'] > self.compute_quantum and not self.exp['compute_quantum_fixed']:
                            logger.info('Increasing compute quantum to %fs' % (message['time'] * 1.1))
                            increase_ratio = message['time'] * 1.1 / self.compute_quantum
                            self.compute_quantum = message['time'] * 1.1
                            # print('New CQ = %f' % self.compute_quantum)
                            self.send_to_children(dict(subject='compute quantum', compute_quantum=self.compute_quantum),
                                                  [self.current_running_learner])
                            # Scale timeouts appropriately
                            for name in self.timeouts.iterkeys():
                                # Only scale up the iterative algorithm times - still want the one shots to only take
                                # up 10% of time
                                if name not in self.one_shot_names:
                                    self.timeouts[name] *= increase_ratio
                    elif message['subject'] == 'finished stacking':
                        if self.state == 'waiting for stack' and self.exp['strategy'] == 'stack-RNN-train':
                            self.state = 'talking to RNN'

    def next_action(self):
        super(FixedLearnersStackingManager, self).next_action()
        time_taken = time.time() - self.start_time
        # Read messages and take appropriate responses
        logger.debug('Reading messages')
        self.read_messages()
        logger.debug('Finished reading messages')

        # Tell the meta learners about the current state if it has asked
        if self.broadcast_scores and len(self.learners_asking_for_scores):
            logger.debug('Broadcasting scores')
            have_data = False
            for child_name in self.learners:
                if len(self.learner_held_out_pred_files[child_name]) > 0:
                    have_data = True
            if have_data:
                while len(self.learners_asking_for_scores) > 0:
                    name = self.learners_asking_for_scores.pop()
                    # if constants.DEBUG:
                    #     print('\n\n\n\nBroadcasting scores %s\n\n\n\n' % name)
                    self.send_to_children(dict(subject='scores',
                                               learner_score_values=self.learner_score_values,
                                               learner_score_times=self.learner_score_times,
                                               learner_order=self.learner_order,
                                               learners=self.learners,
                                               learner_held_out_prediction_times=self.learner_held_out_pred_times,
                                               learner_held_out_prediction_files=self.learner_held_out_pred_files,
                                               learner_valid_prediction_times=self.learner_valid_pred_times,
                                               learner_valid_prediction_files=self.learner_valid_pred_files,
                                               learner_test_prediction_times=self.learner_test_pred_times,
                                               learner_test_prediction_files=self.learner_test_pred_files,
                                               remaining_time=self.time_budget - time_taken,
                                               compute_quantum=self.compute_quantum),
                                          names=[name])

        # Tell meta learners about predictions about learners
        logger.debug('Broadcasting predictions')
        while len(self.learners_asking_for_predictions) > 0:
            name = self.learners_asking_for_predictions.pop()
            # print('Sending predictions to ' + name)
            self.send_to_children(dict(subject='predictions',
                                       times=self.t_star,
                                       means=self.y_mean,
                                       covar=self.y_covar,
                                       samples=self.y_samples),
                                  names=[name])

        # If the current running learner isn't startable, we shouldn't be waiting for it
        if self.current_running_learner not in self.startable_children:
            self.waiting = False
        else:  # in case it isn't running for some reason (e.g. it got paused, or it saved really quickly)
            # TODO - work out the correct logic here
            if not self.exp['strategy'] == 'stack-RNN-train':
                self.start_children(names=[self.current_running_learner])

        # Check to see if anything is running slowly and should be terminated
        if self.current_running_learner is not None:
            logger.debug('Checking for slow processes')
            timeout = 2 * self.timeouts[self.current_running_learner] + self.communication_sleep
            # if constants.DEBUG:
            #     print('\n\n\n\nTimeout = %f Child running = %f\n\n\n\n' %
            #           (timeout, time.time() - self.last_child_start_time))
            if time.time() - self.last_child_start_time > timeout:
                if len(self.learner_test_pred_files[self.current_running_learner]) == 0:
                    self.terminate_children([self.current_running_learner])
                    self.waiting = False
                    logger.info('Terminated child %s since it did not produce predictions in %f seconds',
                                self.current_running_learner,
                                timeout)

        # Pausing and resuming of learners
        if self.state == 'cycling':
            logger.debug('Cycling')
            if not self.waiting:
                if self.current_running_learner is not None:
                    self.pause_children(names=[self.current_running_learner])
                    self.learner_preference = [self.learners[i % len(self.learners)]
                                               for i in range(self.current_learner_index,
                                                              self.current_learner_index-len(self.learners), -1)]

                self.current_running_learner = self.learners[self.next_learner_index]  # *name* of current
                self.current_learner_index = self.next_learner_index
                self.next_learner_index = (self.next_learner_index + 1) % len(self.learners)

                if self.current_running_learner in self.startable_children:
                    started = self.start_children(names=[self.current_running_learner])
                    if started == [self.current_running_learner]:  # then selected one was started
                        # success = True
                        self.waiting = True
                        # print('Manager sent compute quantum message to %s' % self.current_running_learner)
                        self.send_to_children(dict(subject='compute quantum', compute_quantum=self.compute_quantum),
                                              [self.current_running_learner])

                if self.current_learner_index == 0 and not self.first_time:
                    self.cycles += 1
                    if self.exp['strategy'] == 'stack-RNN-train':
                        self.state = 'waiting for stack'
                        self.pause_children(names=[self.current_running_learner])

                self.first_time = False

            # self.waiting = True
        elif self.state == 'following orders':
            logger.debug('Following orders')
            favourite_child = None
            for learner in reversed(self.learner_preference):
                if learner in self.startable_children:
                    favourite_child = learner
                    break
            if favourite_child is None:
                # Can't run any of the children
                logger.warn('None of the learners can be run')
            else:
                self.start_children(names=[favourite_child])
                if not self.current_running_learner == favourite_child:
                    self.pause_children(names=[self.current_running_learner])
                    self.send_to_children(dict(subject='compute quantum', compute_quantum=self.compute_quantum),
                                          [favourite_child])
                self.current_running_learner = favourite_child
        elif self.state == 'random switching':
            logger.debug('Random switching')
            if not self.waiting:
                # Start a new child at random
                # FIXME - random children include the meta learners!
                proposed_learner = random.sample(self.startable_children, 1)
                logger.debug('Current running learner : %s', self.current_running_learner)
                if not proposed_learner == self.current_running_learner:
                    # Pause child
                    if self.current_running_learner is not None:
                        self.pause_children(names=[self.current_running_learner])
                    # Start new child
                    self.current_running_learner = proposed_learner
                    self.start_children(names=[self.current_running_learner])
                    self.send_to_children(dict(subject='compute quantum', compute_quantum=self.compute_quantum),
                                          [self.current_running_learner])

                # Wait for it to finish
                self.waiting = True
        elif self.state == 'talking to RNN':
            # Send message about current state
            logger.info('Talking to RNN')
            t_star_json = dict()
            y_mean_json = dict()
            y_covar_json = dict()
            for name in self.y_covar.iterkeys():
                t_star_json[name] = self.t_star[name].tolist()
                y_mean_json[name] = self.y_mean[name].tolist()
                y_covar_json[name] = self.y_covar[name].tolist()

            message = dict(learners=self.learners,
                           learning_curves=dict(times=self.learner_score_times,
                                                scores=self.learner_score_values),
                           stacking_performance=self.stacking_held_out_scores,
                           time_remaining=(self.time_budget - (time.time() - self.start_time)) /
                                          self.compute_quantum,
                           estimated_utilities=self.utilities,
                           learning_curves_predictions=dict(times=t_star_json,
                                                            means=y_mean_json,
                                                            covars=y_covar_json)
                           )
            # print(message)
            self.checkpoint = time.time()  # TODO - this should be in read_messages()
            self.zmq_socket.send_json(message)
            # Receive message back
            message = self.zmq_socket.recv_json()
            # Add to time budget to account for wasted time
            self.time_budget += time.time() - self.checkpoint
            # Start running the requested learner
            proposed_learner = message['selection']
            self.current_running_learner = proposed_learner
            self.start_children(names=[self.current_running_learner])
            self.send_to_children(dict(subject='compute quantum', compute_quantum=self.compute_quantum),
                                  [self.current_running_learner])
            # Wait for the learner and the stack to complete
            self.state = 'waiting for stack'

        logger.info("Learner preference %s", str(self.learner_preference))

        try:

            # Plot what's happening
            if self.plot and self.update_plots:
                self.update_plots = False
                fig = plt.figure(num=1, figsize=(28, 12))
                fig.clf()
                ax = fig.add_subplot(151)
                # ax = fig.add_subplot(111)
                ax.set_title('Learning curves')
                ax.set_xlabel('Time (seconds)')
                # ax.set_xscale('log')
                ax.set_ylabel('Score')
                for i, child in enumerate(self.learners):
                    if len(self.learner_held_out_pred_files[child]) > 0:
                        alpha = 1
                    else:
                        alpha = 0.5
                    # This algorithm has made predictions - or is anytime -  we are allowed to show it
                    ax.plot(self.learner_score_times[child], self.learner_score_values[child],
                            color=util.colorbrew(i),
                            linestyle='dashed', marker='o',# edgecolors='none',
                            label=child, alpha=alpha)
                    if child == self.best_learner:
                        # Have a gold star
                        try:
                            ax.plot(self.learner_test_pred_times[child][self.best_time_index], self.best_value,
                                    color=util.colorbrew(i),
                                    marker='*', markersize=30)
                        except:
                            ax.plot(0, self.best_value,
                                    color=util.colorbrew(i),
                                    marker='*', markersize=30)
                # leg = ax.legend(loc='best')
                # leg.get_frame().set_alpha(0.5)

                # ax2 = fig.add_subplot(122, sharey=ax)
                ax2 = fig.add_subplot(152)
                ax2.set_title('Learning curves and predictions')
                ax2.set_xlabel('Time (seconds)')
                # ax2.set_xscale('log')
                ax2.set_ylabel('Score')
                data = False
                for i, child in enumerate(self.learners):
                    if len(self.t[child]) > 0:
                        data = True
                        ax2.plot(self.t[child].ravel(), self.y[child].ravel(),
                                 color=util.colorbrew(i),
                                 linestyle='dashed', marker='o',# edgecolors='none',
                                 label=child)
                        if len(self.t_star[child]):
                            # ax2.plot(self.t_star[child].ravel(), self.y_mean[child].ravel(),
                            #          color=util.colorbrew(i),
                            #          linestyle='-', marker='')
                            # ax2.fill_between(self.t_star[child],
                            #                  self.y_mean[child].ravel() - np.sqrt(np.diag(self.y_covar[child])),
                            #                  self.y_mean[child].ravel() + np.sqrt(np.diag(self.y_covar[child])),
                            #                  color=util.colorbrew(i),
                            #                  alpha=0.2)
                            for _ in range(3):
                                sample = random.choice(self.y_samples[child])
                                ax2.plot(self.t_star[child].ravel(), sample.ravel(),
                                         marker='o', color=util.colorbrew(i),
                                         alpha=0.5, linestyle='-')
                        if child == self.best_learner:
                            # Have a gold star
                            try:
                                ax2.plot(self.learner_test_pred_times[child][self.best_time_index], self.best_value,
                                         color=util.colorbrew(i),
                                         marker='*', markersize=30)
                            except:
                                ax2.plot(0, self.best_value,
                                         color=util.colorbrew(i),
                                         marker='*', markersize=30)
                # if data:
                #     leg = ax2.legend(loc='best')
                #     leg.get_frame().set_alpha(0.5)

                ax3 = fig.add_subplot(153)
                ax3.set_title('Estimated utilities per time step')
                ax3.set_xlabel('Time (seconds)')
                # ax.set_xscale('log')
                ax3.set_ylabel('Utility rates')
                data = False
                for i, child in enumerate(self.learners):
                    if len(self.utilities[child]) > 0:
                        data = True
                        ax3.plot(np.arange(1, 1 + len(self.utilities[child]), 1) * self.compute_quantum,
                                 self.utilities[child],
                                 color=util.colorbrew(i),
                                 linestyle='dashed', marker='o',# edgecolors='none',
                                 label=child)
                # if data:
                #     leg = ax3.legend(loc='best')
                #     leg.get_frame().set_alpha(0.5)


                if len(self.stacking_times) > 0:
                    ax4 = fig.add_subplot(154, sharey=ax)
                    ax4.set_title('Stacking performance')
                    ax4.set_xlabel('Time (seconds)')
                    # ax.set_xscale('log')
                    ax4.set_ylabel('Score')
                    ax4.plot(self.stacking_true_times, self.stacking_held_out_scores,
                             linestyle='dashed', marker='o',
                             color=util.colorbrew(0),
                             label='Estimated')
                    if len(self.stacking_test_scores) > 0 and not self.stacking_test_scores[0] is None:
                        ax4.plot(self.stacking_true_times, self.stacking_test_scores,
                                 linestyle='dashed', marker='o',
                                 color=util.colorbrew(1),
                                 label='Test')
                    if self.best_learner == 'stacker':
                        # Have a gold star
                        ax4.plot(self.stacking_true_times[self.best_time_index], self.best_value,
                                 color=util.colorbrew(0),
                                 marker='*', markersize=30)
                    # leg = ax4.legend(loc='best')
                    # leg.get_frame().set_alpha(0.5)

                if len(self.test_set_scores) > 0:
                    ax5 = fig.add_subplot(155, sharey=ax)
                    ax5.set_title('Test set performance')
                    ax5.set_xlabel('Time (seconds)')
                    ax5.set_ylabel('Score')
                    ax5.plot(self.test_set_times, self.test_set_scores,
                             linestyle='dashed', marker='o',
                             color=util.colorbrew(0))

                # Set axes of fig 2
                ax2.set_ylim(ax.get_ylim())

                # fig = plt.figure(2)
                # fig.clf()
                # ax = fig.add_subplot(122, sharey=ax)
                # ax = fig.add_subplot(133)
                # ax.set_title('Stacking marginal improvements')
                # ax.set_xlabel('Time (seconds)')
                # # ax.set_xscale('log')
                # # ax.set_yscale('log')
                # ax.set_ylabel('Score')
                # times_values = []
                # for i, child in enumerate(self.learners):
                #     ax.plot(self.stacking_improvement_times[child], self.stacking_improvement_amounts[child],
                #              color=util.colorbrew(i),
                #              linestyle='dashed', marker='o',# edgecolors='none',
                #              label=child)
                #     for a_time, imp in zip(self.stacking_improvement_times[child], self.stacking_improvement_amounts[child]):
                #         times_values.append((a_time, imp))
                # times_values.sort(key=lambda x: x[0])
                # times_values = times_values[-20:]
                # values = [value for _, value in times_values]
                # if len(times_values) == 20:
                #     ax.set_ylim([min(values), max(values)])
                # leg = ax.legend(loc='best')
                # leg.get_frame().set_alpha(0.5)

                plt.draw()

                if self.exp['movie']:
                    fig.savefig(os.path.join(constants.MOVIE_TEMP_DIR, '%s_%06d.png' % (self.basename, self.count)))
                self.count += 1
                plt.show(block=False)

        except TerminationEx:
            raise TerminationEx
        except:
            logger.error('Plotting failed :(')

        # time_diff = time.time() - time_taken - self.start_time
        # if time_diff < 2:
        #     time.sleep(int(2-time_diff))

if __name__ == '__main__':
    print('Try running an experiment or the demo')
