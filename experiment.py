"""
Main file for setting up experiments, and compiling results.

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          
Created May 2015
"""

# TODO
# - use zeromq communication
# - make stacking tree code stuff easy - add weights as features that are recorded
# - try a test e.g. cycle vs freeze thaw
# - reintroduce database

import os
import sys
import psutil
import logging
from multiprocessing import Process
import numpy as np
import select
import signal

from managers import FixedLearnersStackingManager
import agent
import constants
import util

def load_experiment_details(filename):
    """Just loads the exp dictionary"""
    exp_string = open(filename, 'r').read()
    exp = eval(exp_string)
    exp = exp_param_defaults(exp)
    return exp


def run_experiment_file(filename, plot_override=True, separate_process=False):
    """
    This is intended to be the function that's called to initiate a series of experiments.
    """
    exp = load_experiment_details(filename=filename)
    print('BEGIN EXPERIMENT SPECIFICATIONS')
    print(exp_params_to_str(exp))
    print('END EXPERIMENT SPECIFICATIONS')

    # # Set number of processors
    p = psutil.Process()
    all_cpus = list(range(psutil.cpu_count()-1))
    p.cpu_affinity(all_cpus)

    # Set up logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    form = logging.Formatter("[%(levelname)s/%(processName)s] %(asctime)s %(message)s")

    # Handler for logging to stderr
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.WARN)  # set level here
    # sh.addFilter(ProcessFilter())  # filter to show only logs from manager
    sh.setFormatter(form)
    root_logger.addHandler(sh)

    # Handler for logging to file
    util.move_make_file(constants.LOGFILE)
    fh = logging.handlers.RotatingFileHandler(constants.LOGFILE, maxBytes=512*1024*1024)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(form)
    root_logger.addHandler(fh)

    # Make output dir
    util.move_make(exp['output_dir'])

    # Make score dir and learning curve
    if exp['score_dir'] is not None:
        util.move_make(exp['score_dir'])
        with open(os.path.join(exp['score_dir'],
                               'learning_curve.csv'), 'w') as score_file:
            score_file.write('Time,Score\n')

    # Record start time
    open(os.path.join(exp['output_dir'], exp['basename'] + '.firstpost'), 'wb').close()

    # Plotting?
    if plot_override is not None:
        exp['plot'] = plot_override

    # Start manager
    mgr = FixedLearnersStackingManager(exp['input_dir'], exp['output_dir'], exp['basename'],
                                       exp['time_budget'],
                                       compute_quantum=exp['compute_quantum'], plot=exp['plot'],
                                       overhead_memory=constants.OVERHEAD,
                                       cgroup_soft_limit=constants.CGROUP_SOFT_LIMIT,
                                       cgroup_hard_limit=constants.CGROUP_HARD_LIMIT,
                                       exp=exp)

    if separate_process:

        # Create process
        p = Process(target=agent.start_communication, kwargs=dict(agent=mgr))
        p.name = 'manager'
        p.start()

        print('\nPress enter to terminate at any time.\n')
        while True:
            if not p.is_alive():
                break

            # Wait for one second to see if any keyboard input
            i, o, e = select.select([sys.stdin], [], [], 1)
            if i:
                print('\n\nTerminating')
                try:
                    ps = psutil.Process(pid=p.pid)
                    ps.send_signal(signal.SIGTERM)
                    p.join(timeout=5)
                    if p.is_alive():
                        print("Didn't respond to SIGTERM")
                        util.murder_family(pid=p.pid, killall=True, sig=signal.SIGKILL)
                except psutil.NoSuchProcess:
                    pass  # already dead
                break

    else:
        mgr.communicate()


def exp_param_defaults(exp_params):
    """Sets all missing parameters to their default values"""
    defaults = dict(subset_algos=False,
                    error_metric=None,
                    compute_quantum_fixed=False,
                    score_dir=None,
                    slowdown_factor=1,
                    plot=False,
                    movie=False,
                    use_db=False,
                    strategy='stack-meta',
                    super_fast_subset=1000,
                    super_fast_timeout=np.inf,
                    one_shot_timeout=0.333,
                    anytime_timeout=1,
                    use_data_subsets=True,
                    super_fast_learners='''[
        ('LR-100-subset', CrossValidationAgent, dict(learner=LogisticRegression,
                                                     learner_kwargs=dict(C=100),
                                                     agent=OneShotLearnerAgent,
                                                     agent_kwargs=dict(),
                                                     feature_subset=10))
                        ]''',
                    one_shot_algos='''[
        ('LR-0.01', CrossValidationAgent, dict(learner=LogisticRegression,
                                               learner_kwargs=dict(C=0.01),
                                               agent=OneShotLearnerAgent,
                                               agent_kwargs=dict())),
        ('LR-100', CrossValidationAgent, dict(learner=LogisticRegression,
                                                  learner_kwargs=dict(C=100),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict())),
        ('KNN-1', CrossValidationAgent, dict(learner=KNeighborsClassifier,
                                                  learner_kwargs=dict(n_neighbors=1),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict())),
        ('KNN-5', CrossValidationAgent, dict(learner=KNeighborsClassifier,
                                                  learner_kwargs=dict(n_neighbors=3),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict())),
        ('KNN-25', CrossValidationAgent, dict(learner=KNeighborsClassifier,
                                                  learner_kwargs=dict(n_neighbors=9),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict())),
        ('GNB', CrossValidationAgent, dict(learner=GaussianNB,
                                                  learner_kwargs=dict(),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict())),
        ('DTC-1', CrossValidationAgent, dict(learner=DecisionTreeClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=1),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict())),
        ('DTC-5', CrossValidationAgent, dict(learner=DecisionTreeClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=9),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict())),
        ('DTC-25', CrossValidationAgent, dict(learner=DecisionTreeClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=27),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict())),
        ('LR-l1-1', CrossValidationAgent, dict(learner=LogisticRegression,
                                                  learner_kwargs=dict(C=1, penalty='l1'),
                                                  agent=OneShotLearnerAgent,
                                                  agent_kwargs=dict()))
                        ]''',
                    anytime_algos='''[
        ('RF-1', CrossValidationAgent, dict(learner=WarmLearner,
                                                  learner_kwargs=dict(base_model=RandomForestClassifier,
                                                                      base_model_kwargs=dict(min_samples_leaf=1,
                                                                                             n_estimators=1)),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('RF-54', CrossValidationAgent, dict(learner=WarmLearner,
                                                  learner_kwargs=dict(base_model=RandomForestClassifier,
                                                                      base_model_kwargs=dict(min_samples_leaf=54,
                                                                                             n_estimators=1)),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-1', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=1, warm_start=True,
                                                                      n_estimators=1),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-54', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=54, warm_start=True,
                                                                      n_estimators=1),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-1-5', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=1, warm_start=True,
                                                                      n_estimators=1, max_depth=5),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-54-5', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=54, warm_start=True,
                                                                      n_estimators=1, max_depth=5),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('RF-3', CrossValidationAgent, dict(learner=WarmLearner,
                                                  learner_kwargs=dict(base_model=RandomForestClassifier,
                                                                      base_model_kwargs=dict(min_samples_leaf=3,
                                                                                             n_estimators=1)),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('RF-27', CrossValidationAgent, dict(learner=WarmLearner,
                                                  learner_kwargs=dict(base_model=RandomForestClassifier,
                                                                      base_model_kwargs=dict(min_samples_leaf=27,
                                                                                             n_estimators=1)),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-3', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=3, warm_start=True,
                                                                      n_estimators=1),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-27', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=27, warm_start=True,
                                                                      n_estimators=1),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-3-5', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=3, warm_start=True,
                                                                      n_estimators=1, max_depth=5),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-27-5', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=27, warm_start=True,
                                                                      n_estimators=1, max_depth=5),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('RF-9', CrossValidationAgent, dict(learner=WarmLearner,
                                                  learner_kwargs=dict(base_model=RandomForestClassifier,
                                                                      base_model_kwargs=dict(min_samples_leaf=9,
                                                                                             n_estimators=1)),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-9', CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=9, warm_start=True,
                                                                      n_estimators=1),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum))),
        ('GBM-9-5',CrossValidationAgent, dict(learner=GradientBoostingClassifier,
                                                  learner_kwargs=dict(min_samples_leaf=9, warm_start=True,
                                                                      n_estimators=1, max_depth=5),
                                                  agent_kwargs=dict(time_quantum=self.compute_quantum)))
                        ]'''
                    )
    # Iterate through default key-value pairs, setting all unset keys
    for key, value in defaults.iteritems():
        if not key in exp_params:
            exp_params[key] = value
    return exp_params


def exp_params_to_str(exp_params):
    result = "Running experiment:\n"
    for key, value in exp_params.iteritems():
        result += "%s = %s,\n" % (key, value)
    return result


if __name__ == '__main__':
    run_experiment_file(os.path.join('..', 'experiments', 'test_01.py'), plot_override=True)
    # run_experiment_file(os.path.join('..', 'experiments', 'test_shane.py'), plot_override=True)

    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-5-1-01.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-5-1-02.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-5-1-03.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-5-1-04.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-5-1-05.py'), plot_override=False)

    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-5-1-01.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-5-1-02.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-5-1-03.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-5-1-04.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-5-1-05.py'), plot_override=False)

    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-5-1-01.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-5-1-02.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-5-1-03.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-5-1-04.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-5-1-05.py'), plot_override=False)

    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-2-1-01-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-2-1-02-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-2-1-03-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-2-1-04-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-FT-2-1-05-slow.py'), plot_override=False)
    #
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-2-1-01-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-2-1-02-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-2-1-03-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-2-1-04-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-stack-2-1-05-slow.py'), plot_override=False)
    #
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-2-1-01-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-2-1-02-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-2-1-03-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-2-1-04-slow.py'), plot_override=False)
    # run_experiment_file(os.path.join('..', 'experiments', '2015-05-19-madeline-cycle-2-1-05-slow.py'), plot_override=False)
