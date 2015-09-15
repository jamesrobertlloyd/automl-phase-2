#!/usr/bin/python
from __future__ import division

#############################
# ChaLearn AutoML challenge #
#############################

# Usage: python run.py input_dir output_dir

# This sample code can be used either 
# - to submit RESULTS deposited in the res/ subdirectory or
# - as a template for CODE submission.
#
# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
# 	dataname/dataname_public.info        -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_test_000.predict            -- Provide predictions at regular intervals to make sure you get some results even if the program crashes
# 	dataname_test_001.predict
# 	dataname_test_002.predict
# 	...
# 	dataname_valid_000.predict
# 	dataname_valid_001.predict
# 	dataname_valid_002.predict
# 	...
# 
# Result submission:
# =================
# Search for @RESULT to locate that part of the code.
# ** Always keep this code. **
# If the subdirectory res/ contains result files (predicted values)
# the code just copies them to the output and does not train/test models.
# If no results are found, a model is trained and tested (see code submission).
#
# Code submission:
# ===============
# Search for @CODE to locate that part of the code.
# ** You may keep or modify this template or subtitute your own code. **
# The program saves predictions regularly. This way the program produces
# at least some results if it dies (or is terminated) prematurely. 
# This also allows us to plot learning curves. The last result is used by the
# scoring program.
# We implemented 2 classes:
# 1) DATA LOADING:
#    ------------
# Use/modify 
#                  D = DataManager(basename, input_dir, ...) 
# to load and preprocess data.
#     Missing values --
#       Our default method for replacing missing values is trivial: they are replaced by 0.
#       We also add extra indicator features where missing values occurred. This doubles the number of features.
#     Categorical variables --
#       The location of potential Categorical variable is indicated in D.feat_type.
#       NOTHING special is done about them in this sample code. 
#     Feature selection --
#       We only implemented an ad hoc feature selection filter efficient for the 
#       dorothea dataset to show that performance improves significantly 
#       with that filter. It takes effect only for binary classification problems with sparse
#       matrices as input and unbalanced classes.
# 2) LEARNING MACHINE:
#    ----------------
# Use/modify 
#                 M = MyAutoML(D.info, ...) 
# to create a model.
#     Number of base estimators --
#       Our models are ensembles. Adding more estimators may improve their accuracy.
#       Use M.model.n_estimators = num
#     Training --
#       M.fit(D.data['X_train'], D.data['Y_train'])
#       Fit the parameters and hyper-parameters (all inclusive!)
#       What we implemented hard-codes hyper-parameters, you probably want to
#       optimize them. Also, we made a somewhat arbitrary choice of models in
#       for the various types of data, just to give some baseline results.
#       You probably want to do better model selection and/or add your own models.
#     Testing --
#       Y_valid = M.predict(D.data['X_valid'])
#       Y_test = M.predict(D.data['X_test']) 
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013

import glob
import logging
import subprocess
import sys
import signal
import os
import datetime

# Hack to get around lack of output bug - sends stderr and stdout directly to the relevant files
pid = os.getpid()
fds = subprocess.check_output("ls -l /proc/%d/fd" % pid, shell=True)
lines = fds.splitlines()
stderr_fd, stdout_fd = None, None
for line in lines:
    fields = line.split()
    if 'stderr' in fields[-1]:
        stderr_fd = int(fields[-3])
    if 'stdout' in fields[-1]:
        stdout_fd = int(fields[-3])

sys.stdout.flush()
if stdout_fd and stdout_fd != 1:
    os.dup2(stdout_fd, 1)
sys.stderr.flush()
if stderr_fd and stderr_fd != 2:
    os.dup2(stderr_fd, 2)

print "Check open file descriptors"
print fds
# end hack


# =========================== BEGIN USER OPTIONS ==============================
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True  # outputs messages to stdout and stderr for debug purposes

# ZIP your code
###############
# You can create a code submission archive, ready to submit, with zipme = True.
# This is meant to be used on your LOCAL server.
# zipme = False  # use this flag to enable zipping of your code submission
zipme = True  # use this flag to enable zipping of your code submission
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
submission_filename = 'phase2_submission_' + the_date

# I/O defaults
##############
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = os.path.dirname(__file__)

# default_input_dir = os.path.join(root_dir, '..', 'data', 'dsss_bin_class_fold_01')
default_input_dir = os.path.join(root_dir, '..', 'data', 'phase_0')
# default_input_dir = os.path.join(root_dir, '..', 'data', 'phase_1')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'cv_rf_msl')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'cv_rf_gbm')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'automl')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'automl_rf')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf_gbm')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf_gbm_v2')
default_output_dir = os.path.join(root_dir, 'output')

# =========================== END USER OPTIONS ================================

# Version of the sample code
# Change in 1.1: time is measured by time.time(), not time.clock(): we keep track of wall time
# Change in 1.2: James Robert Lloyd taking control of code
# Change in 1.3: Prediction code run in separate process to keep an eye of memory and time
version = 1.3

# General purpose functions
from sys import argv, path
import time
overall_start = time.time()

# Our directories
# Note: On codalab, there is an extra sub-directory called "program"
running_on_codalab = False
run_dir = os.path.abspath(".")
codalab_run_dir = os.path.join(run_dir, "program")
if os.path.isdir(codalab_run_dir): 
    run_dir = codalab_run_dir
    running_on_codalab = True
    print "Running on Codalab!"
lib_dir = os.path.join(run_dir, "lib")
res_dir = os.path.join(run_dir, "res")

# Our libraries  
path.append(run_dir)
path.append(lib_dir)
from automl_lib import data_io                       # general purpose input/output functions
from automl_lib.data_io import vprint           # print only in verbose mode

from multiprocessing import Process
import psutil

import constants
import managers
import util

import traceback
import experiment


# Define function to be called to start process
def run_automl(input_dir, output_dir, data_name, time_budget, running_on_codalab):
    print('input_dir = "%s"' % input_dir)
    print('output_dir = "%s"' % output_dir)
    print('data_name = "%s"' % data_name)
    print('time_budget = %s' % time_budget)
    try:
        # automl.data_doubling_rf(input_dir, output_dir, data_name, time_budget, 20)
        # automl.cv_growing_rf(input_dir, output_dir, data_name, time_budget)
        # automl.cv_growing_rf_gbm(input_dir, output_dir, data_name, time_budget)

        # automl.competition_example(input_dir, output_dir, data_name, time_budget)
        # automl.competition_example_only_rf(input_dir, output_dir, data_name, time_budget)
        # automl.freeze_thaw_cv_rf(input_dir, output_dir, data_name, time_budget)
        # automl.freeze_thaw_cv_rf_gbm(input_dir, output_dir, data_name, time_budget, compute_quantum=10)
        # automl.automl_phase_0(input_dir, output_dir, data_name, time_budget)

        # mgr = managers.FixedLearnersFreezeThawManager(input_dir=input_dir, output_dir=output_dir,
        #                                               basename=data_name, time_budget=time_budget,
        #                                               compute_quantum=None, plot=not running_on_codalab, min_mem=4,
        #                                               n_folds=5)

        exp = dict()
        exp = experiment.exp_param_defaults(exp)

        mgr = managers.FixedLearnersStackingManager(input_dir=input_dir, output_dir=output_dir,
                                                    basename=data_name, time_budget=time_budget,
                                                    compute_quantum=None, plot=not running_on_codalab,
                                                    n_folds=5,
                                                    overhead_memory=constants.OVERHEAD,
                                                    cgroup_soft_limit=constants.CGROUP_SOFT_LIMIT,
                                                    cgroup_hard_limit=constants.CGROUP_HARD_LIMIT,
                                                    exp=exp)
        mgr.communicate()
    except:
        traceback.print_exc()

# =========================== BEGIN PROGRAM ================================

if __name__ == "__main__":
    # Show library version and directory structure
    if running_on_codalab:
        data_io.show_version()
        data_io.show_dir(run_dir)

    # ### Check whether everything went well (no time exceeded)
    execution_success = True
    
    # ### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = os.path.abspath(argv[2])
    # Move old results and create a new output directory 
    data_io.mvdir(output_dir, output_dir+'_'+the_date) 
    data_io.mkdir(output_dir) 
    
    # ### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)

    # ==================== @RESULT SUBMISSION (KEEP THIS) =====================
    # Always keep this code to enable result submission of pre-calculated results
    # deposited in the res/ subdirectory.
    if len(datanames) > 0:
        vprint(verbose,  "************************************************************************")
        vprint(verbose,  "****** Attempting to copy files (from res/) for RESULT submission ******")
        vprint(verbose,  "************************************************************************")
        OK = data_io.copy_results(datanames, res_dir, output_dir, verbose)  # DO NOT REMOVE!
        if OK: 
            vprint(verbose,  "[+] Success")
            datanames = []  # Do not proceed with learning and testing
        else:
            vprint(verbose, "======== Some missing results on current datasets!")
            vprint(verbose, "======== Proceeding to train/test:\n")
    # =================== End @RESULT SUBMISSION (KEEP THIS) ==================

    if zipme and not running_on_codalab:
        vprint(verbose,  "========= Zipping this directory to prepare for submit ==============")
        ignoredirs = [os.path.abspath(x) for x in glob.glob('./output_*')]

        data_io.zipdir(submission_filename + '.zip', ".",
                       ignoredirs=ignoredirs + [os.path.abspath('./compute_server')])

    # ================ @CODE SUBMISSION (SUBSTITUTE YOUR CODE) =================

    # DEBUG
    print "Check for python processes from other people"
    print subprocess.check_output("ps -eo pid,ppid,pgid,stime,state,user,%mem,command | grep python", shell=True)

    available_mem = psutil.virtual_memory().available  # measured in bytes
    print('Available memory = %fMB' % (available_mem / (1024 * 1024)))

    print "Find server's id"
    print subprocess.check_output("ip addr show", shell=True)

    # Set up logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    class ProcessFilter(logging.Filter):
        def filter(self, record):
            """Returns True for messages that should be printed"""
            if record.processName == "Manager" or record.processName == 'MainProcess':
                return True
            else:
                return False

    form = logging.Formatter("[%(levelname)s/%(processName)s] %(asctime)s %(message)s")

    # Handler for logging to stdout
    # sh = logging.StreamHandler(stream=sys.stdout)
    # sh.setLevel(logging.DEBUG)  # set level here
    # sh.addFilter(ProcessFilter())  # filter to show only logs from manager
    # sh.setFormatter(form)
    # root_logger.addHandler(sh)

    # Handler for logging to stderr
    if constants.LOGFILE:
        util.move_make_file(constants.LOGFILE)
        fh = logging.handlers.RotatingFileHandler(constants.LOGFILE, maxBytes=512*1024*1024)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(form)
        root_logger.addHandler(fh)
    else:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(logging.WARN)  # set level here
        sh.addFilter(ProcessFilter())  # filter to show only logs from manager
        sh.setFormatter(form)
        root_logger.addHandler(sh)

    # Create control group to limit resources
    try:
        subprocess.check_call(['which', 'cgexec'])
    except subprocess.CalledProcessError:
        # Install cgroup-bin
        password = os.environ.get('PW', '')  # Required to create and manage control groups
        installcg = "echo '{}' | sudo -S apt-get -y install cgroup-bin".format(password)
        while True:
            retcode = subprocess.call(installcg, shell=True)
            if retcode != 0:
                root_logger.error("Cgroup-bin installation failed")
                time.sleep(1)
            else:
                root_logger.info("Installed cgroup-bin")
                break

    for basename in datanames:
        print('Processing dataset : ' + basename)
        # Keep track of time
        start_time = time.time()
        # Write a file to record start time
        open(os.path.join(output_dir, basename + '.firstpost'), 'wb').close()
        print('\nStarting\n')
        # Read time budget
        with open(os.path.join(input_dir, basename, basename + '_public.info'), 'r') as info_file:
            for line in info_file:
                if line.startswith('time_budget'):
                    time_budget = int(line.split('=')[-1])
        # Debug code
        # time_budget = 120
        print('Time budget = %ds' % time_budget)
        root_logger.info('Time budget = %ds, dataset %s', time_budget, basename)
        # Start separate process to analyse file
        p = Process(target=run_automl, args=(input_dir,
                                             output_dir,
                                             basename,
                                             time_budget - (time.time() - start_time) - 20,
                                             running_on_codalab))
        p.name = 'Manager'
        p.start()
        # Monitor the process, checking to see if it is complete or if total memory usage too high
        while True:
            time.sleep(0.2)
            if p.is_alive():
                available_mem = psutil.virtual_memory().available  # measured in bytes
                print('Available memory = %fMB' % (available_mem / (1024 * 1024)))
                if available_mem < constants.OVERHEAD / 2:
                    print('Less than %.1f GB memory available - aborting process' %
                          (constants.OVERHEAD / float(2 ** 30)))
                    psutil.Process(pid=p.pid).send_signal(sig=signal.SIGTERM)  # tidy up then die please
                    p.join(timeout=10)  # give it a while to respond to signal
                    if p.is_alive():
                        util.murder_family(p.pid, killall=True, sig=signal.SIGKILL)
                        p.join()
                    print('Process %d terminated' % p.pid)
                    break
                if (time.time() - start_time) > (time_budget - 15):
                    print('Time limit approaching - terminating')
                    psutil.Process(pid=p.pid).send_signal(sig=signal.SIGTERM)
                    p.join(timeout=10)
                    if p.is_alive():
                        util.murder_family(p.pid, killall=True, sig=signal.SIGKILL)
                        p.join()
                    print('Process %d terminated' % p.pid)
                    break
                else:
                    print('Remaining time budget = %s' % (time_budget - (time.time() - start_time)))
                    root_logger.info('Remaining time = %ds, dataset %s',
                                     (time_budget - (time.time() - start_time)), basename)

            else:
                print('Process terminated of its own accord')
                break

        print('\nFinished %s\n' % basename)
        sys.stderr.write('Finished %s\n\n' % basename)

    # DEBUG - want to see if there's any extra pythons running
    print "Check for python processes"
    print subprocess.check_output("ps -eo pid,ppid,pgid,stime,state,user,%mem,command | grep python", shell=True)

    if running_on_codalab:
        if execution_success:
            exit(0)
        else:
            exit(1)
