from __future__ import division
import pwd
import os

__author__ = 'James Robert Lloyd'
__description__ = 'Constants'

# Universal constants
NUMPY_SAVE = False  # whether to use numpy instead of pickle to save arrays
ORIGINAL = 'original'
DENSE = 'dense'
CONVERT_TO_DENSE = 'convert to dense'

DEBUG = False

ZEROMQ_PORT = 5555


# User-dependent constants
_user = pwd.getpwuid(os.getuid())[0]
import socket
_machine = socket.gethostname()

if _user == 'azureuser':  # running on codalab
    SAVE_DIR = './tmp'
    OVERHEAD = 2 * 2 ** 30
    CGROUP_SOFT_LIMIT = 6 * 2 ** 30
    CGROUP_HARD_LIMIT = 2 * 2 ** 30
    LOGFILE = None

elif _user == 'evs25':
    import matplotlib
    matplotlib.use("tkAgg")  # tkinter isn't installed on codalab

    _temp_dir = '/tmp/automl-2015'  # only used in this file
    # TEST_OUTPUT_DIR = "../predictions/test/stacking"
    TEST_OUTPUT_DIR = _temp_dir + "/predictions"
    TEST_INPUT_DIR = os.path.join('..', 'data', 'phase_1_cv_fold_01')
    # TEST_INPUT_DIR = os.path.join('..', 'data', 'phase_0')
    # TEST_INPUT_DIR = os.path.join('..', 'data', 'demo')
    SAVE_DIR = _temp_dir + '/save'
    MOVIE_TEMP_DIR = _temp_dir + '/movie'
    STACK_DATA_FL = _temp_dir + '/stacking_data.csv'
    LOGFILE = _temp_dir + '/auto-ml.log'

    OVERHEAD = 0.5 * 2 ** 30
    CGROUP_SOFT_LIMIT = 1 * 2 ** 30
    CGROUP_HARD_LIMIT = 0.5 * 2 ** 30

    DEBUG = True

elif _user == 'jrl44' or _machine == 'sagarmatha':
    TEST_OUTPUT_DIR = "../predictions/test/stacking"
    TEST_INPUT_DIR = os.path.join('..', 'data', 'demo')
    SAVE_DIR = 'temp/save'
    MOVIE_TEMP_DIR = '../movie_temp'
    STACK_DATA_FL = '../stacking-data/stacking_data.csv'
    LOGFILE = 'temp/log/auto-ml.log'

    OVERHEAD = 2 * 2 ** 30
    # OVERHEAD = 46 * 2 ** 30
    CGROUP_SOFT_LIMIT = 6 * 2 ** 30
    CGROUP_HARD_LIMIT = 2 * 2 ** 30

    DEBUG = True

else:  # some reasonable defaults
    TEST_OUTPUT_DIR = "./output"
    TEST_INPUT_DIR = os.path.join('..', 'data', 'demo')
    SAVE_DIR = '/tmp/automl-2015'
    MOVIE_TEMP_DIR = '../movie_temp'
    OVERHEAD = 2 * 2 ** 30
    CGROUP_SOFT_LIMIT = 6 * 2 ** 30
    CGROUP_HARD_LIMIT = 2 * 2 ** 30
