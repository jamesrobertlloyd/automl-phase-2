from __future__ import division
import sys

import constants
import os
from managers import FixedLearnersStackingManager
import psutil
import util
import logging

__author__ = 'James Robert Lloyd, Emma Smith'
__description__ = 'Entry point for testing'  # TODO - demo should be for demos!


def ft_demo_v1():
    # test_data_name = "dorothea"
    # test_data_name = "coil2000"
    # test_data_name = "magic"
    # test_data_name = "digits"
    # test_data_name = "ring"
    # test_data_name = "banana"
    # test_data_name = "heart"
    # test_data_name = "liver"
    # test_data_name = "christine"
    # test_data_name = "jasmine"
    test_data_name = "madeline"
    # test_data_name = "philippine"
    # test_data_name = "sylvine"
    # test_data_name = 'newsgroups'
    # test_data_name = 'adult'
    test_time_budget = 5 * 60  # seconds TODO - this should also be in an experiment config file

    # FIXME - TEST_INPUT_DIR is not a constant! - this should be in an experiment config file
    constants.TEST_INPUT_DIR = os.path.join('..', 'data', 'phase_1')

    mgr = FixedLearnersStackingManager(constants.TEST_INPUT_DIR, constants.TEST_OUTPUT_DIR, test_data_name,
                                       test_time_budget,
                                       compute_quantum=5, plot=True,
                                       overhead_memory=constants.OVERHEAD,
                                       cgroup_soft_limit=constants.CGROUP_SOFT_LIMIT,
                                       cgroup_hard_limit=constants.CGROUP_HARD_LIMIT)
    mgr.communicate()

if __name__ == '__main__':
    p = psutil.Process()
    current_cpus = p.cpu_affinity()
    p.cpu_affinity(current_cpus[0:-1])

    # Set up logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    manager_pid = os.getpid()

    class ProcessFilter(logging.Filter):
        def filter(self, record):
            """Returns True for messages that should be printed"""
            if record.process == manager_pid:
                return True
            else:
                return False

    form = logging.Formatter("[%(levelname)s/%(processName)s] %(asctime)s %(message)s")

    # Handler for logging to stderr
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)  # set level here
    # sh.addFilter(ProcessFilter())  # filter to show only logs from manager
    sh.setFormatter(form)
    root_logger.addHandler(sh)

    # Handler for logging to file
    util.move_make_file(constants.LOGFILE)
    fh = logging.handlers.RotatingFileHandler(constants.LOGFILE, maxBytes=512*1024*1024)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(form)
    root_logger.addHandler(fh)

    util.move_make(constants.TEST_OUTPUT_DIR)
    ft_demo_v1()
