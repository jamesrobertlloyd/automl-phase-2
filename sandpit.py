__author__ = 'James Robert Lloyd'
__description__ = 'Scraps of code before module structure becomes apparent'

from util import callback_1d

import pybo
from pybo.functions.functions import _cleanup, GOModel

import numpy as np

from sklearn.datasets import load_iris
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

import os
import sys
sys.path.append(os.path.dirname(__file__))

from multiprocessing import Pool

from sandpit_two import print_globals, import_and_print_globals
import global_data

@_cleanup
class Sinusoidal(GOModel):
    """
    Simple sinusoidal function bounded in [0, 2pi] given by cos(x)+sin(3x).
    """
    bounds = [[0, 2*np.pi]]
    xmax = 3.61439678

    @staticmethod
    def _f(x):
        return -np.ravel(np.cos(x) + np.sin(3*x))

@_cleanup
class CV_RF(GOModel):
    """
    Cross validated random forest
    """
    bounds = [[1, 25]]
    xmax = 10  # FIXME - should this not be optional?

    @staticmethod
    def _f(x):
        # iris = load_iris()
        X, y = X, y = make_hastie_10_2(random_state=0)
        x = np.ravel(x)
        f = np.zeros(x.shape)
        for i in range(f.size):
            clf = RandomForestClassifier(n_estimators=1, min_samples_leaf=int(np.round(x[i])), random_state=0)
            # scores = cross_val_score(clf, iris.data, iris.target)
            scores = cross_val_score(clf, X, y, cv=5)
            f[i] = -scores.mean()
        return f.ravel()


from multiprocessing import Process, Queue, Manager, Array
from Queue import Empty as q_Empty

import cPickle as pickle
import time

from agent import Agent, start_communication #, start_communication_debug


# class DummyAgent(AgentWithData):
#     def __init__(self, name='Give me a name', cpu_budget=1, **kwargs):
#         super(DummyAgent, self).__init__(**kwargs)
#         self.name = name
#         self.cpu_budget = cpu_budget
#         self._value = 1
#
#     def serialize(self, filename):
#         del self.shared_array
#         del self.nparray  # we don't need to delete it for pickle to work, but it will be incorrect on unpickling
#         with open(filename, 'wb') as pickle_file:
#             pickle.dump(self, pickle_file)
#         self.terminated = True
#
#     def next_action(self):
#         while len(self.inbox) > 0:
#             message = self.inbox.pop(0)
#             print(self.nparray)
#             print('Received message : %s' % message)
#             if message['subject'] == 'serialize':
#                 self.serialize(message['filename'])


def separate_process():
    print('I am a separate process')

from learners import DummyLearner


# def multi_pickle_experiment():
#     q = Queue()
#     arr = Array('d', range(5))
#     a = DummyLearner(shared_array=arr, inbox_q=q)
#     p = Process(target=start_communication_debug, kwargs=dict(agent=a))
#     del a
#     p.start()
#     arr[0] = 99
#     q.put(dict(subject='A message'))
#     time.sleep(2)
#     raw_input('Press return to continue')
#     arr[0] = 9
#     q.put(dict(subject='Another message'))
#     time.sleep(2)
#     q.put(dict(subject='serialize', filename='temp/dill.pk'))
#     time.sleep(2)
#     raw_input('Press return to continue')
#     p.join()
#     print('Process has serialized itself')
#     raw_input('Press return to revive')
#     arr[0] = 999
#     p = Process(target=start_communication_debug, kwargs=dict(pickle_filename='temp/dill.pk', shared_array=arr))
#     p.start()
#     q.put(dict(subject='A second message'))
#     time.sleep(2)
#     raw_input('Press return to kill')
#     q.put(dict(subject='terminate'))
#     p.join()
#     print('Success')


# def print_globals(_):
#     time.sleep(5)
#     print(globals())
#     X = my_global
#     X = X + 1
#     print(X)
#     time.sleep(5)


def global_test():
    raw_input('I begin')
    # global my_global
    global_data.my_global = np.full((2**17, 2 * 2**10), 42)
    raw_input('Globals created')
    processing_pool = Pool(10)
    processing_pool.map(import_and_print_globals, [None] * 10)
    processing_pool.close()
    processing_pool.join()
    raw_input('Multiprocessing complete')


if __name__ == '__main__':
    # objective = CV_RF()
    #
    # info = pybo.solve_bayesopt(
    #     objective,
    #     objective.bounds,
    #     niter=25,
    #     noisefree=False,
    #     rng=0,
    #     init='uniform',
    #     callback=callback_1d)
    #
    # print('Finished')
    #
    # raw_input('Press enter to finish')

    global_test()
