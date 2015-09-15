from __future__ import division
import datetime

__author__ = 'James Robert Lloyd'
__description__ = 'Miscellaneous utility functions'

import os
import shutil
from glob import glob
import tempfile
import string
import random

import matplotlib.pyplot as pl
import numpy as np
import scipy.io
from sklearn.cross_validation import KFold
import psutil
from constants import SAVE_DIR
import signal

from automl_lib.data_manager import DataManager
from agent import TerminationEx

import time


def move_make(outdir):
    """If outdir exists and isn't empty move its contents to outdir + the date. Then make a fresh outdir."""
    if os.path.isdir(outdir):
        if os.listdir(outdir):  # if it has files, move them
            the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            os.rename(outdir, outdir+'_'+the_date)
            os.makedirs(outdir)
    else:
        os.makedirs(outdir)


def move_make_file(flnm):
    outdir = os.path.dirname(flnm)
    if os.path.exists(flnm):
        the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        os.rename(flnm, flnm+'_'+the_date)
    elif not os.path.isdir(outdir):
        os.makedirs(outdir)

    open(flnm, 'a').close()


# def init_temp_dir():
#     if os.path.isdir(TEMP_DIR):
#         shutil.rmtree(TEMP_DIR)
#     os.mkdir(TEMP_DIR)


def signal_handler(signum, frame):
    """Handlers for signals"""
    if signum == 20:  # SIGTSTP
        signal.pause()  # signal.pause puts *only* the calling thread to sleep, rather than stopping all threads
    if signum == 25:  # SIGCONT
        pass  # as long as there exists a handler for sigcont then signal.pause will return
    if signum == 15:  # SIGTERM
        raise TerminationEx


def mkstemp_safe(directory, suffix):
    """Avoids a file handle leak present on some operating systems"""
    (os_file_handle, file_name) = tempfile.mkstemp(dir=directory, suffix=suffix)
    os.close(os_file_handle)
    return file_name

def callback_1d(model, bounds, info, x, index, ftrue):
    """
    Plot the current posterior, the index, and the value of the current
    recommendation.
    """
    xmin, xmax = bounds[0]
    xx_ = np.linspace(xmin, xmax, 500)                  # define grid
    xx = xx_[:, None]

    # ff = ftrue(xx)                                      # compute true function
    acq = index(xx)                                     # compute acquisition

    mu, s2 = model.posterior(xx)                        # compute posterior and
    lo = mu - 2 * np.sqrt(s2)                           # quantiles
    hi = mu + 2 * np.sqrt(s2)

    # ymin, ymax = ff.min(), ff.max()                     # get plotting ranges
    ymin, ymax = lo.min(), hi.max()                     # get plotting ranges FIXME - remember observed function values
    ymin -= 0.2 * (ymax - ymin)
    ymax += 0.2 * (ymax - ymin)

    kwplot = {'lw': 2, 'alpha': 0.5}                    # common plotting kwargs

    fig = pl.figure(1)
    fig.clf()

    pl.subplot(221)
    # pl.plot(xx, ff, 'k:', **kwplot)                     # plot true function
    pl.plot(xx, mu, 'b-', **kwplot)                     # plot the posterior and
    pl.fill_between(xx_, lo, hi, color='b', alpha=0.1)  # uncertainty bands
    pl.scatter(info['x'], info['y'],                    # plot data
               marker='o', facecolor='none', zorder=3)
    pl.axvline(x, color='r', **kwplot)                  # latest selection
    pl.axvline(info[-1]['xbest'], color='g', **kwplot)  # current recommendation
    pl.axis((xmin, xmax, ymin, ymax))
    pl.ylabel('posterior')

    pl.subplot(223)
    pl.fill_between(xx_, acq.min(), acq,                # plot acquisition
                    color='r', alpha=0.1)
    pl.axis('tight')
    pl.axvline(x, color='r', **kwplot)                  # plot latest selection
    pl.xlabel('input')
    pl.ylabel('acquisition')

    pl.subplot(222)
    pl.plot(ftrue(info['xbest']), 'g', **kwplot)        # plot performance
    pl.axis((0, len(info['xbest']), ymin, ymax))
    pl.xlabel('iterations')
    pl.ylabel('value of recommendation')

    # for ax in fig.axes:                                 # remove tick labels
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def ls(filename):
    return sorted(glob(filename))


def colorbrew(i):
    """Nice colors taken from http://colorbrewer2.org/ by David Duvenaud March 2012"""
    rgbs = [(228,  26,  28),
            (055, 126, 184),
            (077, 175,  74),
            (152,  78, 163),
            (255, 127, 000),
            (255, 255, 051),
            (166,  86, 040),
            (247, 129, 191),
            (153, 153, 153),
            (000, 000, 000)]
    # Convert to [0, 1] range
    rgbs = [(r / 255, g / 255, b / 255) for (r, g, b) in rgbs]
    # Return color corresponding to index - wrapping round
    return rgbs[i % len(rgbs)]


def convert_automl_into_automl_folds(folder, save_folder_root, n_folds=5,
                                     random_state=0, usage='testing'):
    """Convert a dataset in automl format into several folds of automl format"""
    # Load data
    input_dir, basename = os.path.split(folder)
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True)
    X = D.data['X_train']
    y = D.data['Y_train']
    info = D.info
    if not usage is None:
        info['usage'] = usage
    # Now split into folds and save
    folds = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=random_state)
    for (fold, (train_index, test_index)) in enumerate(folds):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_folder = os.path.join(save_folder_root + '_fold_%02d' % (fold + 1), basename)
        mkdir(fold_folder)
        fmt = '%f'
        np.savetxt(os.path.join(fold_folder, basename + '_train.data'), X_train, fmt=fmt, delimiter=' ')
        np.savetxt(os.path.join(fold_folder, basename + '_test.data'), X_test, fmt=fmt, delimiter=' ')
        if info['task'] == 'binary.classification':
            fmt = '%d'
        np.savetxt(os.path.join(fold_folder, basename + '_train.solution'), y_train, fmt=fmt, delimiter=' ')
        np.savetxt(os.path.join(fold_folder, basename + '_test.solution'), y_test, fmt=fmt, delimiter=' ')
        info['train_num'] = X_train.shape[0]
        info['test_num'] = X_test.shape[0]
        with open(os.path.join(fold_folder, basename + '_public.info'), 'w') as info_file:
            for (key, value) in info.iteritems():
                info_file.write('%s = %s\n' % (key, value))
        shutil.copy(os.path.join(folder, basename + '_feat.type'), os.path.join(fold_folder, basename + '_feat.type'))


def convert_automl_into_automl_folds_folder(input_folder, save_folder_root, *args, **kwargs):
    """Converts a folder"""
    folders = sorted(os.listdir(input_folder))
    for folder in folders:
        if os.path.isdir(os.path.join(input_folder, folder)):
            print('Processing ' + folder)
            convert_automl_into_automl_folds(os.path.join(input_folder, folder), save_folder_root,
                                             *args, **kwargs)


def convert_mat_into_automl_folds(filename, save_folder_root, time_budget=300, n_folds=5, input_type='Numerical',
                                  random_state=0, metric='auc_metric', usage='testing', task='binary.classification',
                                  target_type='Binary'):
    """Convert a dataset in .mat format into several folds of automl format"""
    # Load data
    data = scipy.io.loadmat(filename)
    X = data['X']
    y = data['y']
    data_name = os.path.splitext(os.path.split(filename)[-1])[0]
    # Convert data if appropriate
    if task == 'binary.classification':
        y_max = y.max()
        y[y == y_max] = 1
        y[y < y_max] = 0
    # If input_type is 'infer' we now infer input types
    if input_type == 'infer':
        raise Exception('I do not know how to infer input types yet')
    else:
        input_type_list = [input_type] * X.shape[1]
    # Create info dictionary
    # TODO - some of these defaults need to be changed
    info = dict(usage=usage, name=data_name, task=task, target_type=target_type,
                feat_type='Numerical', metric=metric, feat_num=X.shape[1],
                target_num=1, label_num=0, has_categorical=0, has_missing=0, is_sparse=0,
                time_budget=time_budget, valid_num=0)
    # Now split into folds and save
    folds = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=random_state)
    for (fold, (train_index, test_index)) in enumerate(folds):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_folder = os.path.join(save_folder_root + '_fold_%02d' % (fold + 1), data_name)
        mkdir(fold_folder)
        fmt = '%f'
        np.savetxt(os.path.join(fold_folder, data_name + '_train.data'), X_train, fmt=fmt, delimiter=' ')
        np.savetxt(os.path.join(fold_folder, data_name + '_test.data'), X_test, fmt=fmt, delimiter=' ')
        if task == 'binary.classification':
            fmt = '%d'
        np.savetxt(os.path.join(fold_folder, data_name + '_train.solution'), y_train, fmt=fmt, delimiter=' ')
        np.savetxt(os.path.join(fold_folder, data_name + '_test.solution'), y_test, fmt=fmt, delimiter=' ')
        info['train_num'] = X_train.shape[0]
        info['test_num'] = X_test.shape[0]
        with open(os.path.join(fold_folder, data_name + '_public.info'), 'w') as info_file:
            for (key, value) in info.iteritems():
                info_file.write('%s = %s\n' % (key, value))
        with open(os.path.join(fold_folder, data_name + '_feat.type'), 'w') as feature_file:
            for feat_type in input_type_list:
                feature_file.write('%s\n' % feat_type)


def convert_mat_into_automl_folds_folder(mat_folder, save_folder_root, *args, **kwargs):
    """Converts a folder"""
    filenames = sorted(os.listdir(mat_folder))
    for filename in filenames:
        if filename.endswith('.mat'):
            print('Processing ' + filename)
            convert_mat_into_automl_folds(os.path.join(mat_folder, filename), save_folder_root,
                                          *args, **kwargs)


def create_synthetic_classification_problems(mat_folder, save_folder_root, synth_kwargs_list):
    pass


def VmB(pid, VmKey):
    scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
             'KB': 1024.0, 'MB': 1024.0*1024.0}
    try:
        t = open('/proc/%d/status' % pid)
        v = t.read()
        t.close()
    except:
        return -1  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * scale[v[2]]


def memory_usage(pid, since=0.0):
    """Return memory usage in bytes."""
    return VmB(pid, 'VmSize:') - since


def resident_memory_usage(pid, since=0.0):
    """Return resident memory usage in bytes."""
    return VmB(pid, 'VmRSS:') - since


def random_string(N=20):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                   for _ in range(N))


def random_temp_file_name(suffix='.file', dirnm=SAVE_DIR, N=20):
    return os.path.join(dirnm, random_string(N=N) + suffix)


def pack_list(a_list):
    return [element for element in a_list if not element is None]


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def dict_of_lists_to_arrays(a_dict):
    for key, value in a_dict.iteritems():
        a_dict[key] = np.array(value)
    return a_dict


def identity(i):
    return i


def classes_to_1_of_k(classes, k):
    result = np.zeros((classes.shape[0], k))
    for i, a_class in enumerate(classes):
        result[i, a_class] = 1
    return result


def append_1_of_k_encoded_data(D):
    """Takes a data manager object and creates 1 of k encodings for targets for scoring purposes"""
    k = D.info['label_num']
    if 'Y_train' in D.data:
        D.data['Y_train_1_of_k'] = classes_to_1_of_k(classes=D.data['Y_train'], k=k)
    if 'Y_valid' in D.data:
        D.data['Y_valid_1_of_k'] = classes_to_1_of_k(classes=D.data['Y_valid'], k=k)
    if 'Y_test' in D.data:
        D.data['Y_test_1_of_k'] = classes_to_1_of_k(classes=D.data['Y_test'], k=k)


def load_1_of_k_data(folder, basename, D):
    if 'Y_train' in D.data:
        D.data['Y_train_1_of_k'] = np.loadtxt(os.path.join(folder, basename, basename + '_train.solution'),
                                              dtype=np.int8)
    if 'Y_valid' in D.data:
        D.data['Y_valid_1_of_k'] = np.loadtxt(os.path.join(folder, basename, basename + '_valid.solution'),
                                              dtype=np.int8)
    if 'Y_test' in D.data:
        D.data['Y_test_1_of_k'] = np.loadtxt(os.path.join(folder, basename, basename + '_test.solution'),
                                             dtype=np.int8)


def murder_family(pid=None, pgid=None, killall=False, sig=signal.SIGTERM):
    """Send signal to all processes in the process group pgid, apart from pid.
    If killall is true, signal to pid as well.
    SIGTERM - die nicely
    SIGKILL - die right now
    SIGSTOP - uncatchable stop signal - might break queues
    SIGTSTP - makes main thread sleep (with signal.pause()) until sigcont received
    SIGCONT - start anything that is paused or stopped"""
    if pid is None:
        pid = os.getpid()  # pid of current process
    if pgid is None:
        pgid = os.getpgid(pid)  # process group of pid

    for ps in psutil.process_iter():
        if os.getpgid(ps.pid) == pgid and ps.pid != pid:
            try:
                ps.send_signal(sig)
            except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
                pass

    if killall is True:
        try:
            ps = psutil.Process(pid=pid)
            ps.send_signal(sig)
        except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
            pass


def ensure_2d(an_array):
    result = np.array(an_array, ndmin=2)
    if result.shape[0] == 1:
        result = result.T
    return result


def csv_file_size(file_name, delimiter=' '):
    # Peek at first line
    with open(file_name, 'rU') as data_file:
        first_line = data_file.readline()
        elements = first_line.split(delimiter)
        # Strip elements of any empty values
        elements = [el for el in elements if not el == '']
        n_col = len(elements)
    # Count the lines
    n_row = 0
    with open(file_name, 'rU') as data_file:
        for line in data_file:
            n_row += 1
    return n_row, n_col


class NotAnArray(object):
    """Records file location and size of array"""
    def __init__(self, file_name, delimiter=' '):
        # Load file size
        self.n_row, self.n_col = csv_file_size(file_name=file_name, delimiter=delimiter)
        # Remember the filename
        self.file_name = file_name

    @property
    def shape(self):
        return self.n_row, self.n_col


def is_sorted(a_list):
    return all(a_list[i] <= a_list[i+1] for i in xrange(len(a_list)-1))


def waste_cpu_time(a_time):
    start_time = time.clock()
    # foo = 'bar'
    while time.clock() < start_time + a_time:
        # foo = foo[1:] + foo[0]
        pass


if __name__ == '__main__':
    convert_automl_into_automl_folds_folder('../data/phase_1', '../data/phase_1_cv')