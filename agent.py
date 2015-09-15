"""
Generic autonomous agents classes for automatic statistician

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""
import copy

from multiprocessing import Process, Array, Value, get_logger, Queue
from Queue import Empty
import subprocess

import constants
import time
import cPickle as pickle
import numpy as np
import os
from collections import defaultdict
import psutil
import signal
import sys
import util
import tempfile
import logging
import sys

import global_data

# set up logging for the multiprocessing library
mlogger = get_logger()
mlogger.propagate = True
mlogger.setLevel(logging.DEBUG)

# set up logging for agent module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if constants.NUMPY_SAVE:  # how to pickle numpy arrays
    import copy_reg

    def np_pickler(array):
        """Function for pickle to use when pickling numpy arrays"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_pickle.npy', dir=constants.SAVE_DIR, delete=False) as fp:
            np.save(fp, array)
            tempfl = fp.name
        return np_unpickler, (tempfl,)

    def np_unpickler(np_file):
        """Function for pickle to use when unpickling numpy arrays"""
        array = np.load(np_file)
        os.remove(np_file)
        return array

    # Register np array handlers for pickle
    copy_reg.pickle(np.ndarray, np_pickler, np_unpickler)


def start_communication(agent=None, state_filename=None, cgroup=None, password=''):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # if termination signal received, die
    try:
        if cgroup is not None:
            cgclassify = "echo '{}' | sudo -S cgclassify -g memory:{} --sticky {}".format(
                password, cgroup, os.getpid())
            subprocess.call(cgclassify, shell=True)
        if state_filename is not None:
            agent.init_after_serialization(state_filename=state_filename)
        agent.communicate()
    except SystemExit as e:
        os._exit(e.code)
    except:
        sys.excepthook(*sys.exc_info())  # use logging for exceptions


class DummyQueue(object):  # has send and receive methods that do nothing
    def put(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        raise Empty

    def get_nowait(self, *args, **kwargs):
        raise Empty


class DummyProcess(object):
    def __init__(self, pid=65536, exitcode=-15):
        self.pid = pid
        self.exitcode = exitcode

    def is_alive(self):
        return False

    def start(self):
        raise AssertionError

    def join(self):
        pass


class TerminationEx(Exception):
    pass


class SaveEx(Exception):
    pass


class Agent(object):
    def __init__(self,
                 inbox_conn=None, outbox_conn=None,
                 communication_sleep=1, child_timeout=10, name='', exp=None):
        """
        Implements a basic communication and action loop
         - Get incoming messages from parent
         - Perform next action
         - Send outgoing messages to parent
         - Check to see if terminated
        """
        self.saving_children = []
        self.start_time = time.time()

        if inbox_conn is None:
            inbox_conn = DummyQueue()
        if outbox_conn is None:
            inbox_conn = DummyQueue()
        self.inbox_conn = inbox_conn
        self.outbox_conn = outbox_conn
        self.inbox = []

        self.child_processes = dict()  # contains all created processes
        self.conns_to_children = dict()  # dead processes removed periodically by child_tidying
        self.conns_from_children = dict()  # ditto
        self.child_inboxes = defaultdict(list)
        self.child_serialization_filenames = dict()  # contains all created processes
        self.child_classes = dict()  # has entry for every created process
        self.child_kwargs = defaultdict(dict)
        self.child_states = dict()  # this is primarily for debugging purposes
        # if you need to know the state of a process try to check directly
        # 'terminated', 'finished', 'saved', 'saved unterminated',
        # 'unknown', 'sleeping', 'stopped', 'running', 'unstarted'
        # 'unknown' - dead with unrecognised exit code
        # 'saved' means has also terminated. 'saved unterminated' - manager received save, but process not terminated
        self.children_told_to_save = dict()  # contains save start times
        self.child_flags = dict()  # for communication about save
        self.last_child_started = None
        self.last_child_start_time = None

        self.flag = Value('i', 0)  # value shared between child and parent to communicate about save
        self.communication_sleep = communication_sleep
        self.child_timeout = child_timeout
        self.name = name
        self.exp = exp
        if hasattr(global_data, 'data'):
            self.data = global_data.data
        else:
            self.data = None
        self._namegen_count = -1  # used to save value of self.namegen - do not use this directly!
        self.save_file = None
        self.fa_completed = False  # whether first action has been completed
        self.state = ''
        self.cgroup = None
        self.password = ''
        self.startable_children = set()

        self.attributes_not_to_save = ['inbox_conn', 'outbox_conn',  # 'child_processes',
                                       'conns_to_children', 'conns_from_children', 'data',
                                       'flag', 'child_flags']

    def namegen(self):
        self._namegen_count += 1
        return self.name + '.' + str(self._namegen_count)

    def load_file(self, datafile, array_name):
        data = np.loadtxt(datafile, dtype=float, ndmin=2)
        self.load_array(data, array_name)

    def load_array(self, data, array_name):
        # Array 'double' is the same as python 'float' (the default for numpy arrays), illogically
        self.data[array_name] = np.frombuffer(Array('d', data.ravel()).get_obj())
        self.data[array_name].shape = data.shape  # will raise an error if array cannot be reshaped without copying
        logger.info("%s: Loaded array %s", self.name, array_name)

    def init_after_serialization(self, state_filename):
        """Load state from file and create children"""
        with open(state_filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)  # data is an instance of the same class as self
        self.__dict__.update(data.__dict__)

        # Delete state file
        os.remove(state_filename)

        # Create child processes, pass on shared memory
        self.start_children()
        logger.info("%s: Initialised after serialization", self.name)

    @property
    def children(self):
        return self.child_processes.keys()

    def create_children(self, names=None, new_names=None, classes=None, start=False, use_cgroup=True):
        """If classes, create children from list of classes.
         Classes can be a list, or a list of (class, kwargs) tuples
         If names, revive named pickles or rerun child if pickle doesn't exist (i.e. save failed).
         If neither, revive all pickled children."""
        names_and_classes = dict()
        if classes is None and names is None:
            names = self.child_serialization_filenames.keys()

        cgroup = None
        if use_cgroup:
            cgroup = self.cgroup

        if classes is not None:
            if type(classes[0]) is not tuple:  # if there's just classes (no kwargs) then convert
                classes = [(x, {}) for x in classes]
            if new_names is None:
                new_names = [self.namegen() for _ in range(len(classes))]
            for name, a_class in zip(new_names, classes):
                names_and_classes[name] = a_class

        if names is not None:
            for name in names:
                exit_code = self.child_processes[name].exitcode
                if exit_code is None:
                    if self.child_processes[name].is_alive():
                        logger.error("%s: Child %s has not terminated",
                                     self.name, name)
                        continue
                    else:
                        logger.error("%s: Child %s has been created but not started",
                                     self.name, name)
                if exit_code < 0:
                    logger.warn("%s: Child %s exited with code %d and I won't restart it",
                                self.name, name, exit_code)
                    continue
                elif exit_code == 1:
                    logger.info("%s: Child %s has a saved state", self.name, name)
                    names_and_classes[name] = (self.child_classes[name], self.child_kwargs[name])
                elif exit_code == 0:
                    logger.warn("%s: Child %s terminated of its own accord and I won't restart it",
                                self.name, name)
                    continue

        # Loop over names and classes, creating children
        for name, (cl, kwargs) in names_and_classes.iteritems():
            child = cl(**kwargs)
            child.name = name
            self.child_classes[name] = cl
            self.child_kwargs[name] = kwargs
            # Create conns
            child.inbox_conn = self.conns_to_children[child.name] = Queue()
            self.conns_from_children[child.name] = child.outbox_conn = Queue()

            # Set save file
            if name in self.child_processes and self.child_processes[name].exitcode == 1:
                # child has been started before
                pickle_file = child.save_file = self.child_serialization_filenames[name]
            else:
                root, ext = os.path.splitext(self.save_file)
                child.save_file = self.child_serialization_filenames[name] = root + '_' + name + ext
                pickle_file = None

            # share communication value
            child.flag = self.child_flags[name] = Value('i', 0)

            # Create process
            p = Process(target=start_communication,
                        kwargs=dict(agent=child, state_filename=pickle_file, cgroup=cgroup,
                                    password=self.password))
            p.name = name
            logger.info("%s: Created child %s", self.name, name)
            self.child_processes[child.name] = p
            self.child_states[name] = 'unstarted'
            del child
            if start:
                p.start()

        self.startable_children.update(names_and_classes.keys())
        return names_and_classes.keys()  # returns list of created child names

    def start_children(self, names=None):
        # start or resume or recreate the child processes.
        if names is None:
            names = self.child_states.keys()  # all the children ever

        logger.debug("Starting %s", str(names))

        successes = []
        deadkids = []
        started = []
        for name in names:
            assert name in self.child_states  # check if it's a real child
            dead = False
            proc = self.child_processes[name]
            if self.child_processes[name].pid is None:  # has it been started?
                self.child_processes[name].start()
                logger.info("%s: Started %s", self.name, name)
                successes.append(name)
                started.append(name)
            elif proc.is_alive():  # is it alive?
                try:
                    # NB Can't test for aliveness with NoSuchProcess error, as pid might be reused
                    process = psutil.Process(pid=self.child_processes[name].pid)
                    with self.child_flags[name].get_lock():
                        if self.child_flags[name].value == 1:  # FIXME - need informative flag values
                            self.child_flags[name].value = 0
                            logger.info("Child %s had been told to save. I have cancelled this instruction.", name)
                            successes.append(name)
                            continue
                        elif self.child_flags[name].value == 2:
                            logger.warn("Child %s is in the middle of saving and I won't start it", name)
                            continue
                    # We could only send resume to stopped children, but might introduce race conditions
                    try:
                        for child in process.children(recursive=True):
                            try:
                                child.resume()
                            except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied, IOError) as e:
                        logger.warn("Error %s getting children for resume for child %s", e.strerror, name)
                    process.resume()
                    logger.info("Resumed %s", name)
                    successes.append(name)
                except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):  # not alive
                    dead = True
            else:
                dead = True

            if dead:
                deadkids.append(name)

        if deadkids:
            newkids = self.create_children(names=deadkids, start=True)
            successes.append(newkids)

        if len(started) > 0:
            self.last_child_started = started[-1]
            self.last_child_start_time = time.time()

        return successes

    def pause_children(self, names=None):
        # send pause signal to alive children
        if names is None:
            names = self.conns_from_children.keys()

        logger.debug("Pausing %s", str(names))

        for name in names:
            assert name in self.child_states  # check it's a real child
            proc = self.child_processes[name]
            if proc.is_alive():
                try:
                    process = psutil.Process(pid=proc.pid)
                    try:
                        for child in process.children(recursive=True):
                            try:
                                child.send_signal(signal.SIGTSTP)
                            except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied, IOError) as e:
                        logger.warn("Error %s getting children for pause for child %s", e.strerror, name)
                    process.send_signal(signal.SIGTSTP)
                    logger.info("Paused %s", name)
                except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):  # child may have terminated
                    pass

    def resume_children(self, names=None):
        """Resumes paused children. Shouldn't do anything to children that are not paused"""
        if names is None:
            names = self.conns_from_children.keys()

        logger.debug("Resuming %s", str(names))

        for name in names:
            assert name in self.child_states  # check this is a child's name
            proc = self.child_processes[name]
            if proc.is_alive():
                try:
                    process = psutil.Process(pid=proc.pid)
                    try:
                        for child in process.children(recursive=True):
                            try:
                                child.resume()
                            except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied, IOError) as e:
                        logger.warn("Error %s getting children for resume for child %s", e.strerror, name)
                    process.resume()
                    logger.info("Resumed %s", name)
                except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
                    pass

    def terminate_children(self, names=None, kill_unstarted=False):
        # Terminate named children if they are alive or unstarted
        if names is None:
            names = self.conns_from_children.keys()  # names not in this are definitely dead

        logger.debug("Terminating %s", str(names))

        for name in names:
            assert name in self.child_states  # check this is a child's name
            if self.child_processes[name].pid is None:
                if kill_unstarted:
                    self.child_processes[name] = DummyProcess(exitcode=-15)
                    self.conns_to_children.pop(name).close()
                    self.conns_from_children.pop(name).close()
                    self.child_states[name] = 'terminated'
                    logger.info("Terminated unstarted child %s", name)
                else:
                    logger.info("Child %s is unstarted - pid %s", name, str(self.child_processes[name].pid))
            else:
                if self.child_processes[name].exitcode is not None:
                    logger.info("Child %s already dead - exitcode %s", name, str(self.child_processes[name].exitcode))
                    continue
                try:
                    process = psutil.Process(pid=self.child_processes[name].pid)
                    try:
                        for proc in process.children(recursive=True):
                            try:
                                proc.resume()
                                proc.terminate()
                            except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied, IOError) as e:
                        logger.warn("Error %s getting children for terminate for child %s", e.strerror, name)
                    process.resume()
                    process.terminate()
                    logger.info("Terminated %s", name)
                except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):
                    logger.info("Child %s already dead", name)
                self.child_processes[name].join()  # make sure it has time to respond to terminate signal

    def signal_ignore(self, signum, frame):
        logger.warn("Ignored signal %d", signum)

    def send_to_children(self, message, names=None):
        """Send a message to all children"""
        message = copy.deepcopy(message)  # in case message changes before it's put on the pipe

        if names is None:
            names = self.conns_to_children.keys()

        logger.debug('Sending to %s', str(names))

        for name in names:
            # check that process is not unstarted or terminated. Shouldn't send to dead processes
            if self.child_processes[name].is_alive():
                self.conns_to_children[name].put(message)
                logger.debug("Sent to %s, message subject '%s'", name, message['subject'])

    def send_to_parent(self, message):
        message = copy.deepcopy(message)
        self.outbox_conn.put(message)
        logger.debug("Sent to parent, message subject '%s'", message['subject'])

    def standard_responses(self, message):
        # Some standard actions for messages from parent.
        if message['subject'] == 'pause':  # check for special subjects
            self.pause()
        elif message['subject'] == 'save and terminate':
            self.save_file = message['filename']
            raise SaveEx
        elif message['subject'] == 'terminate':
            raise TerminationEx

    def get_parent_inbox(self):
        """Transfer items from inbox queue into local inbox"""
        while True:
            try:
                message = self.inbox_conn.get_nowait()
                self.inbox.append(message)
                logger.debug("Message from parent, subject '%s'", message['subject'])
            except Empty:
                break

    def child_tidying(self):
        # Fill in child state
        # Join dead process to prevent zombies
        # Read last messages from those that have died a good death
        # Close connections with children and remove from child processes
        dead_children = set()
        logger.debug("Tidying children")
        names = self.conns_to_children.keys()

        unstartable_children = []
        self.saving_children = []
        for name in names:
            get_last_messages = False
            dead = False
            if self.child_processes[name].pid is None:  # hasn't been started
                self.child_states[name] = 'unstarted'
            elif self.child_processes[name].is_alive():
                try:  # we don't really need to record this, but it's useful for debugging
                    p = psutil.Process(pid=self.child_processes[name].pid)  # more accurate check for life
                    self.child_states[name] = p.status()  # running, sleeping, stopped, disk sleep, zombie
                except (psutil.NoSuchProcess, psutil.AccessDenied, IOError):  # maybe it just died
                    dead = True
                if self.child_flags[name].value == 3:
                    self.child_states[name] = "saved unterminated"
                    logger.warn("%s still hasn't terminated", name)
                    unstartable_children.append(name)
                    self.saving_children.append(name)
                elif self.child_flags[name].value == 2:
                    self.child_states[name] = "saving now"
                    self.saving_children.append(name)
                    unstartable_children.append(name)
                elif self.child_flags[name].value == 1:
                    self.child_states[name] = "told to save"
            else:
                dead = True

            if dead:
                dead_children.add(name)
                if self.child_processes[name].exitcode == -15:
                    self.child_states[name] = 'terminated'
                    unstartable_children.append(name)
                elif self.child_processes[name].exitcode == -9:
                    self.child_states[name] = 'killed'
                    unstartable_children.append(name)
                elif self.child_processes[name].exitcode == 0:
                    self.child_states[name] = 'finished'  # i.e finished of its own accord
                    get_last_messages = True
                    unstartable_children.append(name)
                elif self.child_processes[name].exitcode == 1:
                    self.child_states[name] = 'saved'
                    self.startable_children.add(name)
                    get_last_messages = True
                else:
                    self.child_states[name] = 'unknown %s' % str(self.child_processes[name].exitcode)
                    logger.error("Child %s died with exitcode %s", name, str(self.child_processes[name].exitcode))
                    unstartable_children.append(name)

                if get_last_messages:
                    # get final messages
                    inbox_conn = self.conns_from_children[name]
                    while True:
                        try:
                            message = inbox_conn.get_nowait()
                            self.child_inboxes[name].append(message)
                            logger.debug("Message from %s, subject '%s'", name, message['subject'])
                        except Empty:
                            break

        self.startable_children.difference_update(unstartable_children)  # remove unstartable children

        if len(self.child_states) > 0:
            logger.info("Child states %s", str(self.child_states))
            logger.info("Startable children: %s", str(self.startable_children))
            logger.info("Children currently saving: %s", str(self.saving_children))

        if len(dead_children) > 0:
            logger.debug("Dead children %s", str(dead_children))

        for name in dead_children:
            self.child_processes[name].join()
            self.conns_to_children.pop(name).close()
            self.conns_from_children.pop(name).close()
            self.children_told_to_save.pop(name, None)

    def get_child_inboxes(self, names=None):
        # Get messages from children
        if names is None:
            names = self.conns_from_children.keys()

        for name in names:
            assert name in self.child_states  # otherwise this child doesn't exist
            if self.child_processes[name].is_alive():
                inbox_conn = self.conns_from_children[name]
                while True:
                    try:
                        message = inbox_conn.get_nowait()
                        self.child_inboxes[name].append(message)
                        logger.debug("Message from %s, subject '%s'", name, message['subject'])
                    except Empty:
                        break

    def first_action(self):
        """Actions to be performed when first created"""
        pass

    def next_action(self):
        """Inspect messages and state and perform next action checking if process stopped or paused"""
        pass

    def tidy_up(self):
        """Run anything pertinent before termination"""
        self.terminate_children()

    def __getstate__(self):
        odict = self.__dict__.copy()  # copy the dict since we change it
        for key in self.attributes_not_to_save:
            del odict[key]              # remove filehandle entry
        for name, proc in self.child_processes.items():  # TODO: fix hack
            self.child_processes[name] = DummyProcess(pid=proc.pid, exitcode=proc.exitcode)
        return odict

    def save(self):
        """Save important things to file"""
        self.flag.value = 2

        logger.info("Saving")
        # Ignore sigtstp messages from now on:
        signal.signal(signal.SIGTSTP, self.signal_ignore)

        # Saving only happens on the first CPU
        p = psutil.Process()
        current_cpus = p.cpu_affinity()
        if len(current_cpus) > 1:
            p.cpu_affinity([current_cpus[0]])

        # Save children
        self.save_children(save_timeout=300)

        # Save myself
        with open(self.save_file, 'wb') as pickle_file:
            pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)

        logger.info("Completed saving")
        self.flag.value = 3

    def save_children(self, names=None, save_timeout=300):
        """Save important things to file"""
        if names is None:
            names = self.conns_to_children.keys()

        logger.info("Saving children %s", str(names))

        self.start_saving_children(names)

        while any(i in self.children_told_to_save.keys() for i in names):
            time.sleep(1)
            self.finish_saving_children(save_timeout=save_timeout)
        # children are guaranteed to be dead after this
        logger.info("Saved children %s", str(names))

    def start_saving_children(self, names=None):
        """Sends save message to children"""
        if names is None:
            names = self.conns_to_children.keys()

        logger.info("Starting saving of %s", str(names))

        self.resume_children(names=names)
        for name in names:
            if self.child_processes[name].is_alive() is False:
                continue
            if self.child_flags[name].value == 0:
                self.child_flags[name].value = 1
                logger.info("Told %s to save", name)
                self.children_told_to_save[name] = time.time()

    def finish_saving_children(self, save_timeout=300):
        """Checks if children are still alive and terminates them if so"""
        for name, child_save_time in self.children_told_to_save.items():
            if self.child_processes[name].is_alive() is False:
                self.children_told_to_save.pop(name)
            elif time.time() - child_save_time > save_timeout:
                logger.warn("Terminating %s - save took too long", name)
                self.terminate_children(names=[name])
                self.children_told_to_save.pop(name)

    def pause(self):
        self.pause_children()
        logger.info("Pausing myself")
        signal.pause()

    def perform_communication_cycle(self):
        time0 = time.time()
        self.child_tidying()
        self.get_child_inboxes()
        self.get_parent_inbox()
        self.next_action()
        time.sleep(self.communication_sleep)
        logger.debug("Communication cycle took %.1f seconds", time.time() - time0)
        if self.flag.value == 1:
            raise SaveEx

    def communicate(self):
        """Receive incoming messages, perform actions as appropriate and send outgoing messages"""
        try:
            if self.fa_completed is False:
                self.first_action()
                self.fa_completed = True
                logger.debug("Completed first action")
            while True:
                self.perform_communication_cycle()
        except SaveEx:
            self.save()
            sys.exit(1)  # exit with exit code 1
        except TerminationEx:
            logger.debug("Caught TerminationEx")
            self.tidy_up()
