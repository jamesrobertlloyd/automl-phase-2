from collections import defaultdict
import functools
from multiprocessing import Array
import os
import time
import psutil
import numpy as np
import signal

from agent import Agent, TerminationEx
import util
import constants


class HeavyLearner(Agent):
    """Multiplies matrices, pointlessly."""
    def __init__(self, **kwargs):
        super(HeavyLearner, self).__init__(**kwargs)

    def first_action(self):
        self.send_to_parent(dict(sender=self.name, subject='HeavyLearner'))

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
                #print message
                self.standard_responses(message)
            except (IndexError, AttributeError):
                break

    def next_action(self):
        #self.send_to_parent(dict(sender=self.name, subject='HeavyLearner'))
        m1 = np.random.random_integers(0, 100, (1000, 1000))
        m2 = np.random.random_integers(0, 100, (1000, 1000))
        m3 = np.dot(m1, m2)


class SimpleLearner(Agent):
    """Very simple learner"""
    def __init__(self, **kwargs):
        super(SimpleLearner, self).__init__(**kwargs)
        self.actioncounter = 0

    def first_action(self):
        self.send_to_parent(dict(sender=self.name, subject='SimpleLearner'))

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
                #print message
                self.standard_responses(message)
            except (IndexError, AttributeError):
                break

    def next_action(self):
        self.read_messages()
        #if self.data is not None:
        #    print self.data.items()
        self.send_to_parent(dict(sender=self.name, subject='SimpleLearner', value=self.actioncounter))
        self.actioncounter += 1


class RecursiveLearner(Agent):
    """Makes a child of the same class, unless recursion_level is zero in which case it
    makes a sub learner. Passes all messages from children to parent, and prints a message
    about itself each second"""
    def __init__(self, recursion_level=0, sub_learner=HeavyLearner, **kwargs):
        super(RecursiveLearner, self).__init__(**kwargs)
        self.recursion_level = recursion_level
        self.sub_learner = sub_learner

    def first_action(self):
        if self.recursion_level == 0:
            learners = [self.sub_learner]
        else:
            # learners = [(RecursiveLearner, {'recursion_level': self.recursion_level - 1})]
            learners = [(RecursiveLearner, {'recursion_level': self.recursion_level - 1,
                                            'sub_learner': self.sub_learner}),
                        (RecursiveLearner, {'recursion_level': self.recursion_level - 1,
                                            'sub_learner': self.sub_learner})]
        self.create_children(classes=learners)
        self.start_children()
        self.send_to_parent(dict(sender=self.name, subject='RecursiveLearner'))

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
                #print message
                self.standard_responses(message)
            except (IndexError, AttributeError):
                break

    def next_action(self):
        self.read_messages()
        for name, inbox in self.child_inboxes.iteritems():
            while True:
                try:
                    message = inbox.pop(0)
                    # message['sender'] = self.name + '.' + message['sender']
                    self.send_to_parent(message)
                except IndexError:
                    break


class ImmortalLearner(Agent):
    """Never dies! Also checks its messages really infrequently"""
    # This tests the process killing functions
    def __init__(self, **kwargs):
        super(ImmortalLearner, self).__init__(**kwargs)

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
                #print message
                self.standard_responses(message)
            except (IndexError, AttributeError):
                break

    def next_action(self):
        self.read_messages()
        self.send_to_parent(dict(sender=self.name, subject='I am immortal'))
        time.sleep(60)


class QuickLearner(Agent):
    """Finishes immediately"""
    # Should have finished, and therefore not be pickled
    def __init__(self, **kwargs):
        super(QuickLearner, self).__init__(**kwargs)

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
                #print message
                self.standard_responses(message)
            except (IndexError, AttributeError):
                break

    def next_action(self):
        self.read_messages()
        self.send_to_parent(dict(sender=self.name, subject='I am really fast'))
        raise TerminationEx


class BigLearner(Agent):
    """Stores a large (64MiB) matrix, pointlessly."""
    def __init__(self, **kwargs):
        super(BigLearner, self).__init__(**kwargs)
        self.m1 = None
        self.extras = None
        self.extras2 = None

    def first_action(self):
        self.send_to_parent(dict(sender=self.name, subject='BigLearner'))
        self.m1 = np.random.random_sample((131072, 1024))
        # set this in first_action so that it gets counted in the memory immediately
        # sizeof(double) = 8 bytes
        # 8 * 131072 = 2^20 B = 1MB
        # 8 * 131072 * 64 = 64 MiB
        # self.extras = BigLearnerExtras()
        # self.extras2 = [BigLearnerListicles() for x in range(1024)]

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
                #print message
                self.standard_responses(message)
            except (IndexError, AttributeError):
                break

    def next_action(self):
        self.read_messages()
        print self.__dict__.keys()
        if self.extras is not None:
            print self.extras.__dict__.keys()
        if self.extras2 is not None:
            print "List length:", len(self.extras2)


class BigLearnerExtras(object):
    def __init__(self):
        self.m1 = np.random.random_sample((131072, 1024))
        # self.m2 = np.random.random_sample((131072, 64))
        # self.m3 = np.random.random_sample((131072, 64))


class BigLearnerListicles(object):
    def __init__(self):
        self.m1 = np.random.random_sample((128, 1024))  # 1MB


class SpawningLearner(Agent):
    """Makes one sub learner each second"""
    def __init__(self, maxspawn=100, sub_learner=BigLearner, **kwargs):
        super(SpawningLearner, self).__init__(**kwargs)
        self.maxspawn = maxspawn
        self.spawncount = 0
        self.sub_learner = sub_learner

    def read_messages(self):
        while True:
            try:
                message = self.inbox.pop(0)
                #print message
                self.standard_responses(message)
            except (IndexError, AttributeError):
                break

    def next_action(self):
        self.read_messages()
        if self.spawncount < self.maxspawn:
            names = self.create_children(classes=[self.sub_learner])
            self.start_children(names=names)
            self.spawncount += 1
            self.send_to_parent(dict(sender=self.name, subject='SpawningLearner', n_babies=self.spawncount))

        for name, inbox in self.child_inboxes.iteritems():
            while True:
                try:
                    message = inbox.pop(0)
                    # message['sender'] = self.name + '.' + message['sender']
                    self.send_to_parent(message)
                except IndexError:
                    break


class OrphaningLearner(Agent):
    """Makes babies and then kills itself"""
    def __init__(self, sub_learner=SimpleLearner, **kwargs):
        super(OrphaningLearner, self).__init__(**kwargs)
        self.sub_learner = sub_learner

    def first_action(self):
        self.create_children(classes=[self.sub_learner]*5)
        self.start_children()

    def next_action(self):
        time.sleep(3)
        psutil.Process().terminate()


class SimpleManager(Agent):
    """Makes sub-learners. Prints all messages received. Terminates once time budget exceeded."""
    def __init__(self, time_budget):
        super(SimpleManager, self).__init__()
        self.time_budget = time_budget
        self.actioncounter = 0
        self.data = dict(shared_value=(np.frombuffer(Array('d', [20]).get_obj())))
        self.child_cpu = defaultdict(lambda: 0)
        self.child_private_memory = defaultdict(lambda: 0)
        self.communication_sleep = 0  # next action has a sleep in it anyway
        self.learner_preference = None
        self.save_file = constants.SAVE_DIR + '/managerV1.pk'
        self.name = 'm'

    def first_action(self):
        # self.load_file('/homes/mlghomes/evs25/Development/extras/automl_datasets/christine/christine_train.data',
        #                'christine_train')
        learners = [# (SimpleLearner, {}), (SimpleLearner, {}),
                    # (RecursiveLearner, {'recursion_level': 1, 'sub_learner': OrphaningLearner}),
                    # (RecursiveLearner, {'recursion_level': 1, 'sub_learner': SimpleLearner}),
                    # (ImmortalLearner, {}),
                    # (QuickLearner, {}),
                    # (SpawningLearner, {'sub_learner': HeavyLearner}),
                    # (HeavyLearner, {}),
                    (BigLearner, {}),
                    ]
        self.create_children(classes=learners)
        self.learner_preference = range(len(learners))
        self.start_children()

    def read_messages(self, print_them=True):
        if print_them:
            print "\nSimpleManager messages:"
        for inbox in self.child_inboxes.values():
            while True:
                try:
                    if print_them:
                        print inbox.pop(0)
                    else:
                        inbox.pop(0)
                except (IndexError, AttributeError):
                    break

    def orphanfinder(self):
        me = psutil.Process()
        my_group = os.getpgid(me.pid)
        for ps in psutil.process_iter():
            if os.getpgid(ps.pid) == my_group and ps.parent().name() == 'init':
                print 'Error! Orphan:', ps

    def init_cpu(self, names=None):
        # cpu measure for a psutils.Process needs to be called once to start counting
        # child_children is a list of psutils processes for each named child and its children
        # we could store child_children as an instance variable, but it might change
        if names is None:
            names = self.child_processes.keys()

        child_children = {}
        for name in names:
            try:
                process = psutil.Process(pid=self.child_processes[name].pid)
                child_children[name] = process.children(recursive=True) + [process]
                for child in child_children[name]:
                    child.cpu_percent()
            except psutil.NoSuchProcess:
                pass
        return child_children

    def get_cpu(self, child_children):
        """Print name and total cpu usage for each named child and its offspring"""
        # Children may show up with less than the expected memory usage when first created. This is because linux only
        # copies memory to the child process when it's first used. To improve the accuracy of the memory measure,
        # large arrays etc. should be created in first_action, not in __init__

        for name in child_children:
            self.child_cpu[name] = 0
            for child in child_children[name]:
                try:
                    self.child_cpu[name] += child.cpu_percent()
                except psutil.NoSuchProcess:
                    pass

        print "\nName, CPU percentage, No. of children"
        for name in child_children:
            print name, self.child_classes[name], self.child_cpu[name], len(child_children[name])

    def monitor_memory(self):
        for name in self.child_processes.keys():
            self.child_private_memory[name] = 0
            try:
                process = psutil.Process(pid=self.child_processes[name].pid)
                for child in process.children(recursive=True) + [process]:
                    try:
                        with open('/proc/{}/smaps'.format(child.pid), 'r') as fp:
                            for line in fp.readlines():
                                if line.startswith("Private"):
                                    self.child_private_memory[name] += int(line.split()[1])
                    except IOError:  # file not found because we're not running on Linux
                        pass
            except psutil.NoSuchProcess:
                pass

        print "\nName, memory in MB"
        for name in self.child_processes.keys():
            print name, self.child_classes[name], "%.1f" % (self.child_private_memory[name]/1024.0)

    def limit_memory(self):
        available = psutil.virtual_memory().available
        print "Available memory", "%.1fMB" % (available/1048576.0)
        if self.learner_preference is None or len(self.learner_preference) == 1:
            return
        #print self.learner_preference
        i = 0
        while available < 0.5*1024*1024*1024:
            worst_child = self.learner_preference[i]
            print "Saving", worst_child
            self.save_children(names=[worst_child])
            time.sleep(1)
            available = psutil.virtual_memory().available
            print "Available memory", "%.1fMB" % (available/1048576.0)
            i += 1
            if i > len(self.learner_preference) - 2:
                print "Only one learner left!"
                break

    def tidy_up(self):
        # apparently children need to be resumed before they can be terminated
        self.resume_children()
        super(SimpleManager, self).tidy_up()

    def next_action(self):
        timenow = time.time() - self.start_time
        print "\nSimpleManager: Time {0:.1f}, actioncount {1}".format(timenow, self.actioncounter)
        if timenow > self.time_budget:
            raise TerminationEx
        self.read_messages(print_them=False)
        # self.orphanfinder()

        # initalise cpu measure
        # child_children = self.init_cpu()
        print self.child_states.items()

        if self.actioncounter == 0:
            pass
        # elif self.actioncounter == 4:
        #     self.pause_children(names=[self.child_states.keys()[0]])
        #     self.resume_children(names=[self.child_states.keys()[0]])
        # elif self.actioncounter == 15:
        #     print "Pausing child"
        #     self.pause_children()
        # elif self.actioncounter == 20:
        #     print "Resuming child"
        #     self.resume_children()
        elif self.actioncounter == 10:
            print "Starting save"
            self.start_saving_children()
        # elif self.actioncounter == 6:
        #     print "Finishing save"
        #     self.finish_saving_children()
        #     while len(self.children_told_to_save) > 0:
        #         time.sleep(1)
        #         self.finish_saving_children()
        elif self.actioncounter == 45:
            print "Unpickling child"
            self.create_children()
            self.start_children()
        # if self.actioncounter == 0:
        #     kids = self.child_processes.keys()
        #     self.paused = kids[0]
        #     self.unpaused = kids[1]
        #
        # if self.actioncounter % 5 == 0:
        #     print "Switching"
        #     self.pause_children(names=[self.unpaused])
        #     self.start_children(names=[self.paused])
        #     self.paused, self.unpaused = self.unpaused, self.paused

        self.finish_saving_children()

        time.sleep(1)
        # print self.child_states.items()
        # self.get_cpu(child_children)
        self.monitor_memory()
        #self.limit_memory(child_order=self.child_processes.keys())
        self.actioncounter += 1


if __name__ == '__main__':
    #signal.signal(signal.SIGTSTP, util.signal_handler)
    ps = psutil.Process()
    print ps.pid, os.getpgid(ps.pid)
    os.setpgid(0, 0)  # sets pgid of this process to this process's pid
    m = SimpleManager(60)
    m.communicate()
    print "All done"