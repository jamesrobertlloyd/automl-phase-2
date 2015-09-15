__author__ = 'jrl44'

# import sandpit
import time
import global_data


def print_globals(_):
    time.sleep(5)
    print(globals())
    # print(sandpit.__dict__)
    # X = my_global
    # X = X + 1
    # print(X)
    time.sleep(5)


def import_and_print_globals(_):
    time.sleep(5)
    print global_data.__dict__
    time.sleep(5)
    X = global_data.my_global
    X = X + 1
    print(X)
    time.sleep(5)