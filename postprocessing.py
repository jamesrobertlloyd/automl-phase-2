__author__ = 'James Robert Lloyd'
__description__ = 'Postprocessing of results of experiments'

import os
import sys
root_dir = os.path.dirname(__file__)
sys.path.append(root_dir)

import shutil

import numpy as np
import matplotlib.pyplot as plt

import util


def display_learning_curves(score_folders, picture_folder):
    """"Draw pictures of learning curves"""
    # Allow one folder to be submitted as input instead of list
    if not isinstance(score_folders, list):
        score_folders = [score_folders]
    # Make picture folder if necessary
    util.mkdir(picture_folder)
    # Determine names of datasets
    dataset_names = []
    for score_folder in score_folders:
        new_names = [name for name in os.listdir(score_folder) if os.path.isdir(os.path.join(score_folder, name))]
        dataset_names = sorted(list(set(dataset_names + new_names)))
    print('Datasets:')
    for dataset_name in dataset_names:
        print(' - %s' % dataset_name)
    print('')
    # Determine algorithm names
    algo_names = []
    for score_folder in score_folders:
        for dataset_name in dataset_names:
            if os.path.isdir(os.path.join(score_folder, dataset_name)):
                new_names = [name for name in os.listdir(os.path.join(score_folder, dataset_name))
                             if os.path.isdir(os.path.join(score_folder, dataset_name, name))]
                algo_names = sorted(list(set(algo_names + new_names)))
    print('Algorithms:')
    for algo_name in algo_names:
        print(' - %s' % algo_name)
    print('')
    # For each dataset plot learning curve of every algorithm across score folders
    for dataset_name in dataset_names:
        print(dataset_name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(dataset_name)
        ax.set_xlabel('Time (seconds)')
        # ax.set_xscale('log')
        ax.set_ylabel('Score')
        n_algos = 0
        for (algo_index, algo_name) in enumerate(algo_names):
            print(algo_name)
            for score_folder in score_folders:
                time_score_file = os.path.join(score_folder, dataset_name, algo_name, 'learning_curve.csv')
                if os.path.isfile(time_score_file):
                    time_scores = np.loadtxt(time_score_file, delimiter=',', skiprows=1)
                    if time_scores.size > 0:
                        ax.plot(time_scores[:, 0], time_scores[:, 1],
                                color=util.colorbrew(algo_index),
                                linestyle='dashed', marker='o',
                                label=algo_name)
                        n_algos += 1
        if n_algos > 0:
            leg = ax.legend(loc='best')
            leg.get_frame().set_alpha(0.5)
            fig.savefig(os.path.join(picture_folder, '%s.pdf' % dataset_name))
            plt.show()


def delete_algorithm_scores(algo_name, score_folder):
    for data_folder in [name for name in os.listdir(score_folder) if os.path.isdir(os.path.join(score_folder, name))]:
        target_folder = os.path.join(score_folder, data_folder, algo_name)
        if os.path.isdir(target_folder):
            print('Removing ' + target_folder)
            shutil.rmtree(target_folder)


if __name__ == "__main__":
    display_learning_curves(os.path.join(root_dir, '..', 'scores', '2015-05-19-5-1', 'fold_03'),
                            os.path.join(root_dir, '..', 'analyses', '2015-05-19-5-1', 'fold_03'))