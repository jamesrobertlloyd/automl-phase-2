"""
Generic autonomous agents classes for automatic statistician

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

import numpy as np
import cPickle as pickle
import os
import time
import scipy.sparse

from automl_lib import data_converter
from automl_lib import data_io
from automl_lib.data_io import vprint

import util
from util import NotAnArray

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MemoryAwareDataManager:
    """Memory aware version of the codalab data manager with some added functionality"""

    def __init__(self, basename, input_dir, replace_missing=True, filter_features=False,
                 only_info=False, max_dense_array_size=1*2**30, verbose=False, absolute_max_features=5000):
        self.use_pickle = False  # TODO - remove me
        self.basename = basename
        # if basename in input_dir:
        #     self.input_dir = input_dir
        # else:
        self.input_dir = input_dir + "/" + basename + "/"
        info_file = os.path.join(self.input_dir, basename + '_public.info')
        self.info = {}
        self.getInfo(info_file)
        # Check to see if we should do anything other than gather info
        if not only_info:
            self.feat_type = self.loadType(os.path.join(self.input_dir, basename + '_feat.type'))
            self.data = {}
            Xtr = self.loadData(os.path.join(self.input_dir, basename + '_train.data'),
                                replace_missing=replace_missing)
            Ytr = self.loadLabel(os.path.join(self.input_dir, basename + '_train.solution'))
            Xva = self.loadData(os.path.join(self.input_dir, basename + '_valid.data'),
                                replace_missing=replace_missing)
            Xte = self.loadData(os.path.join(self.input_dir, basename + '_test.data'),
                                replace_missing=replace_missing)
            Yte = self.loadData(os.path.join(self.input_dir, basename + '_test.solution'))
            # Normally, feature selection should be done as part of a pipeline.
            # However, here we do it as a preprocessing for efficiency reason
            # TODO - What if data has not been loaded?
            idx = []
            if filter_features:  # add hoc feature selection, for the example...
                fn = min(Xtr.shape[1], 500)
                idx = data_converter.tp_filter(Xtr, Ytr, feat_num=fn)
                Xtr = Xtr[:, idx]
                if not Xva is None:
                    Xva = Xva[:, idx]
                if not Xte is None:
                    Xte = Xte[:, idx]
            if Xtr.shape[1] > absolute_max_features:
                Xtr = Xtr[:, :absolute_max_features]
                if not Xva is None:
                    Xva = Xva[:, :absolute_max_features]
                if not Xte is None:
                    Xte = Xte[:, :absolute_max_features]
            self.feat_idx = np.array(idx).ravel()
            self.data['X_train'] = Xtr
            self.data['Y_train'] = Ytr

            if not Xva is None:
                self.data['X_valid'] = Xva
            if not Xte is None:
                self.data['X_test'] = Xte
            if not Yte is None:
                self.data['Y_test'] = Yte

            # Create dense arrays if it will fit in memory
            # TODO - Remember that sparse binary can be stored with single byte precision

            total_dense_array_size = 0
            for data_name in ['X_train', 'X_valid', 'X_test']:
                if data_name in self.data and scipy.sparse.issparse(self.data[data_name]):
                    size = self.data[data_name].shape[0] * self.data[data_name].shape[1] * 8
                    total_dense_array_size += size

            if total_dense_array_size <= max_dense_array_size:
                for data_name in ['X_train', 'X_valid', 'X_test']:
                    if data_name in self.data and scipy.sparse.issparse(self.data[data_name]):
                        logger.info('Creating dense array of size %d * %d * %d = %d bytes',
                                    self.data[data_name].shape[0],
                                    self.data[data_name].shape[1],
                                    8,
                                    self.data[data_name].shape[0] * self.data[data_name].shape[1] * 8)
                        self.data[data_name + '_dense'] = self.data[data_name].toarray()
            else:
                logger.info('Not creating dense arrays - predicted to have size %d bytes' % total_dense_array_size)

            # Create 1 of k encoding using byte precision

            if self.info['task'] == 'multiclass.classification':
                util.load_1_of_k_data(input_dir, basename, self)

    def __repr__(self):
        return "DataManager : " + self.basename

    def __str__(self):
        val = "DataManager : " + self.basename + "\ninfo:\n"
        for item in self.info:
            val = val + "\t" + item + " = " + str(self.info[item]) + "\n"
        val = val + "data:\n"
        val = val + "\tX_train = array" + str(self.data['X_train'].shape) + "\n"
        val = val + "\tY_train = array" + str(self.data['Y_train'].shape) + "\n"
        if 'X_valid' in self.data:
            val = val + "\tX_valid = array" + str(self.data['X_valid'].shape) + "\n"
        if 'X_test' in self.data:
            val = val + "\tX_test = array" + str(self.data['X_test'].shape) + "\n"
        val = val + "feat_type:\tarray" + str(self.feat_type.shape) + "\n"
        val = val + "feat_idx:\tarray" + str(self.feat_idx.shape) + "\n"
        return val

    def loadData(self, filename, verbose=True, replace_missing=True):
        """
        Get the data from a text file in one of 3 formats: matrix, sparse, binary_sparse
        Potentially does not load the data if it is too large
        """
        logger.info("Reading %s", filename)
        start = time.time()

        if not os.path.exists(filename):
            return None
        if 'format' not in self.info.keys():
            self.getFormatData(filename)
        if 'feat_num' not in self.info.keys():
            self.getNbrFeatures(filename)

        data_func = {'dense': data_io.data, 'sparse': data_io.data_sparse, 'sparse_binary': data_io.data_binary_sparse}

        data = data_func[self.info['format']](filename, self.info['feat_num'])

        # INPORTANT: when we replace missing values we double the number of variables

        if self.info['format'] == 'dense' and replace_missing and np.any(map(np.isnan, data)):
            vprint(verbose, "Replace missing values by 0 (slow, sorry)")
            data = data_converter.replace_missing(data)

        end = time.time()
        if verbose:  print( "[+] Success in %5.2f sec" % (end - start))
        return data


    def loadLabel(self, filename, verbose=True):
        ''' Get the solution/truth values'''
        if verbose:  print("========= Reading " + filename)
        start = time.time()
        if self.use_pickle and os.path.exists(os.path.join(self.tmp_dir, os.path.basename(filename) + ".pickle")):
            with open(os.path.join(self.tmp_dir, os.path.basename(filename) + ".pickle"), "r") as pickle_file:
                vprint(verbose,
                       "Loading pickle file : " + os.path.join(self.tmp_dir, os.path.basename(filename) + ".pickle"))
                return pickle.load(pickle_file)
        if 'task' not in self.info.keys():
            self.getTypeProblem(filename)

            # IG: Here change to accommodate the new multiclass label format
        if self.info['task'] == 'multilabel.classification':
            label = data_io.data(filename)
        elif self.info['task'] == 'multiclass.classification':
            label = data_converter.convert_to_num(data_io.data(filename))
        else:
            label = np.ravel(data_io.data(filename))  # get a column vector
        # label = np.array([np.ravel(data_io.data(filename))]).transpose() # get a column vector

        if self.use_pickle:
            with open(os.path.join(self.tmp_dir, os.path.basename(filename) + ".pickle"), "wb") as pickle_file:
                vprint(verbose,
                       "Saving pickle file : " + os.path.join(self.tmp_dir, os.path.basename(filename) + ".pickle"))
                p = pickle.Pickler(pickle_file)
                p.fast = True
                p.dump(label)
        end = time.time()
        if verbose:  print( "[+] Success in %5.2f sec" % (end - start))
        return label


    def loadType(self, filename, verbose=True):
        ''' Get the variable types'''
        if verbose:  print("========= Reading " + filename)
        start = time.time()
        type_list = []
        if os.path.isfile(filename):
            type_list = data_converter.file_to_array(filename, verbose=False)
        else:
            n = self.info['feat_num']
            type_list = [self.info['feat_type']] * n
        type_list = np.array(type_list).ravel()
        end = time.time()
        if verbose:  print( "[+] Success in %5.2f sec" % (end - start))
        return type_list


    def getInfo(self, filename, verbose=True):
        ''' Get all information {attribute = value} pairs from the filename (public.info file),
                  if it exists, otherwise, output default values'''
        if filename == None:
            basename = self.basename
            input_dir = self.input_dir
        else:
            basename = os.path.basename(filename).rsplit('_')[0]
            input_dir = os.path.dirname(filename)
        if os.path.exists(filename):
            self.getInfoFromFile(filename)
            vprint(verbose, "Info file found : " + os.path.abspath(filename))
            # Finds the data format ('dense', 'sparse', or 'sparse_binary')
            self.getFormatData(os.path.join(input_dir, basename + '_train.data'))
        else:
            vprint(verbose, "Info file NOT found : " + os.path.abspath(filename))
            # Hopefully this never happens because this is done in a very inefficient way
            # reading the data multiple times...
            self.info['usage'] = 'No Info File'
            self.info['name'] = basename
            # Get the data format and sparsity
            self.getFormatData(os.path.join(input_dir, basename + '_train.data'))
            # Assume no categorical variable and no missing value (we'll deal with that later)
            self.info['has_categorical'] = 0
            self.info['has_missing'] = 0
            # Get the target number, label number, target type and task
            self.getTypeProblem(os.path.join(input_dir, basename + '_train.solution'))
            if self.info['task'] == 'regression':
                self.info['metric'] = 'r2_metric'
            else:
                self.info['metric'] = 'auc_metric'
            # Feature type: Numerical, Categorical, or Binary
            # Can also be determined from [filename].type
            self.info['feat_type'] = 'Mixed'
            # Get the number of features and patterns
            self.getNbrFeatures(os.path.join(input_dir, basename + '_train.data'),
                                os.path.join(input_dir, basename + '_test.data'),
                                os.path.join(input_dir, basename + '_valid.data'))
            self.getNbrPatterns(basename, input_dir, 'train')
            self.getNbrPatterns(basename, input_dir, 'valid')
            self.getNbrPatterns(basename, input_dir, 'test')
            # Set default time budget
            self.info['time_budget'] = 600
        return self.info


    def getInfoFromFile(self, filename):
        ''' Get all information {attribute = value} pairs from the public.info file'''
        with open(filename, "r") as info_file:
            lines = info_file.readlines()
            features_list = list(map(lambda x: tuple(x.strip("\'").split(" = ")), lines))

            for (key, value) in features_list:
                self.info[key] = value.rstrip().strip("'").strip(' ')
                if self.info[key].isdigit():  # if we have a number, we want it to be an integer
                    self.info[key] = int(self.info[key])
        return self.info


    def getFormatData(self, filename):
        ''' Get the data format directly from the data file (in case we do not have an info file)'''
        if 'format' in self.info.keys():
            return self.info['format']
        if 'is_sparse' in self.info.keys():
            if self.info['is_sparse'] == 0:
                self.info['format'] = 'dense'
            else:
                data = data_converter.read_first_line(filename)
                if ':' in data[0]:
                    self.info['format'] = 'sparse'
                else:
                    self.info['format'] = 'sparse_binary'
        else:
            data = data_converter.file_to_array(filename)
            if ':' in data[0][0]:
                self.info['is_sparse'] = 1
                self.info['format'] = 'sparse'
            else:
                nbr_columns = len(data[0])
                for row in range(len(data)):
                    if len(data[row]) != nbr_columns:
                        self.info['format'] = 'sparse_binary'
                if 'format' not in self.info.keys():
                    self.info['format'] = 'dense'
                    self.info['is_sparse'] = 0
        return self.info['format']


    def getNbrFeatures(self, *filenames):
        ''' Get the number of features directly from the data file (in case we do not have an info file)'''
        if 'feat_num' not in self.info.keys():
            self.getFormatData(filenames[0])
            if self.info['format'] == 'dense':
                data = data_converter.file_to_array(filenames[0])
                self.info['feat_num'] = len(data[0])
            elif self.info['format'] == 'sparse':
                self.info['feat_num'] = 0
                for filename in filenames:
                    sparse_list = data_converter.sparse_file_to_sparse_list(filename)
                    last_column = [sparse_list[i][-1] for i in range(len(sparse_list))]
                    last_column_feature = [a for (a, b) in last_column]
                    self.info['feat_num'] = max(self.info['feat_num'], max(last_column_feature))
            elif self.info['format'] == 'sparse_binary':
                self.info['feat_num'] = 0
                for filename in filenames:
                    data = data_converter.file_to_array(filename)
                    last_column = [int(data[i][-1]) for i in range(len(data))]
                    self.info['feat_num'] = max(self.info['feat_num'], max(last_column))
        return self.info['feat_num']


    def getNbrPatterns(self, basename, info_dir, datatype):
        ''' Get the number of patterns directly from the data file (in case we do not have an info file)'''
        line_num = data_converter.num_lines(os.path.join(info_dir, basename + '_' + datatype + '.data'))
        self.info[datatype + '_num'] = line_num
        return line_num


    def getTypeProblem(self, solution_filename):
        ''' Get the type of problem directly from the solution file (in case we do not have an info file)'''
        if 'task' not in self.info.keys():
            solution = np.array(data_converter.file_to_array(solution_filename))
            target_num = solution.shape[1]
            self.info['target_num'] = target_num
            if target_num == 1:  # if we have only one column
                solution = np.ravel(solution)  # flatten
                nbr_unique_values = len(np.unique(solution))
                if nbr_unique_values < len(solution) / 8:
                    # Classification
                    self.info['label_num'] = nbr_unique_values
                    if nbr_unique_values == 2:
                        self.info['task'] = 'binary.classification'
                        self.info['target_type'] = 'Binary'
                    else:
                        self.info['task'] = 'multiclass.classification'
                        self.info['target_type'] = 'Categorical'
                else:
                    # Regression
                    self.info['label_num'] = 0
                    self.info['task'] = 'regression'
                    self.info['target_type'] = 'Numerical'
            else:
                # Multilabel or multiclass
                self.info['label_num'] = target_num
                self.info['target_type'] = 'Binary'
                if any(item > 1 for item in map(np.sum, solution.astype(int))):
                    self.info['task'] = 'multilabel.classification'
                else:
                    self.info['task'] = 'multiclass.classification'
        return self.info['task']