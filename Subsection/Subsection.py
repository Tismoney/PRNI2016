from reskit.norms import binar_norm
from reskit.core import Transformer, Pipeliner

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split

import os
import pandas as pd
import numpy as np
import random
import time

from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm


def get_autism(path_to_read='../Data/dti/', distances=True):
    def get_autism_distances(loc_name):
        with open(loc_name, 'r') as f:
            read_data = f.readlines()

        read_data = pd.DataFrame(
            np.array([np.array(item[:-1].split()).astype(int) for item in read_data]))

        return read_data

    def get_distance_matrix(coords):
        if type(coords) == pd.core.frame.DataFrame:
            coords = coords.values
        elif type(coords) != np.ndarray:
            print('Provide either pandas df or numpy array!')
            return -1

        shape = len(coords)
        dist_matrix = np.zeros((shape, shape))
        del shape
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist_matrix[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix

    target_vector = []  # this will be a target vector (diagnosis)
    matrices = []  # this will be a list of connectomes
    all_files = sorted(os.listdir(path_to_read))
    matrix_files = [
        item for item in all_files if 'DTI_connectivity' in item and 'All' not in item]
    distance_files = [
        item for item in all_files if 'DTI_region_xyz_centers' in item and 'All' not in item]

    # for each file in a sorted (!) list of files:
    for filename in matrix_files:

        A_dataframe = pd.read_csv(
            path_to_read + filename, sep='   ', header=None, engine='python')
        A = A_dataframe.values  # we will use a list of numpy arrays, NOT pandas dataframes
        matrices.append(A)# append a matrix to our list
        if "ASD" in filename:
            target_vector.append(1)
        elif "TD" in filename:
            target_vector.append(0)
    asd_dict = {}
    asd_dict['X'] = np.array(matrices)
    asd_dict['y'] = np.array(target_vector)
    if distances:
        dist_matrix_list = []
        for item in distance_files:
            # print(item)
            cur_coord = get_autism_distances(path_to_read + item)
            cur_dist_mtx = get_distance_matrix(cur_coord)
            dist_matrix_list += [cur_dist_mtx]

        asd_dict['dist'] = np.array(dist_matrix_list)

    return asd_dict

def matrix_eig(data):

    def convert(A, mode, size = 264):
        if mode == 'mat2vec':
            A_vec = np.reshape(A, A.shape[0]*A.shape[1])
            return A_vec
            
        if mode == 'vec2mat':
            A_mat = np.reshape(A, (size, A.shape[0] / size ))
            return A_mat    

    data['X_vec'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1] * data['X'].shape[1]))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indx = abs(curs).argsort()
        vecs = vecs[indx]
        curs = curs[indx]
        data['X_vec'][i] = convert(vecs.dot(np.diag(curs)).T, 'mat2vec', 264)
    return data

class Subsection_up(object):

    def __init__(self, eval_cv=None, grid_cv=None):

        self.eval_cv = eval_cv
        self.grid_cv = grid_cv

    def init_data(self, orig=False, binar=True, spectral=False):

        self.data = '../Data/dti/'
        self.data = Transformer(get_autism).fit_transform(self.data)
        if not orig:
            if binar: 
                self.data = Transformer(binar_norm).fit_transform(self.data)
            if spectral:
                self.data = Transformer(wbysqdist).fit_transform(self.data)
                self.data = Transformer(spectral_norm).fit_transform(self.data)
        self.data = Transformer(matrix_eig).fit_transform(self.data)
        self.X = self.data['X_vec']
        self.y = self.data['y']

    def init_params(self,
            k_tr            = 3,
            k_bag           = 3,
            collect_n       = 50,
            size_dim        = 264,
            num_iteration   = 300,
            learning_rate   = 0.3,
            max_vec         = -1,
            max_porog       = 0,
            param_grid      = dict(),
            ):

        self.k_tr           = k_tr
        self.k_bag          = k_bag
        self.collect_n      = collect_n
        self.size_dim       = size_dim
        self.num_iteration  = num_iteration
        self.learning_rate  = learning_rate
        self.max_vec        = max_vec
        self.max_porog      = max_porog
        self.param_grid     = param_grid
                
    def print_boxplot(self, data1, data2, latexmode = False, figsize = (10.5,6.5), save = False, filename = 'fig.png'):
        
        #Switch on latexstyle
        if latexmode == True: 
            params = {
                'text.usetex'         : True,
                'text.latex.unicode'  : True,
                'text.latex.preamble' : r"\usepackage[T2A]{fontenc}",
                'font.size'           : 15,
                'font.family'         : 'lmodern'
                }
            plt.rcParams.update(params)
        #Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
        #Plot gpahps
        x = np.arange(len(data1[0]))
        ax1.plot(x, data1[0], 'b')
        ax1.plot(x, data1[1], 'r')
        #Plot boxplot
        bp  = ax2.boxplot(data2, 0, '+')
        plt.setp(bp['boxes'],    color='DarkGreen')
        plt.setp(bp['whiskers'], color='DarkOrange', linestyle = '-')
        plt.setp(bp['medians'],  color='DarkBlue')
        plt.setp(bp['caps'],     color='Gray')
        #Set title
        ax1.set_title(r'Process of learning and real score')
        ax2.set_title(r'BoxPlot')
        ax1.set_ylabel(r'ROC AUC mean')
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        
        plt.show()
        if save: fig.savefig(filename, dpi = 450)        

    def choose_vec(self, X_train, X_test, y_train, y_test):

        def split_features(X, k_tr = 3, k_bag = 3, size_dim = 264):
            if k_tr > 0: X_tr = X[:,-size_dim * k_tr:]
            elif k_tr == 0: X_tr = []
            X_bag = []
            for i in range( (size_dim - k_tr) / k_bag ):
                X_i = X[:, (0+i)*size_dim :(k_bag+i)*size_dim]
                X_bag.append(X_i)
            X_bag = np.array(X_bag)
            return X_tr, X_bag

        def supposed_index(size):
            size = size - 1
            ind = int(size - size * random.expovariate(1) / 5)
            if ind > size : ind = size
            if ind < 0  : ind = 0
            return ind

        def join_features(X_tr, X_i):
            if np.array_equal(X_tr, []):
                return X_i
            else:
                return np.concatenate((X_tr, X_i), axis = 1)

        self.table_score = []
        self.table_real  = []

        X_tr, X_bag = split_features(X_train, self.k_tr, self.k_bag)
        X_ts, X_tag = split_features(X_test , self.k_tr, self.k_bag)
        ar_to_improve = np.zeros(X_bag.shape[0])
        
        score, real_score = self.get_grid_and_score(X_tr, y_train, X_ts, y_test)
        self.table_score.append(score)
        self.table_real.append(real_score)
        print "INIT\t SCORE: {:.3f}\t".format(self.table_score[0])

        for i in tqdm(range(self.num_iteration)):
            ind = supposed_index(X_bag.shape[0])
            score, real_score = self.get_grid_and_score(join_features(X_tr, X_bag[ind]), y_train, 
                                                        join_features(X_ts, X_tag[ind]), y_test)
            if (score - self.table_score[-1]) > self.max_porog:
                ar_to_improve[ind] += self.learning_rate * score
                if ar_to_improve[ind] >= 1:
                    X_tr = join_features(X_tr, X_bag[ind])
                    X_ts = join_features(X_ts, X_tag[ind])
                    X_bag = np.delete(X_bag, ind, axis=0)
                    X_tag = np.delete(X_tag, ind, axis=0) 
                    ar_to_improve = np.zeros(X_bag.shape[0])
                    self.table_score.append(score)
                    self.table_real.append(real_score) 
                    print "epoch # {}\t SCORE: {:.3f}\t ADD: {}\n".format(i, score, ind)
                if self.max_vec != -1:
                    if (self.k_tr + (len(self.table_score)-1)*self.k_bag) >= self.max_vec:
                        break

        self.X_tr = X_tr
        self.y_train = y_train
        self.X_ts = X_ts
        self.y_test = y_test 

    def get_grid_and_score(self, X_train, y_train, X_test, y_test, collect_n=0):
        
        steps = [('selector', VarianceThreshold()), ('scaler', MinMaxScaler()),('classifier', LogisticRegression(penalty='l1', C=0.5, max_iter=50))] 
        pipeline = Pipeline(steps)
        scoring = 'roc_auc'
        if not collect_n:
            scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=self.grid_cv, n_jobs=-1)
            pipeline.fit(X_train, y_train)
            y_pr = pipeline.predict(X_test)
            real_score = roc_auc_score(y_test, y_pr)
            return np.mean(scores), real_score
            
        if collect_n:
            scores = []
            rd = self.grid_cv.random_state
            for i in range(collect_n):
                self.grid_cv.random_state = i
                sc = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=self.grid_cv, n_jobs=-1)
                scores.append(np.mean(sc))
            self.grid_cv.random_state = rd
            pipeline.fit(X_train, y_train)
            y_pr = pipeline.predict(X_test)
            real_score = roc_auc_score(y_test, y_pr)
            print "REAL SCORE: {}".format(real_score)
            return scores, real_score

    def get_result(self, index = []):

        if np.array_equal(index, []):
            index = self.eval_cv.split(self.X, self.y)

        self.test_index = []
        score = []
        for train_index, test_index in index:#test_cv.split(self.X, self.y):
            self.choose_vec(self.X[train_index], self.X[test_index],
                         self.y[train_index], self.y[test_index])
            box_score, real_score = self.get_grid_and_score(self.X_tr, self.y_train,
                                self.X_ts, self.y_test, collect_n=self.collect_n)
            score.append(real_score)
            self.print_boxplot([self.table_score, self.table_real], [box_score, real_score])

        print "REAL SCORE: {}".format(np.mean(score))
        return np.mean(score)
    
    
class Subsection_low(object):

    def __init__(self, eval_cv=None, grid_cv=None):

        self.eval_cv = eval_cv
        self.grid_cv = grid_cv

    def init_data(self, orig=False, binar=True, spectral=False):
        
        self.data = '../Data/dti/'
        self.data = Transformer(get_autism).fit_transform(self.data)
        if not orig:
            if binar: 
                self.data = Transformer(binar_norm).fit_transform(self.data)
            if spectral:
                self.data = Transformer(wbysqdist).fit_transform(self.data)
                self.data = Transformer(spectral_norm).fit_transform(self.data)
        self.data = Transformer(matrix_eig).fit_transform(self.data)
        self.X = self.data['X_vec']
        self.y = self.data['y']

    def init_params(self,
            k_init          = 40,
            k_split         = 2,
            collect_n       = 50,
            size_dim        = 264,
            num_iteration   = 300,
            learning_rate   = 0.3,
            min_vec         = -1,
            max_porog       = 0,
            param_grid      = dict(),
            ):

        self.k_init         = k_init
        self.k_split        = k_split
        self.collect_n      = collect_n
        self.size_dim       = size_dim
        self.num_iteration  = num_iteration
        self.learning_rate  = learning_rate
        self.min_vec        = min_vec
        self.max_porog      = max_porog
        self.param_grid     = param_grid

    def print_boxplot(self, data1, data2, latexmode = False, figsize = (10.5,6.5), save = False, filename = 'fig.png'):
        
        #Switch on latexstyle
        if latexmode == True: 
            params = {
                'text.usetex'         : True,
                'text.latex.unicode'  : True,
                'text.latex.preamble' : r"\usepackage[T2A]{fontenc}",
                'font.size'           : 15,
                'font.family'         : 'lmodern'
                }
            plt.rcParams.update(params)
        #Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
        #Plot gpahps
        x = np.arange(len(data1[0]))
        ax1.plot(x, data1[0], 'b')
        ax1.plot(x, data1[1], 'r')
        #Plot boxplot
        bp  = ax2.boxplot(data2, 0, '+')
        plt.setp(bp['boxes'],    color='DarkGreen')
        plt.setp(bp['whiskers'], color='DarkOrange', linestyle = '-')
        plt.setp(bp['medians'],  color='DarkBlue')
        plt.setp(bp['caps'],     color='Gray')
        #Set title
        ax1.set_title(r'Process of learning and real score')
        ax2.set_title(r'BoxPlot')
        ax1.set_ylabel(r'ROC AUC mean')
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        
        plt.show()
        if save: fig.savefig(filename, dpi = 450)        

    def choose_vec(self, X_train, X_test, y_train, y_test):

        def split_features(X, k_init = 40, k_split = 2, size_dim = 264):
            X_bag = []
            for i in range( k_init/k_split ):
                X_i = X[:, (size_dim-k_init+i)*size_dim :(size_dim-k_init+k_split+i)*size_dim]
                X_bag.append(X_i)
            X_bag = np.array(X_bag)
            return X_bag

        def supposed_index(size):
            size = size - 1
            ind = int(size * random.expovariate(1) / 5)
            if ind > size : ind = size
            if ind < 0  : ind = 0
            return ind

        def del_features(X_bag, ind):
            X_tmp = np.delete(X_bag, ind, axis=0)
            return np.concatenate(X_tmp, axis=1)

        self.table_score = []
        self.table_real  = []

        X_bag = split_features(X_train, self.k_init, self.k_split)
        X_tag = split_features(X_test , self.k_init, self.k_split)
        ar_to_improve = np.zeros(X_bag.shape[0])

        score, real_score = self.get_grid_and_score(np.concatenate(X_bag, axis=1), y_train,
                                                    np.concatenate(X_tag, axis=1), y_test)
        self.table_score.append(score)
        self.table_real.append(real_score)
        print "INIT\t SCORE: {:.3f}\t".format(self.table_score[0])

        for i in tqdm(range(self.num_iteration)):
            ind = supposed_index(X_bag.shape[0])
            score, real_score = self.get_grid_and_score(del_features(X_bag, ind), y_train, 
                                                        del_features(X_tag, ind), y_test)
            if (score - self.table_score[-1]) > self.max_porog:
                ar_to_improve[ind] += self.learning_rate
                if ar_to_improve[ind] >= 1:
                    X_bag = np.delete(X_bag, ind, axis=0)
                    X_tag = np.delete(X_tag, ind, axis=0) 
                    ar_to_improve = np.delete(ar_to_improve, ind, axis=0)
                    self.table_score.append(score)
                    self.table_real.append(real_score)
                    print "epoch # {}\t SCORE: {:.3f}\t DEL: {}\n".format(i, score, ind)
                if self.min_vec != -1:
                    if (self.k_init - (len(self.table_score)-1)*self.k_split) <= self.min_vec:
                        break

        self.X_tr = np.concatenate(X_bag, axis=1)
        self.y_train = y_train
        self.X_ts = np.concatenate(X_tag, axis=1)
        self.y_test = y_test 

    def get_grid_and_score(self, X_train, y_train, X_test, y_test, collect_n=0):
        
        steps = [('selector', VarianceThreshold()), ('scaler', MinMaxScaler()),('classifier', LogisticRegression(penalty='l1', C=1.0, max_iter=50))] 
        pipeline = Pipeline(steps)
        scoring = 'roc_auc'
        if not collect_n:
            scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=self.grid_cv, n_jobs=-1)
            pipeline.fit(X_train, y_train)
            y_pr = pipeline.predict(X_test)
            real_score = roc_auc_score(y_test, y_pr)
            return np.mean(scores), real_score
            
        if collect_n:
            scores = []
            rd = self.eval_cv.random_state
            for i in range(1, collect_n+1):
                self.eval_cv.random_state = i
                sc = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=self.eval_cv, n_jobs=-1)
                scores.append(np.mean(sc))
            self.eval_cv.random_state = rd
            pipeline.fit(X_train, y_train)
            y_pr = pipeline.predict(X_test)
            real_score = roc_auc_score(y_test, y_pr)
            print "REAL SCORE: {}".format(real_score)
            return scores, real_score
     
    def get_result(self, index = []):
        if np.array_equal(index, []):
            index = self.eval_cv.split(self.X, self.y)

        self.test_index = []
        score = []
        for train_index, test_index in index:#test_cv.split(self.X, self.y):
            self.choose_vec(self.X[train_index], self.X[test_index],
                         self.y[train_index], self.y[test_index])
            box_score, real_score = self.get_grid_and_score(self.X_tr, self.y_train,
                                self.X_ts, self.y_test, collect_n=self.collect_n)
            score.append(real_score)
            self.print_boxplot([self.table_score, self.table_real], [box_score, real_score])

        print "REAL SCORE: {}".format(np.mean(score))
        return np.mean(score)

class Subsection_k(object):

    def __init__(self, eval_cv=None, grid_cv=None):

        self.eval_cv = eval_cv
        self.grid_cv = grid_cv

    def init_data(self, orig=False, binar=True, spectral=False):

        self.data = '../Data/dti/'
        self.data = Transformer(get_autism).fit_transform(self.data)
        if not orig:
            if binar: 
                self.data = Transformer(binar_norm).fit_transform(self.data)
            if spectral:
                self.data = Transformer(wbysqdist).fit_transform(self.data)
                self.data = Transformer(spectral_norm).fit_transform(self.data)
        self.data = Transformer(matrix_eig).fit_transform(self.data)
        self.X = self.data['X_vec']
        self.y = self.data['y']

    def init_params(self,
            k_init          = 40,
            collect_n       = 50,
            size_dim        = 264,
            param_grid      = dict(),
            ):

        self.k_init         = k_init
        self.collect_n      = collect_n
        self.size_dim       = size_dim
        self.param_grid     = param_grid

    def print_boxplot(self, data, latexmode = False, figsize = (10.5,6.5), save = False, filename = 'fig.png'):
        
        #Switch on latexstyle
        if latexmode == True: 
            params = {
                'text.usetex'         : True,
                'text.latex.unicode'  : True,
                'text.latex.preamble' : r"\usepackage[T2A]{fontenc}",
                'font.size'           : 15,
                'font.family'         : 'lmodern'
                }
            plt.rcParams.update(params)
        #Figure
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=figsize)
        #Plot boxplot
        bp  = ax.boxplot(data, 0, '+')
        plt.setp(bp['boxes'],    color='DarkGreen')
        plt.setp(bp['whiskers'], color='DarkOrange', linestyle = '-')
        plt.setp(bp['medians'],  color='DarkBlue')
        plt.setp(bp['caps'],     color='Gray')
        #Set title
        ax.set_title(r'BoxPlot')
        ax.set_ylabel(r'ROC AUC mean')
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        
        plt.show()
        if save: fig.savefig(filename, dpi = 450)        

    def choose_vec(self, X_train, X_test, y_train, y_test):

        def split_features(X, k_init = 40, size_dim = 264):
            X_bag = X[:, -k_init*size_dim:]
            return X_bag

        self.X_tr = split_features(X_train, self.k_init)
        self.y_train = y_train
        self.X_ts = split_features(X_test , self.k_init)
        self.y_test = y_test 

    def get_grid_and_score(self, X_train, y_train, X_test, y_test, collect_n=0):
        
        steps = [('selector', VarianceThreshold()), ('scaler', MinMaxScaler()),('classifier', LogisticRegression(penalty='l1', C=0.5, max_iter=50))] 
        pipeline = Pipeline(steps)
        scoring = 'roc_auc'
        if not collect_n:
            scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=self.grid_cv, n_jobs=-1)
            pipeline.fit(X_train, y_train)
            y_pr = pipeline.predict(X_test)
            real_score = roc_auc_score(y_test, y_pr)
            print "REAL SCORE: {}".format(real_score)
            return np.mean(scores), real_score
            
        if collect_n:
            scores = []
            rd = self.grid_cv.random_state
            for i in range(1, collect_n+1):
                self.grid_cv.random_state = i
                sc = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=self.grid_cv, n_jobs=-1)
                scores.append(np.mean(sc))
            self.grid_cv.random_state = rd
            pipeline.fit(X_train, y_train)
            y_pr = pipeline.predict(X_test)
            real_score = roc_auc_score(y_test, y_pr)
            print "REAL SCORE: {}".format(real_score)
            return scores, real_score
        
    def get_result(self, index = []):
        if np.array_equal(index, []):
            index = self.eval_cv.split(self.X, self.y)

        self.test_index = []
        score = []
        for train_index, test_index in index:#test_cv.split(self.X, self.y):
            self.choose_vec(self.X[train_index], self.X[test_index],
                         self.y[train_index], self.y[test_index])
            box_score, real_score = self.get_grid_and_score(self.X_tr, self.y_train,
                                self.X_ts, self.y_test, collect_n=0)
            score.append(real_score)
            #self.print_boxplot(box_score, real_score)

        print "REAL SCORE: {}".format(np.mean(score))
        return np.mean(score)