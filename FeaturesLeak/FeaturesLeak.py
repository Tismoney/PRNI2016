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
    data['X_vec'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1] * data['X'].shape[1]))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indx = abs(curs).argsort()
        vecs = vecs[indx]
        curs = curs[indx]
        data['X_vec'][i] = convert(vecs.dot(np.diag(curs)).T, 'mat2vec', 264)
    return data

def convert(A, mode, size = 264):
    if mode == 'mat2vec':
        A_vec = np.reshape(A, A.shape[0]*A.shape[1])
        return A_vec
        
    if mode == 'vec2mat':
        A_mat = np.reshape(A, (size, A.shape[0] / size ))
        return A_mat

params = {
    'text.usetex'         : True,
    'text.latex.unicode'  : True,
    'text.latex.preamble' : r"\usepackage[T2A]{fontenc}",
    'font.size'           : 15,
    'font.family'         : 'lmodern'
    }

plt.rcParams.update(params)

def print_boxplot(data1_1, data1_2, data2, figsize = (10.5,6.5), save = False):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
    
    bp  = ax2.boxplot(data2, 0, '+')
    
    x = np.arange(len(data1_1))
    ax1.plot(x, data1_1, 'b')
    ax1.plot(x, data1_2, 'r')
    
    plt.setp(bp['boxes'],    color='DarkGreen')
    plt.setp(bp['whiskers'], color='DarkOrange', linestyle = '-')
    plt.setp(bp['medians'],  color='DarkBlue')
    plt.setp(bp['caps'],     color='Gray')
    
    ax1.set_title(r'Process of learning')
    ax2.set_title(r'BoxPlot')
    ax1.set_ylabel(r'ROC AUC mean')
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    
    plt.show()
    if save: fig.savefig('fig.png', dpi = 300)        

def split_features(X, k_tr = 3, k_bag = 3, size_dim = 264):
    if k_tr > 0: X_tr = X[:,-size_dim * k_tr:]
    elif k_tr == 0: X_tr = []
    #print X_tr.shape, size_dim * k_tr
    X_bag = []
    for i in range( (size_dim - k_tr) / k_bag ):
        X_i = X[:, (0+i)*size_dim :(k_bag+i)*size_dim]
        X_bag.append(X_i)
    X_bag = np.array(X_bag)
    #print X_bag[0].shape, size_dim*k_bag
    
    return X_tr, X_bag

def supposed_index(X_bag):
    size = X_bag.shape[0] - 1
    
    ind = int(size - size * random.expovariate(1) / 5)
    if ind > size : ind = size
    if ind < 0  : ind = 0
    return ind

def join_features(X_tr, X_i):

    if np.array_equal(X_tr, []):
        return X_i
    else:
        return np.concatenate((X_tr, X_i), axis = 1)

def get_grid_and_score(X, y, grid_cv, eval_cv, X_ts = None, y_ts = None, collect_n = 0):
    steps = [('selector', VarianceThreshold()), ('scaler', MinMaxScaler()), ('classifier', LogisticRegression())] 
    pipeline = Pipeline(steps)

    param_grid = dict(classifier__penalty=['l1'], 
                      #classifier__C      =[0.1, 0.25, 0.5, 0.75, 1.0]
                      classifier__C      =[0.1, 0.5, 1.0]
                     )
    scoring = 'roc_auc'
    grid_clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, n_jobs=-1, cv=grid_cv)

    grid_clf.fit(X, y)
    
    steps[-1] = steps[-1][0], grid_clf.best_estimator_
    pipeline = Pipeline(steps)
    if not collect_n:
        scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=eval_cv, n_jobs=-1)
        
        pipeline.fit(X, y)
        y_pr = pipeline.predict(X_ts)
        real_score = roc_auc_score(y_ts, y_pr)

        return np.mean(scores), real_score
    if collect_n:
        scores = []
        rd = eval_cv.random_state
        for i in tqdm(range(collect_n)):
            eval_cv.random_state = i
            sc = cross_val_score(pipeline, X, y, scoring=scoring, cv=eval_cv, n_jobs=-1)
            scores.append(np.mean(sc))
        eval_cv.random_state = rd
        #print grid_clf.best_estimator_
        
        pipeline.fit(X, y)
        y_pr = pipeline.predict(X_ts)
        real_score = roc_auc_score(y_ts, y_pr)
        print "REAL SCORE: {}".format(real_score)
        return scores, real_score

class ChooseSubsection:
        

    def __init__(self, grid_cv, eval_cv,
        k_tr            = 3,
        k_bag           = 3,
        test_size       = 0.1,
        collect_n       = 10,
        size_dim        = 264,
        num_iteration   = 300,
        learning_rate   = 0.3,
        print_log       = True,
        max_vec         = -1,
        porog           = 0
    ):
        self.grid_cv        = grid_cv
        self.eval_cv        = eval_cv
        self.k_tr           = k_tr
        self.k_bag          = k_bag
        self.test_size      = test_size
        self.collect_n      = collect_n
        self.size_dim       = size_dim
        self.num_iteration  = num_iteration
        self.learning_rate  = learning_rate
        self.table_score    = []
        self.print_log      = print_log
        self.max_vec        = max_vec
        self.porog          = porog
        
    def init_data(self):

        self.data = '../Data/dti/'
        self.data = Transformer(get_autism).fit_transform(self.data)
        self.data = Transformer(binar_norm).fit_transform(self.data)
        self.data = Transformer(matrix_eig).fit_transform(self.data)
        self.X = self.data['X_vec']
        self.y = self.data['y']

    def split_data(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                            test_size=self.test_size, random_state =  int(time.time()))

    def fit_choose_vec(self, X_train = None, X_test = None, y_train = None, y_test = None):

        self.table_score = []
        self.table_real  = []

        #if X_train == None: X_train = self.X_train
        #if X_test  == None: X_train = self.X_test
        #if y_train == None: X_train = self.X_train
        #if y_test  == None: X_train = self.X_test

        X_tr, X_bag = split_features(X_train, self.k_tr, self.k_bag)
        X_ts, X_tag = split_features(X_test , self.k_tr, self.k_bag)
        ar_to_improve = np.zeros(X_bag.shape[0])
        
        score, real_score = get_grid_and_score(X_tr, y_train,
                             self.grid_cv, self.eval_cv, X_ts, y_test)

        #score = 0.5
        #real_score = 0.5

        self.table_score.append(score)
        self.table_real.append(real_score)

        if self.print_log: print "INIT\t SCORE: {:.3f}\t".format(self.table_score[0])

        for i in tqdm(range(self.num_iteration)):
            ind = supposed_index(X_bag)
            score, real_score = get_grid_and_score(join_features(X_tr, X_bag[ind]), y_train,
                                        self.grid_cv, self.eval_cv, 
                                        join_features(X_ts, X_tag[ind]), y_test)
            if (score - self.table_score[-1]) > self.porog:
                ar_to_improve[ind] += self.learning_rate * score
            for j, ar in enumerate(ar_to_improve):
                if ar >= 1:
                    X_tr = join_features(X_tr, X_bag[j])
                    X_ts = join_features(X_ts, X_tag[j])
                    X_bag = np.delete(X_bag, j, axis=0)
                    X_tag = np.delete(X_tag, j, axis=0) 
                    ar_to_improve = np.delete(ar_to_improve, j, axis=0)
                    self.table_score.append(score)
                    self.table_real.append(real_score)
                    if self.print_log: 
                        print "epoch # {}\t SCORE: {:.3f}\t ADD: {}\n".format(i, score, j)
            if self.max_vec != -1:
                if (self.k_tr + (len(self.table_score)-1)*self.k_bag) >= self.max_vec:
                    break

        self.X_tr = X_tr
        self.y_train = y_train
        self.X_ts = X_ts
        self.y_test = y_test 

    def get_result(self, print_box = True, save_mode = False):
        all_scores = get_grid_and_score(self.X_tr, self.y_train, 
                                    self.grid_cv, self.eval_cv,  
                                    self.X_ts, self.y_test, self.collect_n)

        if print_box: print_boxplot(self.table_score, self.table_real, all_scores, save = save_mode)
        return all_scores[1]

    def get_seed_result(self, index):

        #test_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state =  int(time.time()))
        #self.test_index = []
        score = []
        for train_index, test_index in index:#test_cv.split(self.X, self.y):
            self.fit_choose_vec(self.X[train_index], self.X[test_index],
                         self.y[train_index], self.y[test_index])
            score.append(self.get_result())
        print "REAL SCORE: {}".format(np.mean(score))
        return np.mean(score)
