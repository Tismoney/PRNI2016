import numpy as np

def orig_vec(data):
    matrix = []
    for i in  data['X']:
        matrix.append(np.hstack(i))
    data['X_vec'] = np.array(matrix)
    return data

def matrix_eig_k(data, k):
    new_data = {}
    new_data['y'] = data['y']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = curs.argsort()[:k]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(curs, indeces_del)
        new_data['X'][i] = vecs.dot(np.diag(curs))
    return orig_vec(new_data)


def matrix_eig_0(data):
    return matrix_eig_k(data, k = 0)

def matrix_eig_10(data):
    return matrix_eig_k(data, k = 10)

def matrix_eig_20(data):
    return matrix_eig_k(data, k = 20)

def matrix_eig_30(data):
    return matrix_eig_k(data, k = 30)

def matrix_eig_40(data):
    return matrix_eig_k(data, k = 40)

def matrix_eig_50(data):
    return matrix_eig_k(data, k = 50)

def matrix_eig_60(data):
    return matrix_eig_k(data, k = 60)

def matrix_eig_70(data):
    return matrix_eig_k(data, k = 70)

def matrix_eig_80(data):
    return matrix_eig_k(data, k = 80)

def matrix_eig_90(data):
    return matrix_eig_k(data, k = 90)

def matrix_eig_100(data):
    return matrix_eig_k(data, k = 100)

def matrix_vec_110(data):
    return matrix_eig_k(data, k = 110)

def matrix_eig_120(data):
    return matrix_eig_k(data, k = 120)

def matrix_eig_130(data):
    return matrix_eig_k(data, k = 130)

def matrix_eig_140(data):
    return matrix_eig_k(data, k = 140)

def matrix_eig_150(data):
    return matrix_eig_k(data, k = 150)

def matrix_eig_160(data):
    return matrix_eig_k(data, k = 160)

def matrix_eig_170(data):
    return matrix_eig_k(data, k = 170)

def matrix_eig_180(data):
    return matrix_eig_k(data, k = 180)

def matrix_eig_190(data):
    return matrix_eig_k(data, k = 190)

def matrix_eig_200(data):
    return matrix_eig_k(data, k = 200)

def matrix_eig_210(data):
    return matrix_eig_k(data, k = 210)

def matrix_eig_220(data):
    return matrix_eig_k(data, k = 220)

def matrix_eig_230(data):
    return matrix_eig_k(data, k = 230)

def matrix_eig_240(data):
    return matrix_eig_k(data, k = 240)

def matrix_eig_250(data):
    return matrix_eig_k(data, k = 250)

def matrix_eig_260(data):
    return matrix_eig_k(data, k = 260)

def matrix_eig_263(data):
    return matrix_eig_k(data, k = 263)


def matrix_vec_k(data, k):
    new_data = {}
    new_data['y'] = data['y']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = abs(curs).argsort()[:k]
        vecs = np.delete(vecs, indeces_del, axis=1)
        new_data['X'][i] = vecs
    return orig_vec(new_data)


def matrix_vec_0(data):
    return matrix_vec_k(data, k = 0)

def matrix_vec_10(data):
    return matrix_vec_k(data, k = 10)

def matrix_vec_20(data):
    return matrix_vec_k(data, k = 20)

def matrix_vec_30(data):
    return matrix_vec_k(data, k = 30)

def matrix_vec_40(data):
    return matrix_vec_k(data, k = 40)

def matrix_vec_50(data):
    return matrix_vec_k(data, k = 50)

def matrix_vec_60(data):
    return matrix_vec_k(data, k = 60)

def matrix_vec_70(data):
    return matrix_vec_k(data, k = 70)

def matrix_vec_80(data):
    return matrix_vec_k(data, k = 80)

def matrix_vec_90(data):
    return matrix_vec_k(data, k = 90)

def matrix_vec_100(data):
    return matrix_vec_k(data, k = 100)

def matrix_vec_110(data):
    return matrix_vec_k(data, k = 110)

def matrix_vec_120(data):
    return matrix_vec_k(data, k = 120)

def matrix_vec_130(data):
    return matrix_vec_k(data, k = 130)

def matrix_vec_140(data):
    return matrix_vec_k(data, k = 140)

def matrix_vec_150(data):
    return matrix_vec_k(data, k = 150)

def matrix_vec_160(data):
    return matrix_vec_k(data, k = 160)

def matrix_vec_170(data):
    return matrix_vec_k(data, k = 170)

def matrix_vec_180(data):
    return matrix_vec_k(data, k = 180)

def matrix_vec_190(data):
    return matrix_vec_k(data, k = 190)

def matrix_vec_200(data):
    return matrix_vec_k(data, k = 200)

def matrix_vec_210(data):
    return matrix_vec_k(data, k = 210)

def matrix_vec_220(data):
    return matrix_vec_k(data, k = 220)

def matrix_vec_230(data):
    return matrix_vec_k(data, k = 230)

def matrix_vec_240(data):
    return matrix_vec_k(data, k = 240)

def matrix_vec_250(data):
    return matrix_vec_k(data, k = 250)

def matrix_vec_260(data):
    return matrix_vec_k(data, k = 260)

def matrix_vec_263(data):
    return matrix_vec_k(data, k = 263)



def matrix_old_k(data, k):
    new_data = {}
    new_data['y'] = data['y']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(curs, indeces_del)
        new_data['X'][i] = vecs.dot(np.diag(curs))
    return orig_vec(new_data)


def matrix_old_0(data):
    return matrix_old_k(data, k = 0)

def matrix_old_10(data):
    return matrix_old_k(data, k = 10)

def matrix_old_20(data):
    return matrix_old_k(data, k = 20)

def matrix_old_30(data):
    return matrix_old_k(data, k = 30)

def matrix_old_40(data):
    return matrix_old_k(data, k = 40)

def matrix_old_50(data):
    return matrix_old_k(data, k = 50)

def matrix_old_60(data):
    return matrix_old_k(data, k = 60)

def matrix_old_70(data):
    return matrix_old_k(data, k = 70)

def matrix_old_80(data):
    return matrix_old_k(data, k = 80)

def matrix_old_90(data):
    return matrix_old_k(data, k = 90)

def matrix_old_100(data):
    return matrix_old_k(data, k = 100)

def matrix_old_110(data):
    return matrix_old_k(data, k = 110)

def matrix_old_120(data):
    return matrix_old_k(data, k = 120)

def matrix_old_130(data):
    return matrix_old_k(data, k = 130)

def matrix_old_140(data):
    return matrix_old_k(data, k = 140)

def matrix_old_150(data):
    return matrix_old_k(data, k = 150)

def matrix_old_160(data):
    return matrix_old_k(data, k = 160)

def matrix_old_170(data):
    return matrix_old_k(data, k = 170)

def matrix_old_180(data):
    return matrix_old_k(data, k = 180)

def matrix_old_190(data):
    return matrix_old_k(data, k = 190)

def matrix_old_200(data):
    return matrix_old_k(data, k = 200)

def matrix_old_210(data):
    return matrix_old_k(data, k = 210)

def matrix_old_220(data):
    return matrix_old_k(data, k = 220)

def matrix_old_230(data):
    return matrix_old_k(data, k = 230)

def matrix_old_240(data):
    return matrix_old_k(data, k = 240)

def matrix_old_250(data):
    return matrix_old_k(data, k = 250)

def matrix_old_260(data):
    return matrix_old_k(data, k = 260)

def matrix_old_263(data):
    return matrix_old_k(data, k = 263)

