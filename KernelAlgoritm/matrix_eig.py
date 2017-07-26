import numpy as np

def orig_vec(data):
    matrix = []
    for i in  data['X']:
        matrix.append(np.hstack(i))
    data['X_vec'] = np.array(matrix)
    return data


#Random
def matrix_eig_k(data, k):
    new_data = {}
    new_data['y'] = data['y']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = abs(curs).argsort()[:k]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(curs, indeces_del)
        curs = curs * ( 2*np.random.randint(2, size= curs.shape[0]) - 1 )
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

def matrix_eig_110(data):
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


#No abs
def matrix_eig1_k(data, k):
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


def matrix_eig1_0(data):
    return matrix_eig1_k(data, k = 0)

def matrix_eig1_10(data):
    return matrix_eig1_k(data, k = 10)

def matrix_eig1_20(data):
    return matrix_eig1_k(data, k = 20)

def matrix_eig1_30(data):
    return matrix_eig1_k(data, k = 30)

def matrix_eig1_40(data):
    return matrix_eig1_k(data, k = 40)

def matrix_eig1_50(data):
    return matrix_eig1_k(data, k = 50)

def matrix_eig1_60(data):
    return matrix_eig1_k(data, k = 60)

def matrix_eig1_70(data):
    return matrix_eig1_k(data, k = 70)

def matrix_eig1_80(data):
    return matrix_eig1_k(data, k = 80)

def matrix_eig1_90(data):
    return matrix_eig1_k(data, k = 90)

def matrix_eig1_100(data):
    return matrix_eig1_k(data, k = 100)

def matrix_eig1_110(data):
    return matrix_eig1_k(data, k = 110)

def matrix_eig1_120(data):
    return matrix_eig1_k(data, k = 120)

def matrix_eig1_130(data):
    return matrix_eig1_k(data, k = 130)

def matrix_eig1_140(data):
    return matrix_eig1_k(data, k = 140)

def matrix_eig1_150(data):
    return matrix_eig1_k(data, k = 150)

def matrix_eig1_160(data):
    return matrix_eig1_k(data, k = 160)

def matrix_eig1_170(data):
    return matrix_eig1_k(data, k = 170)

def matrix_eig1_180(data):
    return matrix_eig1_k(data, k = 180)

def matrix_eig1_190(data):
    return matrix_eig1_k(data, k = 190)

def matrix_eig1_200(data):
    return matrix_eig1_k(data, k = 200)

def matrix_eig1_210(data):
    return matrix_eig1_k(data, k = 210)

def matrix_eig1_220(data):
    return matrix_eig1_k(data, k = 220)

def matrix_eig1_230(data):
    return matrix_eig1_k(data, k = 230)

def matrix_eig1_240(data):
    return matrix_eig1_k(data, k = 240)

def matrix_eig1_250(data):
    return matrix_eig1_k(data, k = 250)

def matrix_eig1_260(data):
    return matrix_eig1_k(data, k = 260)

def matrix_eig1_263(data):
    return matrix_eig1_k(data, k = 263)

#Abs
def matrix_eig2_k(data, k):
    new_data = {}
    new_data['y'] = data['y']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = abs(curs).argsort()[:k]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(curs, indeces_del)
        new_data['X'][i] = vecs.dot(np.diag(curs))
    return orig_vec(new_data)


def matrix_eig2_0(data):
    return matrix_eig2_k(data, k = 0)

def matrix_eig2_10(data):
    return matrix_eig2_k(data, k = 10)

def matrix_eig2_20(data):
    return matrix_eig2_k(data, k = 20)

def matrix_eig2_30(data):
    return matrix_eig2_k(data, k = 30)

def matrix_eig2_40(data):
    return matrix_eig2_k(data, k = 40)

def matrix_eig2_50(data):
    return matrix_eig2_k(data, k = 50)

def matrix_eig2_60(data):
    return matrix_eig2_k(data, k = 60)

def matrix_eig2_70(data):
    return matrix_eig2_k(data, k = 70)

def matrix_eig2_80(data):
    return matrix_eig2_k(data, k = 80)

def matrix_eig2_90(data):
    return matrix_eig2_k(data, k = 90)

def matrix_eig2_100(data):
    return matrix_eig2_k(data, k = 100)

def matrix_eig2_110(data):
    return matrix_eig2_k(data, k = 110)

def matrix_eig2_120(data):
    return matrix_eig2_k(data, k = 120)

def matrix_eig2_130(data):
    return matrix_eig2_k(data, k = 130)

def matrix_eig2_140(data):
    return matrix_eig2_k(data, k = 140)

def matrix_eig2_150(data):
    return matrix_eig2_k(data, k = 150)

def matrix_eig2_160(data):
    return matrix_eig2_k(data, k = 160)

def matrix_eig2_170(data):
    return matrix_eig2_k(data, k = 170)

def matrix_eig2_180(data):
    return matrix_eig2_k(data, k = 180)

def matrix_eig2_190(data):
    return matrix_eig2_k(data, k = 190)

def matrix_eig2_200(data):
    return matrix_eig2_k(data, k = 200)

def matrix_eig2_210(data):
    return matrix_eig2_k(data, k = 210)

def matrix_eig2_220(data):
    return matrix_eig2_k(data, k = 220)

def matrix_eig2_230(data):
    return matrix_eig2_k(data, k = 230)

def matrix_eig2_240(data):
    return matrix_eig2_k(data, k = 240)

def matrix_eig2_250(data):
    return matrix_eig2_k(data, k = 250)

def matrix_eig2_260(data):
    return matrix_eig2_k(data, k = 260)

def matrix_eig2_263(data):
    return matrix_eig2_k(data, k = 263)

#AbsAbs
def matrix_eig3_k(data, k):
    new_data = {}
    new_data['y'] = data['y']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = abs(curs).argsort()[:k]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(curs, indeces_del)
        new_data['X'][i] = vecs.dot(np.diag(abs(curs)))
    return orig_vec(new_data)


def matrix_eig3_0(data):
    return matrix_eig3_k(data, k = 0)

def matrix_eig3_10(data):
    return matrix_eig3_k(data, k = 10)

def matrix_eig3_20(data):
    return matrix_eig3_k(data, k = 20)

def matrix_eig3_30(data):
    return matrix_eig3_k(data, k = 30)

def matrix_eig3_40(data):
    return matrix_eig3_k(data, k = 40)

def matrix_eig3_50(data):
    return matrix_eig3_k(data, k = 50)

def matrix_eig3_60(data):
    return matrix_eig3_k(data, k = 60)

def matrix_eig3_70(data):
    return matrix_eig3_k(data, k = 70)

def matrix_eig3_80(data):
    return matrix_eig3_k(data, k = 80)

def matrix_eig3_90(data):
    return matrix_eig3_k(data, k = 90)

def matrix_eig3_100(data):
    return matrix_eig3_k(data, k = 100)

def matrix_eig3_110(data):
    return matrix_eig3_k(data, k = 110)

def matrix_eig3_120(data):
    return matrix_eig3_k(data, k = 120)

def matrix_eig3_130(data):
    return matrix_eig3_k(data, k = 130)

def matrix_eig3_140(data):
    return matrix_eig3_k(data, k = 140)

def matrix_eig3_150(data):
    return matrix_eig3_k(data, k = 150)

def matrix_eig3_160(data):
    return matrix_eig3_k(data, k = 160)

def matrix_eig3_170(data):
    return matrix_eig3_k(data, k = 170)

def matrix_eig3_180(data):
    return matrix_eig3_k(data, k = 180)

def matrix_eig3_190(data):
    return matrix_eig3_k(data, k = 190)

def matrix_eig3_200(data):
    return matrix_eig3_k(data, k = 200)

def matrix_eig3_210(data):
    return matrix_eig3_k(data, k = 210)

def matrix_eig3_220(data):
    return matrix_eig3_k(data, k = 220)

def matrix_eig3_230(data):
    return matrix_eig3_k(data, k = 230)

def matrix_eig3_240(data):
    return matrix_eig3_k(data, k = 240)

def matrix_eig3_250(data):
    return matrix_eig3_k(data, k = 250)

def matrix_eig3_260(data):
    return matrix_eig3_k(data, k = 260)

def matrix_eig3_263(data):
    return matrix_eig3_k(data, k = 263)


#Vecs
def matrix_vec1_k(data, k):
    new_data = {}
    new_data['y'] = data['y']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = curs.argsort()[:k]
        vecs = np.delete(vecs, indeces_del, axis=1)
        new_data['X'][i] = vecs
    return orig_vec(new_data)


def matrix_vec1_0(data):
    return matrix_vec1_k(data, k = 0)

def matrix_vec1_10(data):
    return matrix_vec1_k(data, k = 10)

def matrix_vec1_20(data):
    return matrix_vec1_k(data, k = 20)

def matrix_vec1_30(data):
    return matrix_vec1_k(data, k = 30)

def matrix_vec1_40(data):
    return matrix_vec1_k(data, k = 40)

def matrix_vec1_50(data):
    return matrix_vec1_k(data, k = 50)

def matrix_vec1_60(data):
    return matrix_vec1_k(data, k = 60)

def matrix_vec1_70(data):
    return matrix_vec1_k(data, k = 70)

def matrix_vec1_80(data):
    return matrix_vec1_k(data, k = 80)

def matrix_vec1_90(data):
    return matrix_vec1_k(data, k = 90)

def matrix_vec1_100(data):
    return matrix_vec1_k(data, k = 100)

def matrix_vec1_110(data):
    return matrix_vec1_k(data, k = 110)

def matrix_vec1_120(data):
    return matrix_vec1_k(data, k = 120)

def matrix_vec1_130(data):
    return matrix_vec1_k(data, k = 130)

def matrix_vec1_140(data):
    return matrix_vec1_k(data, k = 140)

def matrix_vec1_150(data):
    return matrix_vec1_k(data, k = 150)

def matrix_vec1_160(data):
    return matrix_vec1_k(data, k = 160)

def matrix_vec1_170(data):
    return matrix_vec1_k(data, k = 170)

def matrix_vec1_180(data):
    return matrix_vec1_k(data, k = 180)

def matrix_vec1_190(data):
    return matrix_vec1_k(data, k = 190)

def matrix_vec1_200(data):
    return matrix_vec1_k(data, k = 200)

def matrix_vec1_210(data):
    return matrix_vec1_k(data, k = 210)

def matrix_vec1_220(data):
    return matrix_vec1_k(data, k = 220)

def matrix_vec1_230(data):
    return matrix_vec1_k(data, k = 230)

def matrix_vec1_240(data):
    return matrix_vec1_k(data, k = 240)

def matrix_vec1_250(data):
    return matrix_vec1_k(data, k = 250)

def matrix_vec1_260(data):
    return matrix_vec1_k(data, k = 260)

def matrix_vec1_263(data):
    return matrix_vec1_k(data, k = 263)

#Vecs
def matrix_vec2_k(data, k):
    new_data = {}
    new_data['y'] = data['y']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = abs(curs).argsort()[:k]
        vecs = np.delete(vecs, indeces_del, axis=1)
        new_data['X'][i] = vecs
    return orig_vec(new_data)


def matrix_vec2_0(data):
    return matrix_vec2_k(data, k = 0)

def matrix_vec2_10(data):
    return matrix_vec2_k(data, k = 10)

def matrix_vec2_20(data):
    return matrix_vec2_k(data, k = 20)

def matrix_vec2_30(data):
    return matrix_vec2_k(data, k = 30)

def matrix_vec2_40(data):
    return matrix_vec2_k(data, k = 40)

def matrix_vec2_50(data):
    return matrix_vec2_k(data, k = 50)

def matrix_vec2_60(data):
    return matrix_vec2_k(data, k = 60)

def matrix_vec2_70(data):
    return matrix_vec2_k(data, k = 70)

def matrix_vec2_80(data):
    return matrix_vec2_k(data, k = 80)

def matrix_vec2_90(data):
    return matrix_vec2_k(data, k = 90)

def matrix_vec2_100(data):
    return matrix_vec2_k(data, k = 100)

def matrix_vec2_110(data):
    return matrix_vec2_k(data, k = 110)

def matrix_vec2_120(data):
    return matrix_vec2_k(data, k = 120)

def matrix_vec2_130(data):
    return matrix_vec2_k(data, k = 130)

def matrix_vec2_140(data):
    return matrix_vec2_k(data, k = 140)

def matrix_vec2_150(data):
    return matrix_vec2_k(data, k = 150)

def matrix_vec2_160(data):
    return matrix_vec2_k(data, k = 160)

def matrix_vec2_170(data):
    return matrix_vec2_k(data, k = 170)

def matrix_vec2_180(data):
    return matrix_vec2_k(data, k = 180)

def matrix_vec2_190(data):
    return matrix_vec2_k(data, k = 190)

def matrix_vec2_200(data):
    return matrix_vec2_k(data, k = 200)

def matrix_vec2_210(data):
    return matrix_vec2_k(data, k = 210)

def matrix_vec2_220(data):
    return matrix_vec2_k(data, k = 220)

def matrix_vec2_230(data):
    return matrix_vec2_k(data, k = 230)

def matrix_vec2_240(data):
    return matrix_vec2_k(data, k = 240)

def matrix_vec2_250(data):
    return matrix_vec2_k(data, k = 250)

def matrix_vec2_260(data):
    return matrix_vec2_k(data, k = 260)

def matrix_vec2_263(data):
    return matrix_vec2_k(data, k = 263)