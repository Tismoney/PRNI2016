import numpy as np

def orig_vec(data):
    matrix = []
    for i in  data['X']:
        matrix.append(np.hstack(i))
    data['X_vec'] = matrix
    return data

def matrix_eig_0(data, k = 0):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_5(data, k = 5):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_10(data, k = 10):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_20(data, k = 20):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_30(data, k = 30):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_40(data, k = 40):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_50(data, k = 50):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_60(data, k = 60):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_70(data, k = 70):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_80(data, k = 80):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_90(data, k = 90):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_100(data, k = 100):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_110(data, k = 110):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_120(data, k = 120):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_130(data, k = 130):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_140(data, k = 140):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_150(data, k = 150):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_160(data, k = 160):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_170(data, k = 170):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_180(data, k = 180):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_190(data, k = 190):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_200(data, k = 200):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_210(data, k = 210):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_220(data, k = 220):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_230(data, k = 230):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_240(data, k = 240):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_250(data, k = 250):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_260(data, k = 260):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)

def matrix_eig_263(data, k = 263):
    new_data = {}
    new_data['y'] = data['y']
    new_data['dist'] = data['dist']
    new_data['X'] = np.zeros(shape = (data['X'].shape[0], data['X'].shape[1], data['X'].shape[1] - k))
    for i in np.arange(data['X'].shape[0]):
        curs, vecs = np.linalg.eig(data['X'][i])
        indeces_del = range(curs.size)[(curs.size - k):]
        vecs = np.delete(vecs, indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=1)
        curs = np.delete(np.diag(curs), indeces_del, axis=0)
        new_data['X'][i] = vecs.dot(np.diag(curs)).astype('float')
    return orig_vec(new_data)
