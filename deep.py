'''
Deep (c) University of Manchester 2018
Deep is licensed under the MIT License.
To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.
@author:  Pablo Carbonell
@description: RBS prediction using deep learning
'''
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from keras.preprocessing.text import hashing_trick
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Activation, Dropout, Merge, Input
from keras.layers.merge import Concatenate
from keras.wrappers.scikit_learn import KerasRegressor
from hashlib import md5
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import normalize, StandardScaler

import csv
import numpy as np
import pandas
import matplotlib.pyplot as plt
import datetime
import pickle

def norms(v):
    return (v - np.mean(v))/np.std(v)

def norm(v):
    return (v - np.min(v))/(np.max(v) - np.min(v))

class Norm:
    """ Convenience class for two-way normaliser/denormaliser """
    def __init__(self,v):
        self.v = v
        self.max = np.max(v)
        self.min = np.min(v)
        self.mean = np.mean(v)
        self.std = np.std(v)

    def norm(self, x=None):
        if x is None:
            return (self.v - self.mean)/self.std
        else:
            return (x - self.mean)/self.std

    def denorm(self, x=None):
        if x is None:
            return self.v
        else:
            return x*self.std + self.mean
        
    def lnorm(self, x=None):
        if x is None:
            return (self.v - self.min)/(self.max - self.min )
        else:
            return (x - self.min)/(self.max - self.min )

    def ldenorm(self, x=None):
        if x is None:
            return self.v
        else:
            return x*(self.max - self.min ) + self.min


# Configuration parameters
conf = {'RBSLIB': 3, 'EPOCHS': 500, 'CV': False, 'myCV': True, 
        'cc1': None, 'cc2': None, 'cc3': None}

# Read dataset
if conf['RBSLIB'] == 3:
    dataframe = pandas.read_csv( os.path.join('rbs3.2', 'trainset.rbs3.csv') )
    train = dataframe.loc[:,'mvaE seq_1_A':'idi seq_2_T']
    odi = dataframe.loc[:,'ODind']
    odh = dataframe.loc[:,'ODharv']
    lim = dataframe.loc[:,'lim mg/l'] 
 #   y = lim * np.log10( odh/odi )
#    y = np.log10( lim /odi )
    y = np.log10(lim)
    nr = Norm(y)
    y = nr.lnorm()
    fullset = pandas.read_csv( os.path.join('rbs3.2', 'fullset.rbs3.csv'), header=None )

else:
    dataframe = pandas.read_csv( os.path.join('rbs2', 'trainset.rbs2.csv') )
    train = dataframe.loc[:,'mvaE seq_1_A':'idi seq_2_T']
    odi = dataframe.loc[:,'ODind']
    odh = dataframe.loc[:,'ODharv']
    lim = dataframe.loc[:,'lim mg/l'] 
    y = np.log10( lim )
 #   y = dataframe.loc[:,'lim mg/l'] 
 #   y = lim * np.log10( odh/odi )
    nr = Norm(y)
    y = nr.lnorm()
    fullset = pandas.read_csv( os.path.join('rbs2', 'newfullset.v2.csv'), header=None  )

def deepmodel():
    """ Model for library 1 """
    HIDDENDIM = 128
    HIDDENDIM2 = 128
    LOSS = 'mse'
    METRICS = 'mse'
    OPTIMIZER='adam'
    model = Sequential()
    model.add( Dense(HIDDENDIM, input_dim=train.shape[1], activation='sigmoid') )
    model.add( Dense(HIDDENDIM2, activation='relu') )
    model.add( Dense(1) )
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[METRICS])
    return model

def deepmodel2():
    """ Model for library 2 """
    HIDDENDIM = 128
    HIDDENDIM2 = 128
    LOSS = 'mse'
    METRICS = 'mse'
    OPTIMIZER='adam'
    model = Sequential()
    model.add( Dense(HIDDENDIM, input_dim=train.shape[1], activation='sigmoid') )
    model.add( Dense(HIDDENDIM2, activation='relu') )
    model.add( Dense(1) )
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[METRICS])
    return model


def deepmodel3():
    """ Model for library 3 """
    HIDDENDIM = 128
    HIDDENDIM2 = 128
    LOSS = 'mse'
    METRICS = 'mse'
    OPTIMIZER='adam'
    model = Sequential()
    model.add( Dense(HIDDENDIM, input_dim=train.shape[1], activation='relu') )
#    model.add( Dense(HIDDENDIM2, activation='sigmoid') )
    model.add( Dense(HIDDENDIM2, activation='relu') )
    model.add( Dense(1) )
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[METRICS])
    return model


#%%
""" Select non-empty columns """
colval = []
collib = []
for i in range(0, len(train.columns)):
    x = train.columns[i]
    if np.sum( train.loc[:,x] ) > 0:
        colval.append(x)
        collib.append(8+i)
train = train.loc[:, colval]
libfeat = fullset.iloc[:, collib]

#%%
""" Build the model """
if conf['RBSLIB'] == 2:
    model = deepmodel2()
else:
    model = deepmodel3()
model.fit(np.array(train), np.array(y), epochs=conf['EPOCHS']) #, validation_split=0.05)
cc1 = np.corrcoef( model.predict(np.array(train)).flatten(), np.array(y)) 
conf['cc1'] = cc1
print(cc1)
#estimator = KerasRegressor(build_fn=model, nb_epoch=EPOCHS, verbose=0)
pred = model.predict(np.array(train)).flatten()
obvs = np.array(y)
plt.scatter(obvs, pred)

ts = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())
outfile = os.path.join('models',ts+'.h5')
model.save(outfile)

#%%
""" Write the predictions """
pred = model.predict( np.array(libfeat) )
fullsetpred = np.append(fullset, pred, axis=1)
newfull = pandas.DataFrame( fullsetpred[ :, np.append(np.arange(0,8), [-1]) ] )
if conf['RBSLIB'] == 2:
    newfull.to_csv( os.path.join('rbs2', 'fullset.rbs2.pred'), header=None )
else:
    newfull.to_csv( os.path.join('rbs3.2', 'fullset.rbs3.pred'), header=None )
    
#%%
""" Cross-validations """
if conf['myCV']:
    k = np.array( [[],[]])
    kfold = KFold(n_splits=10)
    loo = LeaveOneOut()
    i = 1
    for train_index, test_index in loo.split(train):
        trset = np.array( train.iloc[train_index,] )
        teset = np.array( train.iloc[test_index,] )
        tryset = np.array( y[train_index] )
        teyset = np.array( y[test_index] )
        if conf['RBSLIB'] == 2:
            model = deepmodel2()
        else:
            model = deepmodel3()
        model = deepmodel()
        model.fit(trset, tryset, epochs=conf['EPOCHS'], verbose=False)
        tepred = model.predict(teset)
        k = np.stack( (np.append(k[0], teyset), np.append(k[1], tepred)) )
        if len(k[0]) > 1:
            cc3 = np.corrcoef(k[0], k[1])
            print( "%d/%d %.3f" % (i, train.shape[0], cc3[0,1]) )
        i += 1
    plt.scatter(k[0], k[1])
    conf['cc3'] = cc3
    
if conf['CV']:
    if conf['RBSLIB'] == 2:
        estimator = KerasRegressor(build_fn=deepmodel2, epochs=conf['EPOCHS'])
    else:
        estimator = KerasRegressor(build_fn=deepmodel3, epochs=['EPOCHS'])
    loo = LeaveOneOut()
    kfold = KFold(n_splits=10)
#    results = cross_val_score(estimator, np.array(train), np.array(y), cv=kfold)
    results = cross_val_predict(estimator, np.array(train), np.array(y), cv=10) #kfold)
    cc2 = np.corrcoef(results, obvs)
    print(cc2)
    plt.scatter(obvs, results)
    conf['cc2'] = cc2


with open(os.path.join('models',ts+'.pkl'), 'wb') as output:
    pickle.dump(conf, output)
