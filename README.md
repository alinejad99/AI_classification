# AI_classification


This repository contains the code and documentation for a machine learning classification project. The project is divided into two phases.

Phase 1
a. Feature Calculation and Normalization
Calculate a set of features for different channels of training data. Normalize the feature matrix.
b. Feature Selection
Select the top features based on a chosen metric, such as scatter matrix-based metrics or any other suitable metric.
c. MLP Network Design and Training
Design a Multilayer Perceptron (MLP) network and train it using different feature sets.
Calculate the average accuracy of the classifier using 5-fold cross-validation.
Experiment with different network architectures, including the number of layers, neurons per layer, activation functions, and feature subsets.
d. RBF Network Design and Training
Repeat the process for a Radial Basis Function (RBF) network.
e. Results and Reporting
Create a comprehensive report summarizing the results of each part.
Present the best-designed MLP and RBF networks and the selected optimal features.
Compare the results of parts (c) and (d).
Phase 2
Feature Selection Using Evolutionary Algorithms
Utilize evolutionary algorithms or swarm intelligence algorithms for feature selection from the extracted features.
Define an appropriate fitness function, considering criteria such as scatter matrices in higher dimensions, classification accuracy, or other suitable metrics.
Re-train the MLP and RBF classifiers using the selected features.
Apply the trained networks to test data to determine class labels.
Feel free to explore and adapt the provided code and algorithms for feature selection, classifier design, and evaluation. Create a comprehensive report summarizing the results and performance of your designed classifiers in both phases.












import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import random
import sys

#from pyeasyga import pyeasyga
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from seaborn import heatmap

from tensorflow.keras.layers import Input, Dense, Activation, Softmax
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from scipy.io import loadmat
from scipy.signal import butter, lfilter , freqz
from scipy import stats

data = loadmat("/content/drive/MyDrive/Colab Notebooks/CI/All_data.mat");
data.keys()

"""##**(A)**

#####Transposing at first
"""

x_train = np.transpose(data['x_train'])
x_test = np.transpose(data['x_test'])
y_train = np.transpose(data['y_train'])
print( x_train.shape , x_test.shape , y_train.shape )

"""###--> Variance"""

def V(x_train):
  v = np.zeros([x_train.shape[0], x_train.shape[1]])
  for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
     v[i, j] = x_train[i,j, 0:].var()
  return np.array(v)
G1 = V(x_train)
G1.shape

"""###--> Freq. Bounds Energy"""

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def FBE(x_train):
  power_n = np.zeros([x_train.shape[0],7*x_train.shape[1]])

  for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):

      sig     = x_train[i,j,:]

      theta   = np.mean(np.square(butter_bandpass_filter(sig, lowcut=2 , highcut=8 ,   fs=100, order=5)))#/Energy
      alpha   = np.mean(np.square(butter_bandpass_filter(sig, lowcut=9 , highcut=15,   fs=100, order=5)))#/Energy
      beta1   = np.mean(np.square(butter_bandpass_filter(sig, lowcut=16, highcut=22,   fs=100, order=5)))#/Energy
      beta2   = np.mean(np.square(butter_bandpass_filter(sig, lowcut=23, highcut=29,   fs=100, order=5)))#/Energy
      gamma1  = np.mean(np.square(butter_bandpass_filter(sig, lowcut=30, highcut=36,   fs=100, order=5)))#/Energy
      gamma2  = np.mean(np.square(butter_bandpass_filter(sig, lowcut=37, highcut=43,   fs=100, order=5)))#/Energy
      gamma3  = np.mean(np.square(butter_bandpass_filter(sig, lowcut=44, highcut=49.9, fs=100, order=5)))#/Energy

      power_n[ i , 7*(j):7*(j+1)] = np.array([theta, alpha, beta1, beta2, gamma1, gamma2, gamma3])

  return power_n
G2 = FBE(x_train)
G2.shape

"""##--> Corelation"""

def correlation(x_train):
  LM = []
  for i in range(x_train.shape[0]):
    Co = np.corrcoef(x_train[i])
    L = []
    for j in range(Co.shape[0]):
      for k in range(j+1):
        L.append(Co[j,k])
    LM.append(L)
  return np.array(LM)
G3 = correlation(x_train)
G3.shape

"""###--> Histogram"""

def H(x_train, B):
  max_voltage = np.max(x_train)
  min_voltage = np.min(x_train)
  bins = np.linspace(min_voltage, max_voltage, B)
  H_x = np.zeros([x_train.shape[0],x_train.shape[1]*B])
  for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
      signal = x_train[i, j, :]
      M = np.histogram(signal, bins=bins)[0]
      H_x[i, j*B:B*(j+1)-1] = M
  return H_x
G4 = H(x_train, 15)
G_4 = G4.ravel()
G4 = G_4.reshape(-1, 28*15 )
np.sum(G4)
G4.shape

"""###--> Gathering"""

NG = np.concatenate((G1, G2, G3, G4), axis=1)
#G = np.concatenate((G1, G2, G3, G4), axis=1)
#S = np.zeros([G.shape[1]])
#M = np.zeros([G.shape[1]])
#NG = np.zeros([G.shape[0],G.shape[1]])
#for i in range(G.shape[1]):
#  for j in range(G.shape[0]):
#   M[i] = G[:,i].mean()
#   S[i] = G[:,i].var()
#   NG[j,i] = (G[j,i]-M[i])/S[i]

print(NG.shape)

"""##**(B)**

--> Calculating **J-Value**
"""

Class_0 = np.where(y_train == 0)[0]
Class_1 = np.where(y_train == 1)[0]

# u0_G = np.zeros([1,G.shape[1]])
u0_NG = np.mean(NG,axis=0)

# u1_NG = np.zeros([1,NG.shape[1]])
u1_NG = np.mean(NG[Class_0,:],axis=0)

# u2_NG = np.zeros([1,NG.shape[1]])
u2_NG = np.mean(NG[Class_1,:],axis=0)

sigma1 = np.var(NG[Class_0,:],axis=0)
sigma2 = np.var(NG[Class_1,:],axis=0)

J_V = np.zeros([1,NG.shape[1]])
for i in range(NG.shape[1]):
 J_V[0, i] = ((u0_NG[i]-u1_NG[i])**2+(u0_NG[i]-u2_NG[i])**2)/((sigma1[i]**2)+(sigma2[i]**2))

print(NG.shape)
J_V = J_V.reshape((-1, ))
print(J_V.shape)

"""

---




---




**---> Ba tavajoh be tasvir zir , treshold 1.2 ra be onvan e marz taeen mikonim va vizhegi hayee ke J-value bishtar az 1.2 darand ra dar J_best zakhire mikonim.**"""

plt.bar(np.arange(1050), J_V)



"""**--> Dar nahayat feature ha ro select krdim**"""

type(J_V)
TH = 2
#J_best = J_V > 1.2
J_Best = np.where(J_V >= TH)[0]
NG_Best = NG[:,J_Best]
#print(J_Best)
print(NG_Best.shape)

"""###--> x_test and feature extracting"""

G1_test = V(x_test)
G2_test = FBE(x_test)
G3_test = correlation(x_test)
G4_test = H(x_test,B=15)
G_test = np.concatenate((G1_test,G2_test,G3_test,G4_test), axis=1)
G_test.shape

selected = np.where(J_V >= TH)[0]
test_selected = np.zeros([G_test.shape[0],43])
test_selected = G_test[:,selected]
test_selected.shape

"""##**--> Training the NN**

###**--> Train-validation split**
"""

x_tr, x_va, y_tr, y_va = train_test_split(NG_Best, y_train, test_size=0.2, random_state=0)
print(x_tr.shape, x_va.shape)

"""##**--> Normalization**"""

scaler = StandardScaler()
scaler.fit(x_tr)
x_tr_norm = scaler.transform(x_tr)
x_va_norm = scaler.transform(x_va)
x_te_norm = scaler.transform(NG_Best)

"""##**--> One-hot labels**"""

y_tr_hot = to_categorical(y_tr)
y_va_hot = to_categorical(y_va)
print(y_tr_hot.shape)

"""#--> **MLP**

####**--> Layers**

###**برای تعیین تعداد لایه ها و نورون های شبکه با آزمون و خطا و چک کردن چند عدد به این نتیجه رسیدم (بجای 5 fold cross-V)ازونجایی که دیتای محدود و کم تعدادی داریم نرخ لرنینگ رو کم کردم تا روند یادگیری دیرتر صورت بگیرد**
"""

model = keras.Sequential([
                     Input(shape=(x_tr.shape[1],)),
                     Dense(units=5,kernel_initializer=keras.initializers.RandomNormal(stddev=0.0001)),
                     Activation(activation=tf.math.tanh),
                     Dense(units=2,),
                     Softmax(axis=1)
])

model.summary()

"""### Compile Model

"""

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-6),
    loss = 'mean_squared_error',
    metrics = ['accuracy']
)

"""###**--> Training Model**"""

er_stop = EarlyStopping(monitor='val_accuracy',patience=30,restore_best_weights=True)

hist = model.fit(
    x_tr_norm,
    y_tr_hot,
    batch_size = x_tr.shape[0],
    epochs = 200,
    validation_data = (x_va_norm, y_va_hot),
    callbacks = [er_stop]
)

"""**با توجه به نتایج بالا و نزدیکی خطای ولیدیشن و تست نتیجه میگیریم که شبکه خوشبختانه دچار اور فیتینگ نشده**

###**--> Loss and accuracy plots**
"""

print(type(hist))
print (hist.history.keys())

n_epochs = len(hist.history['accuracy'])
epochs = np.arange(1,n_epochs+1)

fig, ax = plt.subplots()
ax.plot(epochs, hist.history['accuracy'])
ax.plot(epochs, hist.history['val_accuracy'])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_ylabel('ACC')
ax.legend(['Training','Validation']);

Bestresults = np.array(loadmat("/content/drive/MyDrive/Colab Notebooks/CI/test_label.mat"));
print(Bestresults)

from numpy import save
Predict = model.predict(test_selected)
Predict = np.argmax(Predict,axis=1)
#save('D',Predict)
#print(type(D))
print(Predict.shape)

"""##**--> در اینجا لیبل ها را قرار میدهیم تا ارزیابی از عمکرد شبکه بدست آوریم و تا آن موقع کامنتش میکنیم**"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#results = model.evaluate(#Label, Predict)
#print(results)
#acc = accuracy_score(y_test, Predict)
#print(f'acc = {acc}')
#
#print(classification_report(y_test, Predict))
#
#cm = confusion_matrix(y_test,Predict, normalize='true')
#print(cm)

"""###**--> RBF**"""

from keras import backend
import keras as K
from keras.layers import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np
import pandas as pd
from keras.models import Sequential


class InitCentersKMeans(Initializer):

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])

	# type checking to access elements of data correctly
        if type(self.X) == np.ndarray:
            return self.X[idx, :]
        elif type(self.X) == pd.core.frame.DataFrame:
            return self.X.iloc[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.

    # Example

    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```


    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas

    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RBFLayer(Layer):
        model = Sequential()
        model.add(RBFLayer(10, betas=2.0, input_shape=(1,)))
        model.add(Dense(1))


  #output_dim: number of hidden units (i.e. number of outputs of the layer)
  #initializer: instance of initiliazer to initialize centers
  #betas: float, initial value for betas

"""###**--> Genetic Algorithms**

"""

!pip install pyeasyga

"""###**--> اینجا سعی میکنیم با الگوریتم های ژنتیک یه سری فیچر استخراج کنیم و در نهایت با فیچر های سلکت شده جدید شبکه عصبی را آموزش دهیم و ارزیابی کنیم (MLP و RBF)**"""

def crossover(parent1, parent2):    # 2-point cross over
    points = sorted([random.randint(0,len(parent1)-1) for _ in (1,2)])
    child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    return child1,child2

def fitness(individual, data):
    indiv = np.array(individual)
    for i in range(len(indiv)):
     for j in range(NG.shape[1]):
      C1 = np.where(indiv==0)[0]
      C2 = np.where(indiv==1)[0]
      u0 = np.mean(NG[:,:],axis=0)
      u1 = np.mean(NG[:,C1],axis=0)
      u2 = np.mean(NG[:,C2],axis=0)
      S_1 = (1/C1.shape[0])*(C1[j]-u1)*(np.transpose(C1[j]-u1))
      S_2 = (1/C2.shape[0])*(C2[j]-u2)*(np.transpose(C2[j]-u2))

      S_W = np.trace(np.concatenate((S_1, S_2),axis=0))
      S_B = np.trace(np.concatenate((((u1-u0)*(np.transpose(u1-u0))),((u2-u0)*(np.transpose(u2-u0)))),axis=0))
      J = (S_B)/(S_W)
    return J


data = [0]*1050
ga = pyeasyga.GeneticAlgorithm(data)
ga.population_size = 100
ga.generations = 50
ga.fitness_function = fitness
ga.crossover_function = crossover

ga.run()
print(f'Best individual: {ga.best_individual()}')
