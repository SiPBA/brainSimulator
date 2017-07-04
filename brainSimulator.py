# -*- coding: utf-8 -*-
"""
Performs a simulation of functional neuroimaging based on parameters
extracted from an existing dataset. 

Created on Thu Apr 28 15:53:15 2016

@author: pakitochus

Copyright (C) 2017 Francisco Jesús Martínez Murcia and SiPBA Research Group

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np

#Decomposition
#Good reconstruction is np.dot(Spca, pca.components_)+pca.mean_
from sklearn.decomposition import PCA, FastICA
def applyPCA(X, regularize=True, n_comp=-1):
    if(regularize):
        mean_ = np.mean(X, axis=0)
        X = X - mean_
        var_ = np.var(X,axis=0)
        X  = X/var_
    if n_comp==-1:
        n_comp = X.shape[0]-1
    pca = PCA(n_components=n_comp)
    Spca = pca.fit_transform(X)
    if not regularize:
        mean_ = pca.mean_
        var_ = None
    return Spca, pca.components_, mean_, var_
    
def applyICA(X, regularize=True, n_comp=-1):
    if(regularize):
        mean_ = np.mean(X, axis=0)
        X = X - mean_
        var_ = np.var(X,axis=0)
        X  = X/var_
    if n_comp==-1:
        n_comp = X.shape[0]-1
    ica = FastICA(n_components=n_comp)
    Sica = ica.fit_transform(X)
    if not regularize:
        mean_ = ica.mean_
        var_ = None
    return Sica, ica.components_, mean_, var_

#Density estimation 
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
import os
#os.chdir('pyStable')
#from stable import StableDist
#os.chdir('..')

class GaussianEstimator:
    """
    This class generates an interface for generating random numbers according
    to a certain gaussian parametrization, estimated from the data
    """
    def __init__(self, mean=0.0, var=1.0):
        self.mu = mean
        self.var = var
        
    def sample(self, dimension = 1.0):
        return self.var*np.random.randn(dimension) + self.mu
        
    def fit(self, x):
        self.mu = x.mean()
        self.var = x.var()
        
    def pdf(self, x):
        return (1/np.sqrt(2*self.var*np.pi))*np.exp(-np.power(x - self.mu, 2.) / (2 * self.var))
        
    def cdf(self, x):
        return np.exp(-np.power(x - self.mu, 2.) / (2 * self.var))

class MVNormalEstimator:
    """
    This class creates an interface for generating random numbers according
    to a given multivariate normal parametrization, estimated from the data
    Works only with python 3.4+ (due to numpy matrix multiplication)
    """
    def __init__(self, mean=0.0, cov=1.0):
        self.mu = mean
        self.cov = cov
        
    def sample(self, dimension = 1.0):
        return np.random.multivariate_normal(self.mu,self.cov,dimension)
        
    def fit(self, x):
        self.mu = x.mean(axis=0)
        self.cov = np.cov(x.T) # Faster and easier
	# self.cov = ((x-x.mean(axis=0))/data.shape[0]).T.dot(x-x.mean(axis=0)) # opcion más compleja.. timeit? 
        
    def pdf(self, x):
        part1 = 1 / ( ((2* np.pi)**(len(self.mu)/2)) * (np.linalg.det(self.cov)**(1/2)) )
        part2 = (-1/2) * ((x-self.mu).T.dot(np.linalg.inv(self.cov))).dot((x-self.mu))
        return float(part1 * np.exp(part2))
        
    def cdf(self, x):
        return np.exp(-np.power(x - self.mu, 2.) / (2 * self.var))
    

from scipy import fftpack, optimize
# Find the largest float available for this numpy
if hasattr(np, 'float128'):
    large_float = np.float128
elif hasattr(np, 'float96'):
    large_float = np.float96
else:
    large_float = np.float64
    
def _botev_fixed_point(t, M, I, a2):
    l = 7
    I = large_float(I)
    M = large_float(M)
    a2 = large_float(a2)
    f = 2 * np.pi ** (2 * l) * np.sum(I ** l * a2 *
                                      np.exp(-I * np.pi ** 2 * t))
    for s in range(l, 1, -1):
        K0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
        f = 2 * np.pi ** (2 * s) * \
            np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
    return t - (2 * M * np.sqrt(np.pi) * f) ** (-2 / 5)


def finite(val):
    return val is not None and np.isfinite(val)

def botev_bandwidth(data):
    """
    Implementation of the KDE bandwidth selection method outline in:

    Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Based on the implementation of Daniel B. Smith, PhD.

    The object is a callable returning the bandwidth for a 1D kernel.
    
    Forked from the package PyQT_fit. 
    """
#    def __init__(self, N=None, **kword):
#        if 'lower' in kword or 'upper' in kword:
#            print("Warning, using 'lower' and 'upper' for botev bandwidth is "
#                  "deprecated. Argument is ignored")
#        self.N = N
#
#    def __call__(self, data):#, model):
#        """
#        Returns the optimal bandwidth based on the data
#        """
    N = 2 ** 10 #if self.N is None else int(2 ** np.ceil(np.log2(self.N)))
#        lower = getattr(model, 'lower', None)
#        upper = getattr(model, 'upper', None)
#        if not finite(lower) or not finite(upper):
    minimum = np.min(data)
    maximum = np.max(data)
    span = maximum - minimum
    lower = minimum - span / 10 #if not finite(lower) else lower
    upper = maximum + span / 10 #if not finite(upper) else upper
    # Range of the data
    span = upper - lower

    # Histogram of the data to get a crude approximation of the density
#        weights = model.weights
#        if not weights.shape:
    weights = None
    M = len(data)
    DataHist, bins = np.histogram(data, bins=N, range=(lower, upper), weights=weights)
    DataHist = DataHist / M
    DCTData = fftpack.dct(DataHist, norm=None)

    I = np.arange(1, N, dtype=int) ** 2
    SqDCTData = (DCTData[1:] / 2) ** 2
    guess = 0.1

    try:
        t_star = optimize.brentq(_botev_fixed_point, 0, guess,
                                 args=(M, I, SqDCTData))
    except ValueError:
        t_star = .28 * N ** (-.4)

    return np.sqrt(t_star) * span


def estimateDensity(X, method='kde'):
    # Returns an estimator of the PDF of the current data. 
    if method is 'kde':
        kernel = KernelDensity(bandwidth=botev_bandwidth(X.flatten()))
#        params = {'bandwidth': np.logspace(-1, 20, 100)}
#        grid = GridSearchCV(KernelDensity(), params)
#        grid.fit(X[:,np.newaxis])
#        kernel = grid.best_estimator_
    elif method is 'stable':
        kernel = StableDist(1, 1, 0, 1)
    elif method is 'gaussian':
        kernel = GaussianEstimator()
    kernel.fit(X[:,np.newaxis])
    return kernel

def createDensityMatrices(X,labels, method='kde'):
    kernels = []    
    uniqLabels = list(set(labels)) # Is an array containing only class labels 
#    for lab in set(labels):
#        uniqLabels.append(lab)
    for idx,lab in enumerate(uniqLabels):
        if method is 'mvnormal':
            kernel = MVNormalEstimator()
            kernel.fit(X[labels==lab,:]) 
            kernels.append(kernel)
        else:
            kernels.append([])
            for el in X.T: # por columnas
                kernels[idx].append(estimateDensity(el[labels==lab], method=method))
    return kernels, uniqLabels


def createNewBrains(kernel, N, coef, mean, var=None):
    if not isinstance(kernel, list):
        newS = kernel.sample(N)
    else:
        newS = np.zeros((N, len(kernel)))
        i = 0
        for k in kernel:
            newS[:,i] = k.sample(N).flatten()
            i+=1
    simStack = np.dot(newS, coef)
    if var is not None:
        simStack = simStack*var
    simStack = simStack + mean
    return simStack
    

def generateDataset(stack, labels, N=100, algorithm='PCA', kernels=None, classes=None, COEFF=None, MEAN=None, method='kde',regularize=False, verbose=False, n_comp=-1):
    labels = labels.astype(int)
    if classes==None:
        classes = list(set(labels))
    selection = np.array([x in classes for x in labels])
    stack_fin = stack[selection,:]
    if kernels==None:
        if(verbose):
            print('Applying decomposition')
        if algorithm=='PCA':
            SCORE, COEFF, MEAN, VAR = applyPCA(stack_fin, regularize, n_comp)
        elif algorithm=='ICA':
            SCORE, COEFF, MEAN, VAR = applyICA(stack_fin, regularize, n_comp)
        if(verbose):
            print('Creating Density Matrices')
        kernels, uniqLabels = createDensityMatrices(SCORE, labels[selection], method=method)    
    for clas in classes:
        if(verbose):
            print('Creating brains with class %d'%clas)
        stackaux = createNewBrains(kernels[uniqLabels.index(clas)],N,COEFF,MEAN,VAR)
        labelsaux = np.array([clas]*N)
        if 'finStack' not in locals():
            finStack = stackaux
            labels = labelsaux
        else:
            finStack = np.vstack((finStack, stackaux))
            labels = np.hstack((labels, labelsaux))
    finStack[finStack<0]=0.
    return labels, finStack
#
