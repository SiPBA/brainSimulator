# -*- coding: utf-8 -*-
"""
Performs a simulation of functional neuroimaging based on parameters
extracted from an existing dataset. 

Created on Thu Apr 28 15:53:15 2016
Last update: 9 Aug, 2017

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
import numbers

#Decomposition
#Good reconstruction is np.dot(Spca, pca.components_)+pca.mean_
from sklearn.decomposition import PCA, FastICA
def applyPCA(X, regularize=True, n_comp=-1):
    if(regularize):
        mean_ = np.mean(X, axis=0)
        X = X - mean_
        var_ = np.var(X,axis=0)
        X  = X/var_
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
    ica = FastICA(n_components=n_comp)
    Sica = ica.fit_transform(X)
    if not regularize:
        mean_ = ica.mean_
        var_ = None
    return Sica, ica.components_, mean_, var_

#os.chdir('pyStable')
#from stable import StableDist
#os.chdir('..')

class GaussianEstimator:
    """
    This class generates an interface for generating random numbers according
    to a per-component gaussian parametrization, estimated from the data
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
        from scipy.stats import multivariate_normal
        self.mu = mean
        self.cov = cov
        self.model = multivariate_normal
        
    def sample(self, dimension = 1.0):
        return np.random.multivariate_normal(mean=self.mu, cov=self.cov, size=dimension)
        
    def fit(self, x):
        from sklearn.covariance import ledoit_wolf
        self.mu = x.mean(axis=0)
        self.cov = ledoit_wolf(x)[0] # Faster and easier
	# self.cov = ((x-x.mean(axis=0))/data.shape[0]).T.dot(x-x.mean(axis=0)) # opcion más compleja.. timeit? 
        
    def pdf(self, x):
        return self.model.pdf(x, mean=self.mu, cov=self.cov)
        
    def cdf(self, x):
        return np.exp(-np.power(x - self.mu, 2.) / (2 * self.var))
    
#Density estimation 
class KDEestimator:
    """
    This class generates an interface for generating random numbers according
    to a certain gaussian parametrization, estimated from the data
    """
    def __init__(self, bandwidth=1.0):
        from sklearn.neighbors.kde import KernelDensity
        self.bandwidth = bandwidth
        self.model = KernelDensity(bandwidth=self.bandwidth)
        
    def _botev_fixed_point(self, t, M, I, a2):
        # Find the largest float available for this numpy
        if hasattr(np, 'float128'):
            large_float = np.float128
        elif hasattr(np, 'float96'):
            large_float = np.float96
        else:
            large_float = np.float64
            
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
    
    
    def finite(self, val):
        return val is not None and np.isfinite(val)
    
    def botev_bandwidth(self, data):
        """
        Implementation of the KDE bandwidth selection method outline in:
    
        Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
        estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
    
        Based on the implementation of Daniel B. Smith, PhD.
    
        The object is a callable returning the bandwidth for a 1D kernel.
        
        Forked from the package PyQT_fit. 
        """
        from scipy import fftpack, optimize
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
            t_star = optimize.brentq(self._botev_fixed_point, 0, guess,
                                     args=(M, I, SqDCTData))
        except ValueError:
            t_star = .28 * N ** (-.4)
    
        return np.sqrt(t_star) * span
    
    def fit(self, x):
        self.bandwidth = self.botev_bandwidth(x.flatten())
        self.model.set_params(**{'bandwidth': self.bandwidth})
        self.model.fit(x.reshape(-1,1))
        
    def sample(self, dimension = 1.0):
        return self.model.sample(dimension)

    def pdf(self, x):
        return self.model.score_samples(x)

class BrainSimulator:
    
    def __init__(self, method = 'kde', algorithm='PCA', N=100, n_comp=-1, regularize=False, verbose=False):
        self.method = method #PDF estimation method
        self.algorithm = algorithm # algorithm used to decompose the dataset (PCA, ICA)
#        self.N = N # Number of samples per class
        self.n_comp = n_comp # Number of components used in the estimation
        self.verbose = verbose
        self.regularize = regularize # Sets regularization of decomposition via ICA or PCA. 
        self.kernels = None
        self.uniqLabels = None
        self.SCORE = None
        self.COEFF = None
        self.MEAN = None
        self.VAR = None
        
        
    def decompose(self, stack, labels): 
        """
        Performs the decomposition of the dataset. 
        Inputs:
            stack -> stack comprising the whole database to be decomposed
            labels -> list (or array) containing the labels of each subject
        """
        if(self.verbose):
            print('Applying decomposition')
        N_el = stack.shape[0]-1
        if self.n_comp==-1:
            self.n_comp = N_el
        if self.algorithm=='PCA':
            # Force to extract all components, and then extract the number of components in model
            self.SCORE, self.COEFF, self.MEAN, self.VAR = applyPCA(stack, self.regularize, N_el)
        elif self.algorithm=='ICA':
            self.SCORE, self.COEFF, self.MEAN, self.VAR = applyICA(stack, self.regularize, self.n_comp)
        
        return self.SCORE, self.COEFF, self.MEAN, self.VAR

    def estimateDensity(self, X):
        """
         Returns an estimator of the PDF of the current data. 
         Inputs:
             X -> the data from which the different kernels are fitted. 
        """
        if self.method is 'kde':
            kernel = KDEestimator()
        elif self.method is 'stable':
    #        kernel = StableDist(1, 1, 0, 1)
            print('Not yet supported')
        elif self.method is 'gaussian':
            kernel = GaussianEstimator()
        kernel.fit(X)
        return kernel
    
    def model(self, labels):
        """
        Models the per-class distribution of scores and sets the kernels. 
        """
        if(self.verbose):
            print('Creating Density Matrices')
        self.kernels = []    
        # UniqLabels is key. Here the different class (either numbers or str)
        # are held in different positions (ordered in number or alphabetic) and
        # the kernels will saved in that very order. 
        self.uniqLabels = list(set(labels)) 
        for idx,lab in enumerate(self.uniqLabels):
            if self.method is 'mvnormal':
                kernel = MVNormalEstimator()
                kernel.fit(self.SCORE[labels==lab,:self.n_comp]) 
                self.kernels.append(kernel)
            else:
                self.kernels.append([])
                for el in self.SCORE[:,:self.n_comp].T: # per column
                    self.kernels[idx].append(self.estimateDensity(el[labels==lab]))
        return self.kernels, self.uniqLabels
            
            
    def fit(self, stack, labels):
        """
        Performs the fitting of the model, in order to draw samples afterwards
        Inputs:
            stack -> stack comprising the whole database to be decomposed
            labels -> list (or array) containing the labels of each subject
        """
        labels = labels.astype(int)
#        selection = np.array([x in self.classes for x in labels])
#        stack_fin = stack[selection,:]
#        labels_fin = labels[selection]
        self.decompose(stack, labels)
        self.model(labels)
        
    def is_fitted(self):
        """ 
        Returns true if the model has been fitted and is ready for use. 
        """
        checkVar = True
        if self.kernels is None:
            checkVar = False
            
        return checkVar
    
    def createNewBrains(self, N, kernel, n_comp=None):
        """
        Creates new samples from the model. 
        """
        import warnings
        if n_comp is None:
            n_comp = self.n_comp
        elif isinstance(n_comp, numbers.Number):
            if n_comp> self.n_comp:
                warnings.warn("The model used less components than specified. Using default n_comp="+str(self.n_comp))
                n_comp = self.n_comp
        else: 
            raise ValueError('n_comp should be a number or None')  
        if not isinstance(kernel, list):
            newS = kernel.sample(N)
        else:
            newS = np.zeros((int(N), n_comp))
            for i in range(n_comp):
                k = kernel[i]
                newS[:,i] = k.sample(N).flatten()
        simStack = np.dot(newS[:,:n_comp], self.COEFF[:n_comp,:])
        if self.VAR is not None:
            simStack = simStack*self.VAR
        simStack = simStack + self.MEAN
        return simStack   

    def sample(self, N, clas=0, n_comp=None):
        """
        Draw samples from the model. 
        Inputs:
            N -> number of subjects to be generated
            clas -> model class ID
        """
        if(self.verbose):
            print('Creating brains with class %d'%clas)
        stackaux = self.createNewBrains(N, self.kernels[self.uniqLabels.index(clas)], n_comp)
        labelsaux = np.array([int(clas)]*N)
        return labelsaux, stackaux

    
    def generateDataset(self, stack, labels, N=100, classes=None, n_comp=None):
        """
        Fits the model and generates a new set of N elements for each class
        specified in "classes". 
        Inputs:
            stack -> the stack from which the model will be created
            labels -> the labels of the stacked dataset
            N -> the number of elements (per class) to be generated
            classes -> the classes we want to generate
            n_comp -> If we choose to change the number of components used in the synthesis
        """
        # If the model has not been fitted, fit it. 
        if not self.is_fitted():
            if self.verbose:
                print('Fitting the model')
            self.fit(stack, labels)
        # Classes input must correspond to the same numbers as labels
        if classes==None:
            clasdef = self.uniqLabels
        else:
            if (isinstance(classes[0], numbers.Number) and isinstance(self.uniqLabels[0], numbers.Number)) or (type(classes[0]) is type(self.uniqLabels[0])):
#                self.classes = []
                clasdef = classes
#                for el in classes:
#                    if el in self.uniqLabels:
#                        clasdef.append(self.uniqLabels.index(el))
#                    else:
#                        print('Error: specified class has not been modeled')
            else:
                print('Error: class not correctly specified')
        for ix, clas in enumerate(clasdef):
            if type(N) is list:
                labelsaux, stackaux = self.sample(N[ix], clas, n_comp)
            else:
                labelsaux, stackaux = self.sample(N, clas, n_comp)
            if 'finStack' not in locals():
                labels, finStack = labelsaux, stackaux
            else:
                finStack = np.vstack((finStack, stackaux))
                labels = np.hstack((labels, labelsaux))
            finStack[finStack<0]=0.
        return labels, finStack


