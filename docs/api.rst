brainSimulator's API
===================================
.. automodule:: brainSimulator

class brainSimulator
-----------------------------------
.. autoclass:: BrainSimulator
	:members: generateDataset, estimateDensity, model, decompose, createNewBrains, fit, sample

auxiliary classes
----------------------
These auxiliary classes define the PDF models that will be applied in the analysis and synthesis of brain images. All feature a set of methods `.fit` and `.sample` that in the case of MVN and Gaussian are a simple interface for their `scipy` counterparts, while for the KDE, it uses automatic estimation of bandwidth and defines more auxiliary functions. See a further discussion of these in the original paper. 

.. autoclass:: MVNormalEstimator

.. autoclass:: GaussianEstimator

.. autoclass:: KDEestimator
	:members: botev_bandwidth

auxiliary functions
--------------------------
.. autofunction:: applyPCA
.. autofunction:: applyICA
