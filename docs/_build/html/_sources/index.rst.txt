.. brainSimulator documentation master file, created by
   sphinx-quickstart on Fri Nov 10 09:01:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to brainSimulator's documentation!
==========================================
Functional brain image synthesis using the KDE or MVN distribution. Currently in beta. Python code.

`brainSimulator` is a brain image synthesis procedure intended to generate a new image set that share characteristics with an original one. The system focuses on nuclear imaging modalities such as PET or SPECT brain images. It analyses the dataset by applying PCA to the original dataset, and then model the distribution of samples in the projected eigenbrain space using a Probability Density Function (PDF) estimator. Once the model has been built, anyone can generate new coordinates on the eigenbrain space belonging to the same class, which can be then projected back to the image space.

.. toctree::
   overview
   api
   license
..   training
..   synthesizing

Cite
----------------

F.J. Martinez-Murcia et al (2017). "Functional Brain Imaging Synthesis Based on Image Decomposition and Kernel Modelling: Application to Neurodegenerative Diseases." Frontiers in neuroinformatics (online). DOI: `10.3389/fninf.2017.00065 <http://doi.org/10.3389/fninf.2017.00065>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
