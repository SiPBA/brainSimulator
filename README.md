# brainSimulator
[![DOI](https://zenodo.org/badge/85931767.svg)](https://zenodo.org/badge/latestdoi/85931767)

Functional brain image synthesis using the KDE or MVN distribution. Currently in beta. Python code. Find the documentation at http://brainsimulator.readthedocs.io/

`brainSimulator` is a brain image synthesis procedure intended to generate a new image set that share characteristics with an original one. The system focuses on nuclear imaging modalities such as PET or SPECT brain images. It analyses the dataset by applying PCA to the original dataset, and then model the distribution of samples in the projected eigenbrain space using a Probability Density Function (PDF) estimator. Once the model has been built, anyone can generate new coordinates on the eigenbrain space belonging to the same class, which can be then projected back to the image space.

## Use
First of all, install the package via pypi with:
```
pip install brainsimulator
```

With the new version, the whole interface has been switched to an object. This allows to train the model once and then perform as many sample drawings as required. 
```python 
#navigate to the folder where simulator.py is located
import brainSimulator as sim

simulator = sim.BrainSimulator(algorithm='PCA', method='mvnormal')
simulator.fit(original_dataset, labels) 
images, classes = simulator.generateDataset(original_dataset, labels, N=200, classes=[0, 1, 2])
```

## Cite
F.J. Martinez-Murcia et al (2017). "Functional Brain Imaging Synthesis Based on Image Decomposition and Kernel Modelling: Application to Neurodegenerative Diseases." Frontiers in neuroinformatics (online). DOI: 10.3389/fninf.2017.00065

## Safeguards
As in the paper, it is best to use MVN modelling, but it is fundamental to test the number of components (L) used in the modelling, otherwise it will lead to overfitting. The KDE modelling works better `out of the box', but the results may be more disperse. 

## License
This code is released under the license [GPL-3.0+](https://choosealicense.com/licenses/gpl-3.0/). 
