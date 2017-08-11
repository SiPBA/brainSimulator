# brainSimulator
Brain synthesis using KDE or MVN distribution. Currently in alpha state. Python code. 

## Use
With the new version, the whole interface has been switched to an object. This allows to train the model once and then perform as many sample drawings as required. 
```python 
#navigate to the folder where simulator.py is located
import brainSimulator as sim

simulator = sim.BrainSimulator(algorithm='PCA', method='mvnormal')
simulator.fit(original_dataset, labels) 
images, classes = simulator.generateDataset(original_dataset, labels, N=200, classes=[0, 1, 2])
```

## License
This code is released under the license [GPL-3.0+](https://choosealicense.com/licenses/gpl-3.0/). 
