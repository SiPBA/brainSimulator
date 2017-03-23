# brainSimulator
Brain simulation using KDE or MVN distribution. Currently in alpha state. Python code. 

## Use
```python 
#navigate to the folder where simulator.py is located
import brainSimulator as sim

images, classes = sim.generateDataset(original_dataset, labels, N=200, classes=[0, 1, 2], algorithm='PCA', method='mvnormal')
```
