# Keras Network Visualization

## Introduction
Neural Network visualization is a key part of communicating novel architectures, whether in the academic or research space. Commonly used tools like ```keras.utils.plot_model``` do well to help in presenting model complexity and in debugging purposes, but fail to communicate architectures effectively. Alternatively, Tensorboard offers a semantically more pleasing presentation, but requires that tensors be loaded into memory - making visualization computationally expensive. This repository aims to combine the best of both by drawing comprehensive and semantically pleasing network graphs.

## Setup
```shell
pip install -r requirements.txt
python install setup.py 
pip install . 
```

## Use
```python
from keras.applications.inception_v3 import InceptionV3
from network_visualization import draw_graph

inception = InceptionV3()
draw_graph(inception)
```

## (TODO) Examples


