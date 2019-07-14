# Keras Visualization Utils 2

## Introduction
Neural Network visualization is a key part of communicating novel architectures, whether in the academic or research space. Commonly used tools like ```keras.utils.plot_model``` do well to help in presenting model complexity and in debugging purposes, but fail to communicate architectures effectively. Alternatively, Tensorboard offers a semantically more pleasing presentation, but requires that tensors be loaded into memory - making visualization computationally expensive.

This repository extends the ```keras.utils.plot_model``` function to add semantically more pleasing representation.

Styles are inspired from [this Google AI blog](https://ai.googleblog.com/2016/08/improving-inception-and-image.html)

**TODO**
- Fork keras repository
- Find modules
- Consider parsing model_json instead

## Use
```python
from keras.applications.vgg16 import VGG16
from viz_utils2 import plot_model

vgg16 = VGG16()
draw_graph(inception)
```

## Examples
### VGG16
<img src='docs/vgg16.png'>

### ResNet50
<img src='docs/resnet50.png'>

### InceptionV3
<img src='docs/inceptionv3.png'>

### MobileNet
<img src='docs/mobilenet.png'>

### Inception ResNet
<img src='docs/inceptionresnet.png'>
