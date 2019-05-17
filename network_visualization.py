""" Module for drawing Keras netowork graph architectures """

import glob
import numpy as np
from itertools import compress
import pydot
import graphviz
import pickle
import keras
import matplotlib.pyplot as plt
from IPython.display import Image, display


def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)


layer_color_dict = {
    keras.engine.input_layer.InputLayer: "grey",
    keras.layers.core.Reshape: "#F5A286",
    keras.layers.convolutional.Conv1D: "#F7D7A8",
    keras.layers.convolutional.Conv2D: "#F7D7A8",
    keras.layers.pooling.MaxPooling1D: "#AADFA2",
    keras.layers.pooling.MaxPooling2D: "#AADFA2",
    keras.layers.convolutional.ZeroPadding3D: "grey",
    keras.layers.core.Flatten: "grey",
    keras.layers.pooling.AveragePooling2D: "#A8CFE7",
    keras.layers.pooling.GlobalAveragePooling2D: "#A8CFE7",
    keras.layers.core.Dropout: "#9896C8",
    keras.layers.core.Dense: "#C66AA7",
    keras.layers.merge.Concatenate: "#F5A286",
    keras.engine.training.Model: "#292D30",
    keras.layers.core.RepeatVector: "grey",
    keras.layers.merge.Multiply: "grey",
    keras.layers.merge.Add: "grey",
    keras.layers.normalization.BatchNormalization: "grey",
    keras.layers.core.Activation: "grey"
}


def draw_graph(keras_model: keras.engine.training.Model,
               title: str = "Model Visualization",
               layer_color_dict: dict = layer_color_dict):
    """
    Draw graphviz diagram of the network architecture.

    Inspired by the Inception-ResNet diagrams from
    https://ai.googleblog.com/2016/08/improving-inception-and-image.html

    args
      keras_model: a module or blueprint
      layer_color_dict: a dictionary of keras layer types and the
                        respective color to draw them as.
    """

    graph = pydot.Dot(graph_type='graph',
                      label=title,
                      fontsize=16,
                      fontname='Roboto Light',
                      rankdir="LR",
                      ranksep=0.4)

    # unfortunately layer names and inbound names are not exactly the same..
    layer_names = [layer.name for layer in keras_model.layers]
    print("Initializing layer graph nodes ..")
    for layer in tqdm.tqdm(keras_model.layers):

        layer_name = layer.name
        layer_type = type(layer)

        node = pydot.Node(layer_name,
                          orientation=0,
                          height=1.4,
                          width=0.5,
                          fontsize=0.01,
                          shape='box',
                          style='rounded, filled',
                          color=layer_color_dict[layer_type],
                          fontcolor=layer_color_dict[layer_type])

        graph.add_node(node)
        inbound_layer = layer.get_input_at(0)

        if layer_name.startswith("input"):
            pass

        else:
            if type(inbound_layer) is list:
                inbound_names = [inbound.name for inbound in inbound_layer]
            else:
                inbound_names = [inbound_layer.name]

            for inbound_name in inbound_names:
                # temporary fix for issues related to input
                name_in_layers = [inbound_name.startswith(layer_name) for
                                  layer_name in layer_names]

                inbound_layer_names = list(compress(layer_names,
                                                    name_in_layers))

                for inbound_layer_name in inbound_layer_names:
                    inbound_node = pydot.Node(inbound_layer_name)

                    graph.add_node(inbound_node)
                    graph.add_edge(pydot.Edge(inbound_node, node,
                                              color='grey'))

    view_pydot(graph)


def draw_legend(layer_color_dict: dict = layer_color_dict):

    for i in np.arange(0, len(layer_color_dict.keys()), 4):
        graph = pydot.Dot(graph_type='graph', ranksep=0.1)

        for layer_type in list(layer_color_dict.keys())[i:i+4]:
            layer_name = str(layer_type).split("'>")[0].split(".")[-1]
            node = pydot.Node(layer_name,
                              height=0.5,
                              width=2,
                              fontsize=14,
                              shape='box',
                              style='rounded, filled',
                              color=layer_color_dict[layer_type],
                              fontcolor='white',
                              fontname='Roboto Light')

            graph.add_node(node)

        view_pydot(graph)
        del graph
