"""Utilities related to model visualization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras
from keras.models import Model
from keras.layers.wrappers import Wrapper

DOT_KWARGS = {"graph_type": "graph",
              "rotate": 90,
              "ranksep": 0.4}

NODE_KWARGS = {"height": 0.5,
               "width": 1.4,
               "shape": "box",
               "style": "rounded, filled",
               "fontsize": 12,
               "fontcolor": "white",
               "fontname": "Roboto Light"}


LAYER_COLOR_DICT = {
    keras.engine.input_layer.InputLayer: "grey",
    keras.layers.core.Reshape: "#F5A286",
    keras.layers.convolutional.Conv1D: "#F7D7A8",
    keras.layers.convolutional.Conv2D: "#F7D7A8",
    keras.layers.pooling.MaxPooling1D: "#AADFA2",
    keras.layers.pooling.MaxPooling2D: "#AADFA2",
    keras.layers.convolutional.ZeroPadding2D: "grey",
    keras.layers.convolutional.ZeroPadding3D: "grey",
    keras.layers.core.Flatten: "#d44ddb",
    keras.layers.pooling.AveragePooling2D: "#A8CFE7",
    keras.layers.pooling.GlobalAveragePooling2D: "#A8CFE7",
    keras.layers.core.Dropout: "#9896C8",
    keras.layers.core.Dense: "#C66AA7",
    keras.layers.advanced_activations.ReLU: "#C66AA7",
    keras.layers.merge.Concatenate: "#F5A286",
    keras.engine.training.Model: "#292D30",
    keras.layers.core.RepeatVector: "grey",
    keras.layers.merge.Multiply: "grey",
    keras.layers.merge.Add: "grey",
    keras.layers.normalization.BatchNormalization: "#add8e6y",
    keras.layers.recurrent.LSTM: "#A8CFE7",
    keras.layers.recurrent.GRU: "#ff6961",
    keras.layers.core.Activation: "#9896C8",
}

# `pydot` is an optional dependency,
# see `extras_require` in `setup.py`.
try:
    import pydot
except ImportError:
    pydot = None


def _check_pydot():
    """Raise errors if `pydot` or GraphViz unavailable."""
    if pydot is None:
        raise ImportError(
            'Failed to import `pydot`. '
            'Please install `pydot`. '
            'For example with `pip install pydot`.')
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except OSError:
        raise OSError(
            '`pydot` failed to call GraphViz.'
            'Please install GraphViz (https://www.graphviz.org/) '
            'and ensure that its executables are in the $PATH.')


def is_model(layer):
    return isinstance(layer, Model)


def is_wrapped_model(layer):
    return isinstance(layer, Wrapper) and isinstance(layer.layer, Model)


def add_edge(dot, src, dst):
    if not dot.get_edge(src, dst):
        dot.add_edge(pydot.Edge(src, dst))


def model_to_dot(model,
                 show_shapes=False,
                 show_layer_names=True,
                 rankdir='TB',
                 expand_nested=False,
                 dpi=96,
                 subgraph=False):
    """Convert a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
        expand_nested: whether to expand nested models into clusters.
        dpi: dot DPI.
        subgraph: whether to return a pydot.Cluster instance.

    # Returns
        A `pydot.Dot` instance representing the Keras model or
        a `pydot.Cluster` instance representing nested model if
        `subgraph=True`.
    """
    from keras.layers.wrappers import Wrapper
    from keras.models import Model
    from keras.models import Sequential

    _check_pydot()
    if subgraph:
        dot = pydot.Cluster(style='dashed', graph_name=model.name)
        dot.set('label', model.name)
        dot.set('labeljust', 'l')
    else:
        dot = pydot.Dot(**DOT_KWARGS)
        # dot.set('rankdir', rankdir)
        dot.set('concentrate', True)
        dot.set('dpi', dpi)
        dot.set_node_defaults(shape='record')

    sub_n_first_node = {}
    sub_n_last_node = {}
    sub_w_first_node = {}
    sub_w_last_node = {}

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
    layers = model._layers

    # Create graph nodes.
    for i, layer in enumerate(layers):
        layer_id = str(id(layer))
        layer_type = type(layer)

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__

        if isinstance(layer, Wrapper):
            if expand_nested and isinstance(layer.layer, Model):
                submodel_wrapper = model_to_dot(layer.layer, show_shapes,
                                                show_layer_names, rankdir,
                                                expand_nested,
                                                subgraph=True)
                # sub_w : submodel_wrapper
                sub_w_nodes = submodel_wrapper.get_nodes()
                sub_w_first_node[layer.layer.name] = sub_w_nodes[0]
                sub_w_last_node[layer.layer.name] = sub_w_nodes[-1]
                dot.add_subgraph(submodel_wrapper)
            else:
                layer_name = '{}({})'.format(layer_name, layer.layer.name)
                child_class_name = layer.layer.__class__.__name__
                class_name = '{}({})'.format(class_name, child_class_name)

        if expand_nested and isinstance(layer, Model):
            submodel_not_wrapper = model_to_dot(layer, show_shapes,
                                                show_layer_names, rankdir,
                                                expand_nested,
                                                subgraph=True)
            # sub_n : submodel_not_wrapper
            sub_n_nodes = submodel_not_wrapper.get_nodes()
            sub_n_first_node[layer.name] = sub_n_nodes[0]
            sub_n_last_node[layer.name] = sub_n_nodes[-1]
            dot.add_subgraph(submodel_not_wrapper)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label,
                                                           inputlabels,
                                                           outputlabels)

        if not expand_nested or not isinstance(layer, Model):
            node = pydot.Node(layer_id,
                              label=label.split(":")[-1],
                              color=LAYER_COLOR_DICT.get(layer_type, 'grey'),
                              **NODE_KWARGS)
            dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    if not expand_nested:
                        assert dot.get_node(inbound_layer_id)
                        assert dot.get_node(layer_id)
                        dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
                    else:
                        # if inbound_layer is not Model or wrapped Model
                        if not is_model(inbound_layer) and (
                                not is_wrapped_model(inbound_layer)):
                            # if current layer is not Model or wrapped Model
                            if not is_model(layer) and (
                                    not is_wrapped_model(layer)):
                                assert dot.get_node(inbound_layer_id)
                                assert dot.get_node(layer_id)
                                dot.add_edge(pydot.Edge(inbound_layer_id,
                                                        layer_id))
                            # if current layer is Model
                            elif is_model(layer):
                                add_edge(dot, inbound_layer_id,
                                         sub_n_first_node[layer.name].get_name())
                            # if current layer is wrapped Model
                            elif is_wrapped_model(layer):
                                dot.add_edge(pydot.Edge(inbound_layer_id,
                                                        layer_id))
                                name = sub_w_first_node[layer.layer.name].get_name()
                                dot.add_edge(pydot.Edge(layer_id,
                                                        name))
                        # if inbound_layer is Model
                        elif is_model(inbound_layer):
                            name = sub_n_last_node[inbound_layer.name].get_name()
                            if is_model(layer):
                                output_name = sub_n_first_node[layer.name].get_name()
                                add_edge(dot, name, output_name)
                            else:
                                add_edge(dot, name, layer_id)
                        # if inbound_layer is wrapped Model
                        elif is_wrapped_model(inbound_layer):
                            inbound_layer_name = inbound_layer.layer.name
                            add_edge(dot,
                                     sub_w_last_node[inbound_layer_name].get_name(),
                                     layer_id)
    return dot


def plot_model(model,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB',
               expand_nested=False,
               dpi=96):
    """Converts a Keras model to dot format and save to a file.

    # Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
        expand_nested: whether to expand nested models into clusters.
        dpi: dot DPI.

    # Returns
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
    """
    dot = model_to_dot(model, show_shapes, show_layer_names, rankdir,
                       expand_nested, dpi)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
    # Return the image as a Jupyter Image object, to be displayed in-line.
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except ImportError:
        pass
