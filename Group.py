#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import unique, IntEnum
import numpy as np
from ANNFramework.Activations import ActType
from ANNFramework.Layer import LayerFactory, LayerType
from ANNFramework.LearningRule import LearningType


@unique
class ConPattern(IntEnum):
    ONE_TO_ONE = 0
    DENSE = 1


class Group:
    """
    Defines a group of Layers, each independent of the others, but all having the same characteristics
    (size, activation function, ...).
    """

    def __init__(self, name, nb_layers=None, nb_neurons=None, layer_type=LayerType.HIDDEN, activation_type=ActType.TANH, norm=None):
        """
        Simply initialize the required number of layers, using the other given characteristics.
        :param name: A string representing the name of the group.
        :param nb_layers: An integer indicating the number of layers this group should contain.
        :param nb_neurons: An integer indicating the number of neurons for all layers. Or a list indicating different
        number of neurons for each layer separately. If None an empty group is created.
        :param layer_type: An instance of the LayerType enumeration, indicating the type of layer, the group contains.
        :param activation_type: An instance of the ActType enumeration, indicating the type of non-linearity the layers
        will be using.
        :param norm: A string indicating the type of norm to use. Currently available: 'frobenius' and 'malsburg'. A
        value of None means that no normalization should be applied.
        adapt.
        """

        # Initialize the groups attributes
        self._name = name
        self._size = nb_layers
        self._activation_type = activation_type
        self._norm = norm

        # Create a set of layers if there is at least one neuron per layer
        # See Brain.load for why this is
        if nb_layers is not None and nb_neurons is not None:
            # Transform number of neurons given in parameter into a list for each layer
            if not isinstance(nb_neurons, list):
                nb_neurons = [nb_neurons] * nb_layers

            self._layers = [LayerFactory.get_layer(layer_type, "{}@{}".format(name, idx), nn, activation_type,
                                                   norm) for idx, nn in enumerate(nb_neurons)]
        else:
            self._layers = []

    def add_input(self, pre_layer, con_type, weights=None, learning=True, learning_type=LearningType.NONE,
                  learning_rate=0.02, con_pattern=ConPattern.DENSE, start=0, end=None):
        """
        Add a single input to all the layers or only the one corresponding to the given index, if any.
        :param pre_layer: An instance of the Layer class, representing the input layer. Or a Group representing a list
        of pre-synpatic input layers.
        :param con_type: An instance of the ConType enumeration, representing the type of connection to instantiate.
        :param weights: A list of numpy array representing the synaptic strengths of the connections between the pre-
        and post-layers.
        :param learning: A boolean value indicating whether the connection's strength is modifiable or not.
        :param learning_type: A instance of the LearningType enumeration, indicating the type of learning rule that
        will be applied to update the connection's synaptic's strength.
        :param learning_rate: A float representing the rate at which the connection's strength is able to evolve.
        :param con_pattern: An instance of the ConPattern enumeration, representing the connection pattern between the
        pre- and post-synaptic Group. This parameter only makes sense when connecting Groups of Layers.
        :param start: An integer indicating the index at which to begin the slice.
        :param end: An integer representing the index at which to stop the slice.
        :return: A list of Connections established between the different layers.
        """

        # Initialize a list of connections
        connections = []

        if not isinstance(weights, list):
            weights = [weights for _ in range(len(self._layers))]

        # The connection pattern parameter only makes sense when connecting Groups together
        if isinstance(pre_layer, Group):
            # Make sure the end of the slice points toward something
            if end is None:
                end = self._size

            # Connect the layers from both groups following the given pattern
            if con_pattern == ConPattern.ONE_TO_ONE:
                for weight, in_layer, out_layer in zip(weights, pre_layer.layers, self._layers[start:end]):
                    connections.extend(out_layer.add_input(in_layer, con_type, weights=weight, learning=learning,
                                                           learning_type=learning_type, learning_rate=learning_rate))

            elif con_pattern == ConPattern.DENSE:
                for weight, out_layer in zip(weights, self._layers[start:end]):
                    connections.extend(out_layer.add_input(pre_layer, con_type, weights=weight, learning=learning,
                                                           learning_type=learning_type, learning_rate=learning_rate))
        else:  # This should only apply to Layers
            # Make sure the end of the slice points toward something
            if end is None:
                end = self._size

            # Connect the pre_layer to all the layers contained within the slice
            for weight, post_layer in zip(weights, self._layers[start:end]):
                connections.extend(post_layer.add_input(pre_layer, con_type, weights=weight,
                                                        learning=learning, learning_type=learning_type,
                                                        learning_rate=learning_rate))

        # Return the created connections
        return connections

    def activate(self):
        """
        Update the activation values of all layers within the Group.
        :return: Nothing.
        """

        # Simply let the layers do their job
        for layer in self._layers:
            layer.activate()

    def save(self, grp):
        """
        Build a record of the current group and its underlying layers in the given HDF5 group.
        :param grp: An HDF5 group.
        :return: Nothing.
        """

        # Record the main attributes of the group
        grp.attrs['class'] = type(self).__name__
        grp.attrs['name'] = self._name
        grp.attrs['size'] = self._size
        if self._norm is not None:
            grp.attrs['norm'] = self._norm
        grp.attrs['activation_type'] = self._activation_type

        # For each layer in the group
        for idx, layer in enumerate(self._layers):
            # Create a new group
            layer_grp = grp.create_group("{}_{}".format(self._name, idx))
            # Let the layer do its thing
            layer.save(layer_grp)

    def normalize_weights(self):
        """
        Normalize the weight matrices of each of the sub-layers in turn using one of the available norms.
        :return: Nothing.
        """

        # Let each layer normalize its own weights
        for layer in self._layers:
            layer.normalize_weights()

    def find_layer_by_name(self, layer_name):
        """
        Find and return the Layer instance corresponding to the given layer_name, if present in this group.
        :param layer_name: A string representing the name given to the Layer's instance.
        :return: An instance a Layer's sub-class, if any found. None, otherwise.
        """

        # Look for the layer in the list
        for layer in self._layers:
            if layer.name == layer_name:
                return layer

        # If none has been found, return None
        return None

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return self._name

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, layers):
        self._layers = layers
        self._size = len(layers)

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, val):
        # Set the value of the Group's norm
        if isinstance(val, str):
            self._norm = val.lower()
        else:
            self._norm = val

        # Broadcast the change to all the sub-layers
        # No need to perform any checks here since the layer.norm setter already does it for us
        for layer in self._layers:
            layer.norm = val

    @property
    def output(self):
        # Return all outputs stacked next to each others
        return np.stack([layer.output for layer in self._layers], axis=0)

    @property
    def weights(self):
        return np.stack([layer.weights for layer in self._layers], axis=0)
