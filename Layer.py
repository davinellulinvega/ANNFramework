#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import IntEnum, unique
from collections import defaultdict
from itertools import chain
import numpy as np
from ANNFramework.Activations import ActType, ActsFactory
from ANNFramework.Connection import ConType, ConnectionFactory
from ANNFramework.LearningRule import LearningType


@unique
class LayerType(IntEnum):
    GROUP = -1
    INPUT = 0
    HIDDEN = 1


class LayerFactory:
    """
    A factory that builds Layers depending on the given LayerType.
    """

    @classmethod
    def get_layer(cls, layer_type, name, size, activation_type=ActType.NONE, norm=None):
        """
        Given an instance of the LayerType enumeration, return an initialized instance of the corresponding Layer
        sub-class.
        :param layer_type: An instance of the LayerType enumeration, indicating the type of Layer to build.
        :param name: A string defining the name of the layer.
        :param size: An integer representing the number of neurons contained in the layer.
        :param activation_type: An instance of the ActType enumeration, representing the non-linearity applied to each
        neuron's output.
        :param norm: A string indicating the type of norm to use. Currently available: 'frobenius' and 'malsburg'. A
        value of None means that no normalization should be applied.
        :return: An instance of a Layer's sub-class.
        """

        if layer_type == LayerType.INPUT:
            return InputLayer(name, size)
        elif layer_type == LayerType.HIDDEN:
            return HiddenLayer(name, size, activation_type=activation_type, norm=norm)

    @classmethod
    def get_from_str(cls, layer_class, name, size, activation_type=ActType.NONE, norm=None):
        """
        Given the string representation of a Layer sub-class, return an initialized instance of the corresponding Layer
        sub-class.
        :param layer_class: A string representing the name of the class to instantiate.
        :param name: A string defining the name of the layer.
        :param size: An integer representing the number of neurons contained in the layer.
        :param activation_type: An instance of the ActType enumeration, representing the non-linearity applied to each
        neuron's output.
        :param norm: A string indicating the type of norm to use. Currently available: 'frobenius' and 'malsburg'. A
        value of None means that no normalization should be applied.
        :return: An instance of a Layer's sub-class.
        """

        if layer_class == 'InputLayer':
            return InputLayer(name, size)
        elif layer_class == 'HiddenLayer':
            return HiddenLayer(name, size, activation_type=activation_type, norm=norm)


class Layer:
    """
    Defines a generic layer with a name, a set of inputs, neurons and weights connecting both. This container does not
    do anything on its own. It is better to use one of its child implementation.
    """

    def __init__(self, name, size, activation_type=ActType.NONE, norm=None):
        """
        Initialize the main attributes of a layer.
        :param name: A string defining the name of the layer.
        :param size: An integer representing the number of neurons contained in the layer.
        :param activation_type: An instance of the ActType enumeration, representing the non-linearity applied to each
        neuron's output.
        :param norm: A string indicating the type of norm to use. Currently available: 'frobenius' and 'malsburg'. A
        value of None means that no normalization should be applied.
        """

        self._name = name
        self._size = size
        self._output = np.zeros((size, 1), dtype=np.float32)
        self._activation_type = activation_type
        self._activation_function = ActsFactory.get_activation(activation_type)
        self._connections = defaultdict(list)  # Store connections according to their ConType
        if isinstance(norm, str):
            self._norm = norm.lower()
        else:
            self._norm = norm

    def frobenius(self):
        """
        Normalize in-place the connections' weight matrices using the Frobenius norm.
        :return: Nothing.
        """

        # Extract the modifiable connections
        weights = [conn.weight for conn in chain(*self._connections.values()) if conn.learning]

        # Compute the Frobenius norm of the complete layer
        # TODO: Make sure this norm is correct and has the right shape (self._size, 1)
        frob_norm = np.linalg.norm(np.hstack(weights), axis=1).reshape(self._size, 1)

        # Normalize all the connections
        for weight in weights:
            weight /= frob_norm

    def malsburg(self):
        """
        Normalize in-place the connections' weight matrices using the algorithm described in (Malsburg, 1973).
        :return: Nothing.
        """

        # Concatenate all the weight matrices into one matrix
        weights = [conn.weight for conn in chain(*self._connections.values()) if conn.learning]
        stack = np.hstack(weights)

        # Compute the average synaptic strength for the whole layer
        avg_strength = np.mean(stack)

        # Compute the sum total of the input synaptic strength for each neuron
        sum_strength = np.sum(stack, axis=1).reshape(self._size, 1)

        # Normalize the synaptic strength for each connection
        for weight in weights:
            weight *= avg_strength / sum_strength

    def save(self, grp):
        """
        Build a record of the current layer in the given HDF5 group.
        :param grp: An HDF5 group.
        :return: Nothing.
        """

        # Record the main attributes of the layer
        grp.attrs['class'] = type(self).__name__
        grp.attrs['name'] = self._name
        grp.attrs['size'] = self._size
        grp.attrs['activation_type'] = self._activation_type
        if self._norm is not None:
            grp.attrs['norm'] = self._norm
        for idx, con in enumerate(chain(*self._connections.values())):
            # Create a new sub-group for the connection
            sub_grp = grp.create_group("con@{}".format(idx))
            # Let the connection do its thing
            con.save(sub_grp)

    def activate(self):
        raise NotImplementedError("You should never instantiate a Layer, but rather sub-class it and override the "
                                  "activate() method.")

    def normalize_weights(self):
        raise NotImplementedError("You should never instantiate a Layer, but rather sub-class it and override the "
                                  "normalize_weights() method.")

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return self._name

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, vals):
        if not isinstance(vals, np.ndarray):
            vals = np.array(vals, dtype=np.float32)
        # Make sure the array of values provided are of the right shape
        assert vals.shape == (self._size, 1), "Error: Cannot feed an array of shape: {}, " \
                                              "into a layer of shape: {}".format(vals.shape, (self._size, 1))
        # After all this, finally set the output
        self._output = vals

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, val):
        if isinstance(val, str):
            self._norm = val.lower()
        else:
            self._norm = val

    @property
    def connections(self):
        return list(chain(*self._connections.values()))


class HiddenLayer(Layer):
    """
    Defines a hidden layer in the sense of the classic feed forward neural network.
    """

    def __init__(self, name, size, activation_type=ActType.TANH, norm=None):
        """
        Initialize all the required attributes to perform the role of a simple layer.
        :param name: A string defining the name of the layer.
        :param size: An integer representing the number of neurons contained in the layer.
        :param activation_type: An instance of the ActType enumeration, representing the non-linearity applied to each
        neuron's output.
        :param norm: A string indicating the type of norm to use. Currently available: 'frobenius' and 'malsburg'. A
        value of None means that no normalization should be applied.
        """

        # Initialize the parent class
        super(HiddenLayer, self).__init__(name, size, activation_type=activation_type, norm=norm)

    def add_input(self, pre_layer, con_type, weights=None, learning=True, learning_type=LearningType.NONE,
                  learning_rate=0.02):
        """
        Add the given layer as an input with the given type of connection.
        :param pre_layer: A Layer or GroupLayer, whose outputs serve as inputs to this layer.
        :param con_type: The type of connection between the two layers.
        :param weights: A list of numpy array representing the synaptic strengths of the connections between the pre-
        and post-layers.
        :param learning: A boolean indicating whether the connection is modifiable or not.
        :param learning_type: A instance of the LearningType enumeration, indicating the type of learning rule that
        will be applied to update the connection's synaptic's strength.
        :param learning_rate: A float representing the rate at which the connection's strength is able to evolve.
        :return: A list containing the connections created between pre- and post-synaptic layer.
        """

        # Encompasses any kind of Layer object
        if isinstance(pre_layer, Layer):
            layers = [pre_layer]
        else:  # Targets Groups only
            layers = pre_layer.layers

        if not isinstance(weights, list):
            weights = [weights for _ in range(len(layers))]

        # Initialize a list of connections to return
        connections = []
        for weight, layer in zip(weights, layers):
            # Create a new connection between the pre- and post-synaptic layers
            conn = ConnectionFactory.get_connection(con_type, layer, self, weight=weight, learning=learning,
                                                    learning_type=learning_type, learning_rate=learning_rate)

            # Store the connection within the layer itself and the list to be returned
            self._connections[con_type].append(conn)
            connections.append(conn)

        # Return the list of connections
        return connections

    def activate(self):
        """
        Compute the neurons' activation values.
        :return: Nothing.
        """

        # Reset the output's activation values
        self._output = np.zeros((self._size, 1))

        # Get the activity of all non-modulatory connections
        for con_type, connections in self._connections.items():
            # Avoid those connection for the moment, since they are computed later on
            if con_type in [ConType.MODULATION_P, ConType.MODULATION_N, ConType.LAT_INHIB]:
                continue
            # Get the activity from all the other connections
            for conn in connections:
                self._output += conn.activity

        # Onward with the modulation
        for conn in chain(self._connections[ConType.MODULATION_N], self._connections[ConType.MODULATION_P]):
            self._output *= conn.activity

        # Some lateral inhibition?
        for conn in self._connections[ConType.LAT_INHIB]:
            self._output += conn.activity

        # Finally some non-linearity on top
        self._output = self._activation_function(self._output)

    def normalize_weights(self):
        """
        Apply one of the available normalization algorithm to the weights of the connections.
        :return: Nothing.
        """

        # Just make sure the layer is supposed to be normalized
        if self._norm is not None:
            if isinstance(self._norm, str):
                # Simply call the required norm
                getattr(self, self._norm)()
            else:
                self._norm()

    @property
    def weights(self):
        return np.stack([conn.weight for conn in chain(*self._connections.values())], axis=0)


class InputLayer(Layer):
    """
    Define a specific type of layers, that do not learn and have a fixed output, externally set.
    """

    def __init__(self, name, size):
        """
        Initialize the parent class with standard parameters and override some of its attributes.
        :param size: An integer indicating the number of input neurons to build.
        """

        # Initialize the parent class
        super(InputLayer, self).__init__(name, size, activation_type=ActType.NONE, norm=None)

    def activate(self):
        """
        Does nothing, since this is an input layer.
        :return: Nothing.
        """

        pass
