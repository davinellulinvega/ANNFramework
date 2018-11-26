#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import IntEnum, unique
import numpy as np
from ANNFramework.LearningRule import LearningType


@unique
class ConType(IntEnum):
    INPUT_P = 0
    INPUT_N = 1
    EXCITATORY = 2
    INHIBITORY = 3
    MODULATION_P = 4
    MODULATION_N = 5
    RECURRENT_P = 6
    RECURRENT_N = 7
    REC_MOD_P = 8
    REC_MOD_N = 9
    BIAS_P = 10
    BIAS_N = 11
    REC_BIAS_P = 12
    REC_BIAS_N = 13
    LAT_INHIB = 14


class ConnectionFactory:
    """
    A factory that builds connections depending on the given ConType.
    """

    @classmethod
    def get_connection(cls, con_type, pre_layer, post_layer=None, weight=None, learning=True,
                       learning_type=LearningType.NONE, learning_rate=1e-7):
        """
        Given the connection type, return an instance of the right Connection class.
        :param con_type: An instance of the ConType enumeration, representing the type of connection to instantiate.
        :param pre_layer: An instance of a Layer's sub-class, representing the pre-synaptic input layer.
        :param post_layer: An instance of a Layer's sub-class, representing the pos-synaptic output layer.
        :param weight: A numpy array representing the synaptic strength of the connections between the pre- and
        post-layers.
        :param learning: A Boolean indicating whether the synaptic strength is adaptable or not.
        :param learning_type: A instance of the LearningType enumeration, indicating the type of learning rule that
        will be applied to update the connection's synaptic's strength.
        :param learning_rate: A float representing the rate at which the connection's strength is able to adapt.
        :return: An instance of a Connection's child class.
        """

        if con_type == ConType.INPUT_P:
            return InputConnection(pre_layer, post_layer, weight=weight, learning=learning, learning_type=learning_type,
                                   learning_rate=learning_rate, inhibitory=False)
        elif con_type == ConType.INPUT_N:
            return InputConnection(pre_layer, post_layer, weight=weight, learning=learning, learning_type=learning_type,
                                   learning_rate=learning_rate, inhibitory=True)
        elif con_type == ConType.EXCITATORY:
            return StdConnection(pre_layer, post_layer, weight=weight, learning=learning, learning_type=learning_type,
                                 learning_rate=learning_rate, inhibitory=False, recurrent=False)
        elif con_type == ConType.INHIBITORY:
            return StdConnection(pre_layer, post_layer, weight=weight, learning=learning, learning_type=learning_type,
                                 learning_rate=learning_rate, inhibitory=True, recurrent=False)
        elif con_type == ConType.MODULATION_P:
            return ModulationConnection(pre_layer, post_layer, inhibitory=False, recurrent=False)
        elif con_type == ConType.MODULATION_N:
            return ModulationConnection(pre_layer, post_layer, inhibitory=True, recurrent=False)
        elif con_type == ConType.RECURRENT_P:
            return StdConnection(pre_layer, post_layer, weight=weight, learning=learning, learning_type=learning_type,
                                 learning_rate=learning_rate, inhibitory=False, recurrent=True)
        elif con_type == ConType.RECURRENT_N:
            return StdConnection(pre_layer, post_layer, weight=weight, learning=learning, learning_type=learning_type,
                                 learning_rate=learning_rate, inhibitory=True, recurrent=True)
        elif con_type == ConType.REC_MOD_P:
            return ModulationConnection(pre_layer, post_layer, inhibitory=False, recurrent=True)
        elif con_type == ConType.REC_MOD_N:
            return ModulationConnection(pre_layer, post_layer, inhibitory=True, recurrent=True)
        elif con_type == ConType.BIAS_P:
            return BiasConnection(pre_layer, post_layer, inhibitory=False, recurrent=False)
        elif con_type == ConType.BIAS_N:
            return BiasConnection(pre_layer, post_layer, inhibitory=True, recurrent=False)
        elif con_type == ConType.REC_BIAS_P:
            return BiasConnection(pre_layer, post_layer, inhibitory=False, recurrent=True)
        elif con_type == ConType.REC_BIAS_N:
            return BiasConnection(pre_layer, post_layer, inhibitory=True, recurrent=True)
        elif con_type == ConType.LAT_INHIB:
            return LateralConnection(pre_layer, post_layer=post_layer, weight=weight, learning=learning,
                                     learning_type=learning_type, learning_rate=learning_rate)

    @classmethod
    def get_from_str(cls, con_str, pre_layer, post_layer=None, weight=None, learning=True,
                     learning_type=LearningType.NONE, learning_rate=1e-7, inhibitory=False, recurrent=False):
        """
        Given the connection type, return an instance of the right Connection class.
        :param con_str: A string representing the name of the sub-class to instantiate.
        :param pre_layer: An instance of a Layer's sub-class, representing the pre-synaptic input layer.
        :param post_layer: An instance of a Layer's sub-class, representing the pos-synaptic output layer.
        :param weight: A numpy array representing the synaptic strength of the connections between the pre- and
        post-layers.
        :param learning: A Boolean indicating whether the synaptic strength is adaptable or not.
        :param learning_type: A instance of the LearningType enumeration, indicating the type of learning rule that
        will be applied to update the connection's synaptic's strength.
        :param learning_rate: A float representing the rate at which the connection's strength is able to adapt.
        :param inhibitory: A Boolean indicating whether the connection is inhibitory or not.
        :param recurrent: A Boolean indicating whether the connection is recurrent or not.
        :return: An instance of a Connection's child class.
        """

        if con_str == "InputConnection":
            return InputConnection(pre_layer, post_layer, weight, learning, learning_type, learning_rate, inhibitory)
        elif con_str == "StdConnection":
            return StdConnection(pre_layer, post_layer, weight, learning, learning_type, learning_rate, inhibitory,
                                 recurrent)
        elif con_str == "ModulationConnection":
            return ModulationConnection(pre_layer, post_layer, inhibitory, recurrent)
        elif con_str == "BiasConnection":
            return BiasConnection(pre_layer, post_layer, inhibitory, recurrent)
        elif con_str == "LateralConnection":
            return LateralConnection(pre_layer, post_layer, weight, learning, learning_type, learning_rate)


class Connection:
    """
    Define a connection of a given type between a pre- and post-synaptic layer. A connection's strength is modifiable by
    default, but can be made static.
    """

    def __init__(self, pre_layer, post_layer, weight=None, learning=True, learning_type=LearningType.NONE,
                 learning_rate=1e-7, inhibitory=False, recurrent=False):
        """
        Initialize the connection's attributes, as well as its weight matrix.
        :param pre_layer: An instance of a Layer's sub-class, representing the pre-synaptic input layer.
        :param post_layer: An instance of a Layer's sub-class, representing the pos-synaptic output layer.
        :param weight: A numpy array representing the synaptic strength of the connections between the pre- and
        post-layers.
        :param learning: A Boolean indicating whether the synaptic strength is adaptable or not.
        :param learning_type: A instance of the LearningType enumeration, indicating the type of learning rule that
        will be applied to update the connection's synaptic's strength.
        :param learning_rate: A float representing the rate at which the connection's strength is able to adapt.
        :param inhibitory: A Boolean indicating whether the connection is inhibitory or not.
        :param recurrent: A Boolean indicating whether the connection is recurrent or not.
        """

        # Initialize the basic attributes required for a connection to do its job
        self._pre_layer = pre_layer
        self._post_layer = post_layer
        self._learning = learning
        self._learning_type = learning_type
        self._learning_rate = learning_rate
        self._inhibitory = inhibitory
        # Store the recurrent status if needed in the future. However, since the Brain/Network activates the layers in
        # a set order, Layer.output keeps the output value from t-1.
        self._recurrent = recurrent
        # Let any sub-class do the initialization of the weight
        if weight is None:
            self._weight = weight
        else:
            # Just to make sure the weight matrix is not simply a list of lists and the shape property is available
            if isinstance(weight, list):
                weight = np.array(weight, dtype=np.float32)
            # Make sure the weight matrix provided has the right shape
            assert weight.shape == (self._post_layer.size, self._pre_layer.size), \
                "Provided a weight matrix of shape: {}, while connection between {} and {} needs a matrix of " \
                "shape: {}".format(weight.shape, pre_layer.name, post_layer.name,
                                   (self._post_layer.size, self._pre_layer.size))

            # Assign the weight
            self._weight = weight

    def __hash__(self):
        """
        Define a unique hash for the connection, based on the name of the pre- and post-synaptic layers.
        :return: An integer representing the hash corresponding to this connection.
        """

        return hash(self._post_layer.name) + hash(self._pre_layer.name)

    def save(self, grp):
        """
        Build a record of the current connection in the given HDF5 group.
        :param grp: A HDF5 group.
        :return: Nothing.
        """

        # Record the connection's attributes
        grp.attrs['class'] = type(self).__name__
        grp.attrs['learning'] = self._learning
        grp.attrs['learning_rate'] = self._learning_rate
        grp.attrs['learning_type'] = self._learning_type
        grp.attrs['inhibitory'] = self._inhibitory
        grp.attrs['recurrent'] = self._recurrent
        grp.attrs['pre_layer_name'] = self._pre_layer.name
        grp.attrs['post_layer_name'] = self._post_layer.name
        if self._weight is not None:
            grp.create_dataset('weight', data=self._weight, compression=9)  # Gzip compression level 9

    @property
    def activity(self):
        """
        This is where the connection's logic is defined. Depending on the type of connection, return the weighted sum,
        the negative weighted sum, a simple bias or the modulated activity.
        :return: A 2-D numpy array of shape (post_layer.size, 1).
        """

        raise NotImplementedError("The Connection class should never be directly instantiated. You should sub-class it"
                                  "and override its activity property.")

    @property
    def weight(self):
        """
        Simply return the weight matrix.
        :return: A numpy array representing the synaptic strength of the connection between pre- and post-layers. Or
        None if the weight matrix has not been initialized.
        """

        return self._weight

    @weight.setter
    def weight(self, val):
        """
        Set the weight matrix for the current connection.
        :param val: A numpy array representing the synaptic strengths of the connections.
        :return: Nothing.
        """

        if val is None:
            self._weight = val
        else:
            # Make sure the value provided has the right shape
            assert val.shape == (self._post_layer.size, self._pre_layer.size), \
                "Error: Provided a weight matrix of shape: {}, while connection needs a matrix of " \
                "shape: {}".format(val.shape, (self._post_layer.size, self._pre_layer.size))

            # Assign the weight
            self._weight = val

    @property
    def learning(self):
        return self._learning

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, val):
        self._learning_rate = val

    @property
    def learning_type(self):
        return self._learning_type

    @property
    def pre_layer(self):
        return self._pre_layer

    @property
    def post_layer(self):
        return self._post_layer


class InputConnection(Connection):
    """
    Define a simple connection between an InputLayer and any Layer.
    """

    def __init__(self, pre_layer, post_layer, weight=None, learning=True, learning_type=LearningType.NONE, learning_rate=1e-7,
                 inhibitory=False):
        """
        Initialize the parent class and set the weight matrix.
        :param pre_layer: An instance of a Layer's sub-class, representing the pre-synaptic input layer.
        :param post_layer: An instance of a Layer's sub-class, representing the pos-synaptic output layer.
        :param weight: A numpy array representing the synaptic strength of the connections between the pre- and
        post-layers.
        :param learning: A Boolean indicating whether the synaptic strength is adaptable or not.
        :param learning_type: A instance of the LearningType enumeration, indicating the type of learning rule that
        will be applied to update the connection's synaptic's strength.
        :param learning_rate: A float representing the rate at which the connection's strenght is able to adapt.
        :param inhibitory: A Boolean indicating whether the connection is inhibitory or not.
        """

        if weight is None:
            # Initialize the weight matrix
            if learning:
                weight = np.clip(np.random.normal(0.5, 0.25, (post_layer.size, pre_layer.size)), 0, 1)
            else:
                # Divide by the square root of the input population, so that all pre-synaptic neurons have the same
                # contribution to the activity of the post-synaptic neurons
                weight = np.ones((post_layer.size, pre_layer.size),
                                 dtype=np.float32) / np.sqrt(pre_layer.size)

        # Initialize the parent class
        super(InputConnection, self).__init__(pre_layer, post_layer, weight=weight, learning=learning,
                                              learning_type=learning_type, learning_rate=learning_rate,
                                              inhibitory=inhibitory, recurrent=False)

    @property
    def activity(self):
        """
        Return the weighted activity of the pre-synaptic input layer.
        :return: A numpy array of shape (post_layer.size, 1).
        """

        if self._inhibitory:
            return np.negative(np.matmul(self._weight, self._pre_layer.output))
        else:
            return np.matmul(self._weight, self._pre_layer.output)


class StdConnection(Connection):
    """
    Define a standard connection between two hidden layers.
    """

    def __init__(self, pre_layer, post_layer, weight=None, learning=True, learning_type=LearningType.NONE, learning_rate=1e-7,
                 inhibitory=False, recurrent=False):
        """
        Initialize the parent class and set the weight matrix.
        :param pre_layer: An instance of a Layer's sub-class, representing the pre-synaptic input layer.
        :param post_layer: An instance of a Layer's sub-class, representing the pos-synaptic output layer.
        :param weight: A numpy array representing the synaptic strength of the connections between the pre- and
        post-layers.
        :param learning: A Boolean indicating whether the synaptic strength is adaptable or not.
        :param learning_type: A instance of the LearningType enumeration, indicating the type of learning rule that
        will be applied to update the connection's synaptic's strength.
        :param learning_rate: A float representing the rate at which the connection's strenght is able to adapt.
        :param inhibitory: A Boolean indicating whether the connection is inhibitory or not.
        :param recurrent: A Boolean indicating whether the connection is recurrent or not.
        """

        if weight is None:
            # Initialize the weight matrix
            if learning:
                weight = np.clip(np.random.normal(0.5, 0.25, (post_layer.size, pre_layer.size)), 0, 1)
            else:
                # Divide by the square root of the input population, so that all pre-synaptic neurons have the same
                # contribution to the activity of the post-synaptic neurons
                weight = np.ones((post_layer.size, pre_layer.size),
                                 dtype=np.float32) / np.sqrt(pre_layer.size)

        # Initialize the parent class
        super(StdConnection, self).__init__(pre_layer, post_layer, weight, learning, learning_type, learning_rate,
                                            inhibitory, recurrent)

    @property
    def activity(self):
        """
        Return the weighted activity of the pre-synaptic input layer.
        :return: A numpy array of shape (post_layer.size, 1).
        """

        if self._inhibitory:
            return np.negative(np.matmul(self._weight, self._pre_layer.output))
        else:
            return np.matmul(self._weight, self._pre_layer.output)


class ModulationConnection(Connection):
    """
    Define a modulatory connection between two hidden layers. Such as the one between the Ventral Tegmental Area and
    Amygdala for example.
    """

    def __init__(self, pre_layer, post_layer, inhibitory=False, recurrent=False):
        """
        Initialize the parent class and set the weight matrix.
        :param pre_layer: An instance of a Layer's sub-class, representing the pre-synaptic input layer.
        :param post_layer: An instance of a Layer's sub-class, representing the pos-synaptic output layer.
        :param inhibitory: A Boolean indicating whether the connection is inhibitory or not.
        :param recurrent: A Boolean indicating whether the connection is recurrent or not.
        """

        # Check if the pre-layer is broadcastable to the post-layer
        if not all((m == n) or (m == 1) or (n == 1)
                   for m, n in zip(pre_layer.output.shape[::-1], post_layer.output.shape[::-1])):
            raise IndexError("The pre-synaptic layer's output should have a shape that can be broadcast to the "
                             "post-synaptic layer. Pre-layer is: {}, while "
                             "post-layer is: {}".format(pre_layer.output.shape, post_layer.output.shape))

        # Initialize the parent class
        super(ModulationConnection, self).__init__(pre_layer, post_layer, weight=None, learning=False,
                                                   learning_type=LearningType.NONE, learning_rate=0.,
                                                   inhibitory=inhibitory, recurrent=recurrent)

    @property
    def activity(self):
        """
        Simply return the modulation values provided by the pre-synaptic layer.
        :return: A numpy array of the same shape as the post-synaptic layer's output.
        """

        if self._inhibitory:
            return np.negative(self._pre_layer.output)
        else:
            return self._pre_layer.output


class BiasConnection(Connection):
    """
    Define a connection that biases the activity of the post-synaptic layer with that of the pre-synaptic layer. To be
    connected, both pre- and post-synaptic layers should have the same shape.
    """

    def __init__(self, pre_layer, post_layer, inhibitory=False, recurrent=False):
        """
        Initialize the connection's attributes, as well as its weight matrix.
        :param pre_layer: An instance of a Layer's sub-class, representing the pre-synaptic input layer.
        :param post_layer: An instance of a Layer's sub-class, representing the pos-synaptic output layer.
        :param inhibitory: A Boolean indicating whether the connection is inhibitory or not.
        :param recurrent: A Boolean indicating whether the connection is recurrent or not.
        """

        # Check that the outputs of both layers have the same shape
        if pre_layer.size!= post_layer.size:
            raise IndexError("A BiasConnection can only connect layers of the same shape. "
                             "Pre-layer is: {}, while post-layer is: {}".format(pre_layer.output.shape,
                                                                                post_layer.output.shape))

        # Initialize the parent class
        super(BiasConnection, self).__init__(pre_layer, post_layer, weight=None, learning=False,
                                             learning_type=LearningType.NONE, learning_rate=0, inhibitory=inhibitory,
                                             recurrent=recurrent)

    @property
    def activity(self):
        """
        Simply return the bias values.
        :return: A numpy array containing the bias values from the pre-synaptic layer.
        """

        if self._inhibitory:
            return np.negative(self._pre_layer.output)
        else:
            return self._pre_layer.output


class LateralConnection(Connection):
    """
    Define a connection between neurons of the same layer for lateral inhibition.
    """

    def __init__(self, pre_layer, post_layer=None, weight=None, learning=True, learning_type=LearningType.NONE,
                 learning_rate=1e-7):
        """
        Initialize the connection's attributes, as well as its weight matrix.
        :param pre_layer: An instance of a Layer's sub-class, representing the pre-synaptic layer. Or both the pre-
        and post-synaptic layer if post_layer is None.
        :param post_layer: An instance of a Layer's sub-class, representing the post-synaptic layer. A value of None,
        indicates that the lateral influence is performed on the same layer.
        :param weight: A numpy array representing the synaptic strength of the connections between the pre- and
        post-layers.
        :param learning: A Boolean indicating whether the synaptic strength is adaptable or not.
        :param learning_type: A instance of the LearningType enumeration, indicating the type of learning rule that
        will be applied to update the connection's synaptic's strength.
        :param learning_rate: A float representing the rate at which the connection's strenght is able to adapt.
        """

        # In this case the lateral influence is local
        if post_layer is None:
            post_layer = pre_layer

        # Initialize the weight matrix
        if weight is None:
            if learning:
                # Initialize the weight matrix
                weight = np.zeros((post_layer.size, pre_layer.size), dtype=np.float32)
                # All values are initialized to zero see paper entitled: "Adaptive network for optimal linear feature
                # extraction", Foldiak (1989), for more details on why this is.
            else:
                if post_layer == pre_layer:  # Local lateral inhibition
                    # Initialize the weight matrix, so that each neuron has the same impact on the other neurons
                    # The whole np.abs(np.eye() - 1) is to create a matrix of ones with a diagonal of zeros
                    weight = -np.abs(np.eye(pre_layer.size, pre_layer.size, dtype=np.float32) - 1) / np.sqrt(pre_layer.size)
                else:  # Remote lateral inhibition
                    # Initialize the weight matrix, so that each neuron has the same impact on the other neurons
                    weight = -np.ones((post_layer.size, pre_layer.size), dtype=np.float32) / np.sqrt(pre_layer.size)

        # Initialize the parent class
        super(LateralConnection, self).__init__(pre_layer=pre_layer, post_layer=post_layer, weight=weight,
                                                learning=learning, learning_type=learning_type,
                                                learning_rate=learning_rate, inhibitory=True, recurrent=False)

    @property
    def activity(self):
        """
        Simply return the value of the lateral inhibition for each neuron.
        :return: A numpy array containing the inhibition values.
        """

        # No need to call np.negative() here, even though the connection is inhibitory, since the update rule produces
        # negative weights. See previously mentioned paper for details on Foldiak's update rule.
        return np.matmul(self._weight, self._pre_layer.output)
