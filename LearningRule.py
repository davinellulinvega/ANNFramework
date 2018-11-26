#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import unique, IntEnum
from collections import defaultdict
import numpy as np
from ANNFramework.Activations import ActsFactory


@unique
class LearningType(IntEnum):
    NONE = -1
    OJA = 0
    BCM = 1
    AltBCM = 2
    # For both the IBCM and LBCM learning rule, see: http://www.scholarpedia.org/article/BCM_theory, for more details.
    IBCM = 3
    LBCM = 4
    OJA_BCM = 5
    FOLDIAK = 6
    HEBBIAN = 7


class LearningRuleFactory:
    """
    A factory that builds LearningRule depending on the given LearningType.
    """

    @classmethod
    def get_learning_rule(cls, learning_type, epsilon=0.001, win_size=100):
        """
        Given the learning_type, return an instance of the right LearningRule sub-class.
        :param learning_type: An instance of the LearningType enumeration, representing the type of learning rule to
        instantiate.
        :param epsilon: A float, in the range [0, 1[, representing the weight decay rate for the BCM learning rule.
        :param win_size: An integer representing the size of the window on which to compute the moving average.
        :return: A sub-class of the LearningRule class.
        """

        if learning_type == LearningType.OJA:
            return OjaLearningRule()
        elif learning_type == LearningType.BCM:
            return BcmLearningRule(epsilon, win_size)
        elif learning_type == LearningType.AltBCM:
            return AltBcmLearningRule()
        elif learning_type == LearningType.IBCM:
            return IBcmLearningRule(win_size)
        elif learning_type == LearningType.LBCM:
            return LBcmLearningRule(win_size)
        elif learning_type == LearningType.FOLDIAK:
            return FoldiakLearningRule()
        else:
            return None


class LearningRule:
    """
    An abstract class defining the basics required for a learning rule to adapt the synaptic strength of lists of
    Connections. This class should never be instantiated as is, but rather sub-classed and modified to suite your needs.
    """

    def update(self, connections):
        """
        Update in-place the synaptic strength of the connections, in the list. This is the heart of the learning rule,
        where the magic happens.
        :param connections: A list of Connection instances, whose weights have to be updated.
        :return: Nothing.
        """

        raise NotImplementedError("You should never instantiate a LearningRule directly, but rather sub-class it and "
                                  "override its update() method to suite your needs.")


class OjaLearningRule(LearningRule):
    """
    Adapt the synaptic strength of a connection using Oja's learning rule.
    """

    def update(self, connections):
        """
        Update in-place the synaptic strength of the connections contained within the list given in parameter.
        :param connections: A list of Connection instances.
        :return: Nothing.
        """

        # Update the weight matrix of each connection following Oja's rule
        for conn in connections:
            # Make sure the connection is modifiable
            if not conn.learning or conn.weight is None:
                continue
            # Update all the weights
            out = conn.post_layer.output
            in_line = conn.pre_layer.output.T
            conn.weight += conn.learning_rate * (in_line * out - np.square(out) * conn.weight)


class BcmLearningRule(LearningRule):
    """
    Adapt the synaptic strength of a connection using the Original BCM (1982) learning rule.
    """

    def __init__(self, epsilon=0.001, win_size=5):
        """
        Initialize the attributes required by the learning rule.
        :param epsilon: A float in the range [0, 1], representing the weight's rate of decay.
        :param win_size: An integer representing the size of the window on which the average is computed.
        """

        assert 0 < win_size, "The size of the window to compute the moving average should be a positive integer."

        # Initialize the parent class
        super(BcmLearningRule, self).__init__()

        self._epsilon = epsilon
        self._window_size = win_size
        # Used to keep track of the input window for computing the moving average for each connection
        self._windows = defaultdict(list)

    def update(self, connections):
        """
        Update in-place the synaptic strength of the connections contained within the list given in parameter.
        :param connections: A list of Connection instances.
        :return: Nothing.
        """

        # Update the weight matrix of each connection following the original BCM rule
        for conn in connections:
            # Make sure the connection is modifiable
            if not conn.learning or conn.weight is None:
                continue

            # Retrieve the window on which the average is to be computed
            win = self._windows[conn]
            # Divide by 0.5 so that the activity is regulated toward this value, which is central for the SIGMOID and SOFTMAX activation 
            # functions. See BCM (1982), Intrator (1992) and Law's (1994) papers for more details on this.
            win.append(conn.post_layer.output / 0.5)
            if len(win) > self._window_size:
                win.pop(0)

            # Compute the modification threshold
            thres = np.mean(win, axis=0)
            # Make sure the threshold never goes too low
            thres = np.minimum(thres, 1e-7)

            # Update the weight
            out = conn.post_layer.output
            in_line = conn.pre_layer.output.T
            conn.weight += out * in_line * (out - thres) - self._epsilon * conn.weight


class AltBcmLearningRule(LearningRule):
    """
    Adapt the synaptic strength of a connection using an alternative formulation of the Original BCM (1982) learning
    rule. See: Dayan and Abbott (2001) and Udeigwe (2017) for more details.
    """

    def __init__(self):
        """
        Declare and initialize the different properties required for computing the Alternative formulation of the BCM
        learning rule.
        """
        
        # Initialize the parent class
        super(AltBcmLearningRule, self).__init__()

        # Declare a mapping between a connection and its threshold
        self._thresholds = defaultdict(float)

    def update(self, connections):
        """
        Update in-place the synaptic strength of the connections contained within the list given in parameter.
        :param connections: A list of Connection instances.
        :return: Nothing.
        """

        # Update the weight matrix of each connection following the alternative version of the BCM learning rule.
        for conn in connections:
            # Make sure the connection is modifiable
            if not conn.learning or conn.weight is None:
                continue

            # Update the threshold for the current connection
            out = conn.post_layer.output
            self._thresholds[conn] += (conn.learning_rate / 10) * (np.square(out) - self._thresholds[conn])

            # Update the weight
            in_line = conn.pre_layer.output.T
            conn.weight += conn.learning_rate * out * in_line * (out - self._thresholds[conn])


class IBcmLearningRule(LearningRule):
    """
    Adapt the synaptic strength of a connection using the Intrator-BCM (1992) learning rule.
    """

    def __init__(self, win_size=5):
        """
        Initialize the attributes required by the learning rule.
        :param win_size: An integer representing the size of the window on which the average is computed.
        """

        assert 0 < win_size, "The size of the window to compute the moving average should be a positive integer."

        # Initialize the parent class
        super(IBcmLearningRule, self).__init__()

        self._window_size = win_size
        self._windows = defaultdict(list)

    def update(self, connections):
        """
        Update in-place the synaptic strength of the connections contained within the list given in parameter.
        :param connections: A list of Connection instances.
        :return: Nothing.
        """

        # Update the weight matrix of each connection following the IBCM rule
        # See: http://www.scholarpedia.org/article/BCM_theory, for more details
        for conn in connections:
            # Make sure the connection is modifiable
            if not conn.learning or conn.weight is None:
                continue

            # Retrieve the moving window for the average
            win = self._windows[conn]
            win.append(conn.post_layer.output / 0.5)
            if len(win) > self._window_size:
                win.pop(0)

            # Compute the modification threshold
            thres = np.mean(np.square(win), axis=0)
            # Make sure the threshold never goes too low
            thres = np.minimum(thres, 1e-7)

            # Update the weight
            out = conn.post_layer.output
            in_line = conn.pre_layer.output.T
            act_fun_deriv = ActsFactory.get_derivative(conn.learning_type)
            conn.weight += conn.learning_rate * in_line * out * (out - thres) * act_fun_deriv(out)


class LBcmLearningRule(LearningRule):
    """
    Adapt the synaptic strength of a connection using the Law-BCM (1994) learning rule.
    """

    def __init__(self, win_size=5):
        """
        Initialize the attributes required by the learning rule.
        :param win_size: An integer representing the size of the window on which the moving average is computed.
        """

        assert 0 < win_size, "The size of the window to compute the moving average should be a positive integer."

        # Initialize the parent class
        super(LBcmLearningRule, self).__init__()

        self._window_size = win_size
        self._windows = defaultdict(list)

    def update(self, connections):
        """
        Update in-place the synaptic strength of the connections contained within the list given in parameter.
        :param connections: A list of Connection instances.
        :return: Nothing.
        """

        # Update the weight matrix of each connection following the LBCM rule
        # See: http://www.scholarpedia.org/article/BCM_theory, for more details
        for conn in connections:
            # Make sure the connection is modifiable
            if not conn.learning or conn.weight is None:
                continue
            # Retrieve the moving window on which the average is to be computed
            win = self._windows[conn]
            win.append(conn.post_layer.output / 0.5)
            if len(win) > self._window_size:
                win.pop(0)

            # Compute the modification threshold
            thres = np.mean(np.square(win), axis=0)
            # Make sure the threshold never goes too low
            thres = np.minimum(thres, 1e-7)

            # Update the weight
            out = conn.post_layer.output
            in_line = conn.pre_layer.output.T
            conn.weight += conn.learning_rate * (out * in_line * (out - thres)) / thres


class FoldiakLearningRule(LearningRule):
    """
    Adapt the synaptic strength of a connection using Foldiak's update equation. This LearningRule sub-class is
    dedicated to adjusting the weights of lateral inhibitory connections.
    Please read the paper entitled: "Adaptive network for optimal linear feature extraction", Foldiak (1989), for more
    details.
    """

    def update(self, connections):
        """
        Update in-place the synaptic strength of the lateral connections contained within the list given in parameter.
        :param connections: A list of LateralConnection instances.
        :return: Nothing.
        """

        # Update the weight matrix of each connection following Foldiak's rule
        for conn in connections:
            # Make sure the connection is modifiable
            if not conn.learning or conn.weight is None:
                continue
            # Build a mask since the weight's diagonal should always be zero
            mask = np.abs(np.eye(*conn.weight.shape, dtype=np.float32) - 1)
            # Modify all the weights
            out = conn.post_layer.output
            conn.weight -= conn.learning_rate * out * out.T * mask
