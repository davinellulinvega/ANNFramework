#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import IntEnum, unique
import numpy as np
import scipy.special


def relu(x):
    """
    Element-wise and in-place application of the relu function.
    :param x: A numpy array.
    :return: A numpy array containing the relu values computed from x.
    """

    return np.maximum(x, 0)


def relu_derivative(x):
    """
    Define the derivative of the relu function.
    :param x: A numpy array.
    :return: A numpy array containing the value of the derivative for each value in x.
    """

    return np.where(x >= 0, 1, 0)


def leaky_relu(x, alpha=0.01):
    """
    Element-wise and in-place application of the leaky relu function.
    :param x:  A numpy array.
    :param alpha: A float representing the leak current.
    :return: A numpy array containing the leaky relu values computed from x.
    """

    return np.maximum(x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """
    Define the derivative function of the leaky relu function.
    :param x: A numpy array.
    :param alpha: A float representing the leak current.
    :return: A numpy array containing the computed value of the derivative for each value in x.
    """

    return np.where(x >= 0, 1, alpha)


def softmax(x):
    """
    Element-wise application of the softmax function.
    :param x: A numpy array.
    :return: A numpy array containing the softmax values computed from x.
    """

    # Compute the exponential once
    e_m = np.exp(x - np.max(x, axis=0))

    return e_m / e_m.sum(axis=0)


def softmax_derivative(x):
    """
    Define the element-wise computation of the value of the derivative of the softmax function.
    :param x: A numpy array.
    :return: A numpy array containing the computed values of the derivative for each value in x.
    """

    return softmax(x) * (1 - softmax(x))


def linear(x):
    """
    Simply return x as is, without any modifications.
    :param x: A numpy array.
    :return: The same numpy array as given in parameter.
    """

    return x


def linear_derivative(x):
    """
    Returns 1 regardless of the given parameter.
    :param x: A numpy array.
    :return: A numpy array containing only ones.
    """

    return np.ones(x.shape, dtype=x.dtype)


def tanh_derivative(x):
    """
    Define the derivative of the hyperbolic tangent function.
    :param x: A numpy array.
    :return: A numpy array containing the value of the derivative function for each value in x.
    """

    return 1 - np.square(np.tanh(x))


def sigmoid_derivative(x):
    """
    Define the derivative of the sigmoid functions
    :param x: A numpy array.
    :return: A numpy array containing the value of the derivative function computed for each value in x.
    """

    return scipy.special.expit(x) * (1 - scipy.special.expit(x))


class ActsFactory:
    """
    A factory that builds activation function depending on the given ActType.
    """

    @classmethod
    def get_activation(cls, act_type):
        """
        Given the activation type, return the corresponding activation function.
        :param act_type: An instance of the ActType enumeration, representing the type of activation function to
        procduce.
        :return: A callable method, representing the non-linearity to apply to a neuron's activity.
        """

        if act_type == ActType.SOFTMAX:
            return softmax
        elif act_type == ActType.RELU:
            return relu
        elif act_type == ActType.LEAKY_RELU:
            return leaky_relu
        elif act_type == ActType.TANH:
            return np.tanh
        elif act_type == ActType.SIGMOID:
            return scipy.special.expit
        elif act_type == ActType.LINEAR:
            return linear
        else:
            return None

    @classmethod
    def get_derivative(cls, act_type):
        """
        Given the activation type, return the derivative of the corresponding activation function. This is mostly used
        in the implemented LearningRules.
        :param act_type: An instance of the ActType enumeration, representing the type of activation function to derive.
        :return: A callable method, representing the derivative of the non-linearity applied to a neuron's activity.
        """

        if act_type == ActType.TANH:
            return tanh_derivative
        elif act_type == ActType.SIGMOID:
            return sigmoid_derivative
        elif act_type == ActType.RELU:
            return relu_derivative
        elif act_type == ActType.LEAKY_RELU:
            return leaky_relu_derivative
        elif act_type == ActType.LINEAR:
            return linear_derivative
        else:
            # TODO: implement derivative for SOFTMAX activation function
            return None


@unique
class ActType(IntEnum):
    NONE = -1
    SOFTMAX = 0
    RELU = 1
    LEAKY_RELU = 2
    TANH = 3
    SIGMOID = 4
    LINEAR = 5

