#!/usr/bin/python
# -*- coding: utf-8 -*-
import psutil
import logging
import logging.config
import h5py
import numpy as np
from collections import OrderedDict, defaultdict
from ANNFramework.Layer import LayerType, LayerFactory
from ANNFramework.Activations import ActType
from ANNFramework.Group import ConPattern, Group
from ANNFramework.LearningRule import LearningRuleFactory, LearningType
from ANNFramework.Connection import ConnectionFactory
try:
    import ANNFramework.local_settings as settings
    print("Loading local configuration file.")
except ImportError:
    print("Loading global configuration file.")
    import ANNFramework.settings as settings

# Configure the logging facility for this module
logging.config.dictConfig(settings.LOGGING)
# Reset the process' cpu affinity
p = psutil.Process()
all_cpus = list(range(psutil.cpu_count()))
p.cpu_affinity(all_cpus)


class Network:
    """
    Defines a model of the Basal Ganglia along with its cortico-thalamic loops. With possible extensions using
    sub-cortical areas, such as the amygdala and hypothalamus. Hence, implementing the full PrimEmo architecture.
    """

    def __init__(self, lesion=False):
        """
        Define the weights and layers attributes of the brain.
        :param lesion: A Boolean indicating whether the network has been lesioned or not.
        """

        # Initialize the Network's attributes
        self._lesion = lesion
        self._logger = logging.getLogger('BrainLes' if lesion else 'Brain')

        self._layers = OrderedDict()  # {name: layer, ... }
        self._connections = defaultdict(list)  # {learning_type: [con1, con2, ... ], ... }
        self._learning_rules = dict()  # {learning_type: LearningRule, ... }

    def build(self, dict_conf):
        """
        Build the network from scratch depending on the given configuration.
        :param dict_conf: A dictionary containing the Network's configuration.
        :return: Nothing.
        """

        # Create all the layers/groups in order
        for lay_name, lay_data in dict_conf.items():
            # Extract the type of layer to be built
            lay_type = lay_data['type']

            # Create the Layer or Group
            if lay_type == LayerType.GROUP:
                self._layers[lay_name] = Group(lay_name, lay_data['size'], lay_data.get('layer_sizes', None),
                                               lay_data.get('layer_type', LayerType.HIDDEN),
                                               lay_data.get('activation', ActType.TANH), lay_data.get('norm', None))
            else:
                self._layers[lay_name] = LayerFactory.get_layer(lay_data['type'], lay_name, lay_data['size'],
                                                                lay_data.get('activation', ActType.NONE),
                                                                lay_data.get('norm', None))

        # Add the connections between layers
        for post_name, post_layer in self._layers.items():
            # Retrieve the data related to the current layer
            lay_data = dict_conf.get(post_name)

            # If the add_input() method is not implemented, do no waist your time and simply move on to the next layer
            if not hasattr(post_layer, 'add_input'):
                continue

            # Get the list of inputs
            inputs = lay_data.get('inputs', None)

            # Establish the different connections
            if inputs is not None:
                for pre_name, in_data in inputs.items():
                    # Get the pre-synaptic layer
                    pre_layer = self._layers[pre_name]
                    # Extract the common connection configurations
                    learning = in_data.get('learning', True)
                    lrn_type = in_data.get('learning_type', LearningType.NONE)
                    lrn_rate = in_data.get('learning_rate', 1e-7)
                    conn_weight = in_data.get('weight', None)
                    if isinstance(post_layer, Group):
                        conn_list = post_layer.add_input(pre_layer=pre_layer, con_type=in_data['type'],
                                                         weights=conn_weight,
                                                         learning=learning,
                                                         learning_type=lrn_type,
                                                         learning_rate=lrn_rate,
                                                         con_pattern=in_data.get('pattern', ConPattern.DENSE),
                                                         start=in_data.get('start', 0), end=in_data.get('end', None))
                    else:
                        # Create the connection(s)
                        conn_list = post_layer.add_input(pre_layer=pre_layer, con_type=in_data['type'],
                                                         weights=conn_weight,
                                                         learning=learning,
                                                         learning_type=lrn_type,
                                                         learning_rate=lrn_rate)

                    # Add the connection(s) to the dictionary
                    if learning and lrn_type != LearningType.NONE:
                        self._connections[lrn_type].extend(conn_list)

        # Make sure the learning rule corresponding to the given type is already instantiated
        # Since connections are stored according to their learning type, the keys are the learning type to implement
        self._learning_rules = {lrn_type: LearningRuleFactory.get_learning_rule(lrn_type) for lrn_type in
                                self._connections}

    def activate(self, input_dict, out_layer):
        """
        Given a dictionary containing the activation values for any input layer, propagate the activation throughout the
        network. Propagation is done in the same order that layers are defined in the configuration.
        :param input_dict: A dictionary containing any 2-D array representing the different formatted sensory data
        to be fed as input to the network.
        :param out_layer: A string or instance of the Layer class, representing the network's output.
        :return: Nothing.
        """

        # Get the output layer
        if isinstance(out_layer, str):
            out_layer = self._layers[out_layer]

        # Initialize the variable keeping track of the previous activation values for the output layer
        old_out = np.zeros((out_layer.size, 1), dtype=np.float32)

        # Inject all inputs to their respective layers.
        for layer_name, in_data in input_dict.items():
            self._layers[layer_name].output = in_data

        # Loop at most 100 times in search of stability in the output layer
        for _ in range(100):
            # Propagate the activation into the network
            for layer in self._layers.values():
                # Simply activate the layer
                layer.activate()

            # Check if the output is stable
            if np.all(old_out == out_layer.output):
                # Nothing to be done any more
                break

            # Keep track of the old activation values
            old_out = out_layer.output

    def learn(self, input_dict, out_layer):
        """
        Given a dictionary containing the activation values for all the input layers, propagate the activation
        throughout the network, then update the connections' synaptic weights.
        :param input_dict: A dictionary containing any 2-D array representing the different formatted sensory data
        to be fed as input to the network.
        :param out_layer: A string or instance of the Layer class, representing the network's output.
        :return: Nothing.
        """

        # Activate the network until stable (the output is not stored since it has no impact on the learning process)
        self.activate(input_dict, out_layer)

        # Call on all the registered learning rules
        for lrn_type, lrn_rule in self._learning_rules.items():
            if lrn_rule is None:
                continue
            # Get the connections to be updated
            conns = self._connections.get(lrn_type, [])
            # Call the learning rule on the list of connections
            lrn_rule.update(conns)

        for name, layer in self._layers.items():
            if layer.norm is not None:
                layer.normalize_weights()

    def shutdown(self):
        """
        Simply clear anything from the tensorflow session.
        :return: Nothing.
        """
        pass

    def save(self, file_path):
        """
        Saves both the weights, with their corresponding input/output layers, and the layers.
        :param file_path: A string representing the absolute path where the model should be saved.
        :return: Nothing.
        """

        # Create a new save file using the HDF5 format
        file = h5py.File(file_path, "w")

        # Save the network's attributes
        file.attrs['lesion'] = self._lesion
        dt = h5py.special_dtype(vlen=str)
        file.attrs.create('ordered_layer_names', list(self._layers.keys()), dtype=dt)

        # Save each layer in a different group
        for name, layer in self._layers.items():
            # Create the group
            grp = file.create_group(name)
            # Let the layer do its own thing
            layer.save(grp)

        # Close the file
        file.close()

    def load(self, file_path):
        """
        Loads a brain, previously saved at the given location.
        :param file_path: A string giving the absolute location where the brain has been saved.
        :return: Nothing.
        """

        # Open the file in read mode
        file = h5py.File(file_path, "r")

        # Reset the lesion attribute
        self._lesion = file.attrs['lesion']

        # Get the ordered list of layers' name
        ordered_layer_names = file.attrs['ordered_layer_names']

        # Load the layers in order
        for layer_name in ordered_layer_names:
            # Get the h5py.group containing the layer's data
            layer_grp = file[layer_name]

            if layer_grp.attrs.get('class') == 'Group':
                # Create an empty Group
                group = Group(layer_name, activation_type=layer_grp.attrs.get('activation_type'),
                              norm=layer_grp.attrs.get('norm', None))
                # Instantiate the different layers contained within the group
                # The sub-group's keys are the layers' names. Order does not matter
                layers = []
                for sub_lay_name, sub_lay_grp in layer_grp.items():
                    layers.append(LayerFactory.get_from_str(sub_lay_grp.attrs.get('class'), sub_lay_name,
                                                            sub_lay_grp.attrs.get('size'),
                                                            activation_type=sub_lay_grp.attrs.get('activation_type'),
                                                            norm=sub_lay_grp.attrs.get('norm', None)))

                # Assign the layers to the group
                group.layers = layers

                # Assign the group to the Network
                self._layers[layer_name] = group

            else:
                # Get an instance of the correct Layer sub-class
                layer = LayerFactory.get_from_str(layer_grp.attrs.get('class'), layer_name, layer_grp.attrs.get('size'),
                                                  activation_type=layer_grp.attrs.get('activation_type'),
                                                  norm=layer_grp.attrs.get('norm', None))
                # Assign the layer to the Network
                self._layers[layer_name] = layer

        # Re-build the connections between layers
        for name, layer in self._layers.items():
            # Get the corresponding h5py.group
            layer_grp = file.get(name)

            # If layer is in fact a group
            if layer_grp.attrs.get('class') == 'Group':

                # Go through all the sub-layers
                for post_layer in layer.layers:
                    # Get the corresponding h5py.group
                    post_layer_grp = layer_grp.get(post_layer.name)

                    # And rebuild the connection
                    for conn_name, conn_grp in post_layer_grp.items():
                        # Get the pre-synaptic layer
                        pre_layer = self._find_layer_by_name(conn_grp.attrs.get('pre_layer_name'))

                        # Get the Connection instance
                        connection = ConnectionFactory.get_from_str(conn_grp.attrs.get('class'), pre_layer, post_layer,
                                                                    weight=conn_grp.get('weight', None),
                                                                    learning=conn_grp.attrs.get('learning', True),
                                                                    learning_type=conn_grp.attrs.get('learning_type'),
                                                                    learning_rate=conn_grp.attrs.get('learning_rate'),
                                                                    inhibitory=conn_grp.attrs.get('inhibitory'),
                                                                    recurrent=conn_grp.attrs.get('recurrent'))

                        # Store the connection according to its learning_type
                        if connection.learning:
                            self._connections[conn_grp.attrs.get('learning_type')].append(connection)

            else:  # It is indeed a standard layer
                # And rebuild the connection
                for conn_name, conn_grp in layer_grp.items():
                    # Get the pre-synaptic layer
                    pre_layer = self._find_layer_by_name(conn_grp.attrs.get('pre_layer_name'))

                    # Get the Connection instance
                    connection = ConnectionFactory.get_from_str(conn_grp.attrs.get('class'), pre_layer, layer,
                                                                weight=conn_grp.get('weight', None),
                                                                learning=conn_grp.attrs.get('learning', True),
                                                                learning_type=conn_grp.attrs.get('learning_type'),
                                                                learning_rate=conn_grp.attrs.get('learning_rate'),
                                                                inhibitory=conn_grp.attrs.get('inhibitory'),
                                                                recurrent=conn_grp.attrs.get('recurrent'))

                    # Store the connection according to its learning_type
                    if connection.learning:
                        self._connections[conn_grp.attrs.get('learning_type')].append(connection)

        # Finally initialize the learning_rule dictionary
        self._learning_rules = {lrn_type: LearningRuleFactory.get_learning_rule(lrn_type) for lrn_type in
                                self._connections}
        # Since connections are stored according to their learning type, the keys are the learning type to implement

        # Close the HDF5 formatted file
        file.close()

    def reset(self):
        """
        Simply empties the layers, connections and learning rules.
        :return: Nothing.
        """

        # Reset everything
        self._layers = OrderedDict()
        self._connections = defaultdict(list)
        self._learning_rules = dict()

    def _find_layer_by_name(self, layer_name):
        """
        Find and return the layer corresponding to the given name. Or None if not found within any Group or the list of
        Layers.
        :param layer_name: A string representing the layer's name.
        :return: An instance of a Layer's sub-class if it has been found. None, otherwise.
        """

        if layer_name.find('@'):  # The layer belongs to a group
            # Get the Group instance
            group_name = layer_name.split('@')[0]
            try:
                group = self._layers[group_name]
            except IndexError:
                return None

            # Return the Layer's instance from within the group
            return group.find_layer_by_name(layer_name)
        else:  # It is a standard layer
            return self._layers[layer_name]

    @ property
    def layers(self):
        return self._layers
