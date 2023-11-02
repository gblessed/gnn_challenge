"""
   Copyright 2023 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import tensorflow as tf


class Xigmoid(tf.keras.layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super(Xigmoid, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        cond = inputs>0
        ax = self.alpha * inputs
        if_x = tf.exp(ax)
        else_x = 1.0 / if_x

        if_x = if_x - 1.0
        else_x = 1.0 - else_x

        cond_x = tf.where(cond, if_x, else_x)
        return tf.math.sigmoid(self.alpha * cond_x)

class Baseline_cbr_mb(tf.keras.Model):
    min_max_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
    }
    min_max_scores = None

    name = "Baseline_cbr_mb"

    def __init__(self, override_min_max_scores=None, name=None):
        super(Baseline_cbr_mb, self).__init__()

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32

        if override_min_max_scores is not None:
            self.set_min_max_scores(override_min_max_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        #GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate"),
            return_sequences=True,
            return_state=True,
 


            name="PathUpdateRNN",
        )

        # self.path_update = tf.keras.layers.LSTM(
        #     units=self.path_state_dim,
        #     return_sequences=True,
        #     return_state=True,
        #     name="PathUpdateLSTM"
        # )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            recurrent_regularizer =tf.keras.regularizers.l2(0.01),
        )

        self.flow_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=5),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.gelu
                ),
            ],
            name="PathEmbedding",
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=2),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.gelu
                ),
            ],
            name="LinkEmbedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
            ],
            name="PathReadout",
        )
    
    def set_min_max_scores(self, override_min_max_scores):
        assert (
            type(override_min_max_scores) == dict
            and all(kk in override_min_max_scores for kk in self.min_max_scores_fields)
            and all(len(val) == 2 for val in override_min_max_scores.values())
        ), "overriden min-max dict is not valid!"
        self.min_max_scores = override_min_max_scores

    @tf.function
    def call(self, inputs):
        # Ensure that the min-max scores are set
        assert self.min_max_scores is not None, "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_packets = inputs["flow_packets"]
        flow_packet_size = inputs["flow_packet_size"]
        flow_type = inputs["flow_type"]
        link_capacity = inputs["link_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]

        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)
        # print("flow_traffic",flow_traffic )
        # Initialize the initial hidden state for paths
        path_state = self.flow_embedding(
            tf.concat(
                [
                    (flow_traffic - self.min_max_scores["flow_traffic"][0])
                    * self.min_max_scores["flow_traffic"][1],
                    (flow_packets - self.min_max_scores["flow_packets"][0])
                    * self.min_max_scores["flow_packets"][1],
                    (flow_packet_size - self.min_max_scores["flow_packet_size"][0])
                    * self.min_max_scores["flow_packet_size"][1],
                    flow_type,
                ],
                axis=1,
            )
        )


        # Initialize the initial hidden state for links
        link_state = self.link_embedding(
            tf.concat(
                [
                    (link_capacity - self.min_max_scores["link_capacity"][0])
                    * self.min_max_scores["link_capacity"][1],
                    load,
                ],
                axis=1,
            ),
        )
        # initial_h_state = tf.zeros((32, self.path_state_dim))
        # initial_c_state = tf.zeros((32, self.path_state_dim))
        # path_state = [initial_h_state, initial_c_state]
        # Iterate t times doing the message passing

        for _ in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            previous_path_state = path_state

            path_state_sequence, path_state = self.path_update(
                link_gather, initial_state=path_state
            )
            # path_state_sequence, path_state, path_c = self.path_update(
            #     link_gather, initial_state=None
            # )
            # We select the element in path_state_sequence so that it corresponds to the state before the link was considered

            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )
            # print("*********", path_state_sequence.shape)
            ###################
            #   PATH TO LINK  #
            ###################
            path_gather = tf.gather_nd(
                path_state_sequence, path_to_link, name="PathToRLink"
            )

            
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            
            

            link_state, _ = self.link_update(path_sum, states=link_state)

        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])
        capacity_gather = tf.gather(link_capacity, link_to_path)
        delay_sequence = occupancy / capacity_gather
        delay = tf.math.reduce_sum(delay_sequence, axis=1)
        return delay

class Baseline_mb(tf.keras.Model):
    min_max_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
    }

    name = "Baseline_mb"

    def __init__(self, override_min_max_scores=None, name=None):
        super(Baseline_mb, self).__init__()

        self.iterations = 4
        self.path_state_dim = 64
        self.link_state_dim = 64

        if override_min_max_scores is not None:
            self.set_min_max_scores(override_min_max_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate"),
            return_sequences=True,
            return_state=True,
            name="PathUpdateRNN",
        )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )
        

        ## 2 layer neural network to get the embedding of the paths.
        #(3x1) average traffic volume, distribution parameters 
        self.path_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=5),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.gelu
                ),
            ],
            name="PathEmbedding",
        )

         ## 2 layer neural network to get the embedding of the links, 
         ## (2x1) = link load and scheduling policy
        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=2),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.gelu
                ),
            ],
            name="LinkEmbedding",
        )

        # all read ours are 3 layer neural networks
        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
            ],
            name="PathReadout",
        )

    def set_min_max_scores(self, override_min_max_scores):
        assert (
            type(override_min_max_scores) == dict
            and all(kk in override_min_max_scores for kk in self.min_max_scores_fields)
            and all(len(val) == 2 for val in override_min_max_scores.values())
        ), "overriden min-max dict is not valid!"
        self.min_max_scores = override_min_max_scores

    @tf.function
    def call(self, inputs):
        # Ensure that the min-max scores are set
        assert self.min_max_scores is not None, "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_packets = inputs["flow_packets"]
        flow_packet_size = inputs["flow_packet_size"]
        link_capacity = inputs["link_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs[" "]
        ipg_mean = inputs["ipg_mean"]
        ipg_var = inputs["ipg_var"]


        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)

        # Initialize the initial hidden state for paths
        path_state = self.path_embedding(
            tf.concat(
                [
                    (flow_traffic - self.min_max_scores["flow_traffic"][0])
                    * self.min_max_scores["flow_traffic"][1],
                    (flow_packets - self.min_max_scores["flow_packets"][0])
                    * self.min_max_scores["flow_packets"][1],
                    (flow_packet_size - self.min_max_scores["flow_packet_size"][0])
                    * self.min_max_scores["flow_packet_size"][1],
                    ipg_mean,
                    ipg_var,

                ],
                axis=1,
            )
        )


        # Initialize the initial hidden state for links
        link_state = self.link_embedding(
            tf.concat(
                [
                    (link_capacity - self.min_max_scores["link_capacity"][0])
                    * self.min_max_scores["link_capacity"][1],
                    load,
                ],
                axis=1,
            ),
        )

        # Iterate t times doing the message passing
        for _ in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            previous_path_state = path_state
            path_state_sequence, path_state = self.path_update(
                link_gather, initial_state=path_state
            )
            # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #   PATH TO LINK  #
            ###################
            path_gather = tf.gather_nd(
                path_state_sequence, path_to_link, name="PathToRLink"
            )
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            link_state, _ = self.link_update(path_sum, states=link_state)

        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])
        capacity_gather = tf.gather(link_capacity, link_to_path)
        delay_sequence = occupancy / capacity_gather
        delay = tf.math.reduce_sum(delay_sequence, axis=1)
        return delay