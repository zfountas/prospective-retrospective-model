#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This file contains the logic of the perception loop of the system and
connects together semantic and episodic memories.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__authors__ = ["Zafeirios Fountas", "Kyriacos Nikiforou", "Anastasia Sylaidi"]
__credits__ = ["Warrick Roseboom", "Anil Seth",  "Murray Shanahan"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Zafeirios Fountas"
__email__ = "fountas@outlook.com"
__status__ = "Published"

from __future__ import print_function
import numpy as np
import pickle
import os
import json
import time

class PredictiveCodingSystem:
    def __init__(self, parameters_filename, episodic_recall, spatial_attention):
        params = json.load(open(parameters_filename))

        self.no_layers = len(params["network"]["layers_to_use"])
        self.www = params["pred_coding"]["www"]
        self.recall_effort = params["episodic_memory"]["recall_effort"]
        self.error_vector = {}
        self.Mmean = {}

        self.fw_for_pred = False
        self.blindspot = False
        self.dreaming = False
        self.timing = False

        self.recalled_list = {}

        # Flag of whether episodic recall is used in this simulation
        self.episodic_recall = episodic_recall
        self.spatial_attention = spatial_attention
        self.attention_mask = {} # all layers

        self.timing = False

    def update(self, atn, semantic_memory, episodic_memory, layer, attention_mask):
        # Initialization
        if layer not in self.error_vector:
            self.error_vector[layer] = np.zeros(np.shape(atn))

        ## ARCTITECTURE STEPS
        ## ---------------------------------------------------------------------
        # 1. update sensory input (atn, via  alexnet connections)
        # 2. Make prediction with the generative model (Mn, via semantic_memory)
        # 3. Check episodic memory for potential recall
        # 4. calculate error 3i = (an - Mn)
        # 5. update semantic memory (priors, via Mn, g, and error - Kalman filter)
        # 6. return error
        # ----------------------------------------------------------------------

        ## 1. UPDATE SENSORY INPUT
        # ... this is done outside this method (in pseudocodix) for now ...

        ## 2. MAKE PREDICTION
        # Get the predicited label from semantic memory, for layer n, given the
        # activation of layer n+1 (atn1), the last activation (for t-1) of layer
        # n (at1n), and the last episodic memory node of layer n (e1n)

        ## If we're in the top layer, we consider activation always the same!

        ## Get the distribution of the prior that is more likely to be
        # currently observed, as well as the level of suprise based on this
        # prediction

        # If there is a node that we are currently in the process of recalling,
        # don't run the generative model!
        if self.timing:
            time_before1 = time.time()
        e1n_index = episodic_memory.get_last_in_layer(layer).prior_index

        if self.dreaming and layer == 0:
            semantic_memory.WW[layer] = 0.0

        recalled_prior_index = -1
        if layer in self.recalled_list:
            # If there is something left on the recalled_list, use & remove it!
            if len(self.recalled_list[layer]) > 0:
                # Returns the index in the EPISODIC memory
                recalled_node = self.recalled_list[layer].pop(0)
                print("\rRECALLING:", layer, recalled_node, end='')
            else:
                recalled_node = -1
            # If list empty, remove the key..!
            if len(self.recalled_list[layer]) == 0:
                self.recalled_list.pop(layer)

            if recalled_node >= 0:
                # Returns the index in the SEMANTIC memory
                recalled_prior_index = episodic_memory.all_nodes[layer][recalled_node].prior_index

                # Update sensory information
                atn = self.www*atn + (1.0-self.www)*episodic_memory.all_nodes[layer][recalled_node].activation

            # If we just finished recalling, delete the step3 tree
            if len(self.recalled_list) == 0:
                episodic_memory.step2_tree = []
                episodic_memory.step3_tree = []

        # Either use the generative model or force a particular prior and return the mean and variance from semantic memory
        self.Mmean[layer], Mvar = semantic_memory.predict_layer(n=layer, e1n_index=e1n_index, forced_prior_index=recalled_prior_index)

        if self.timing:
            time_before2 = time.time()
            time1 = time_before2 - time_before1

        # 3. CHECK EPISODIC MEMORY
        if self.episodic_recall and self.recalled_list == {}:
            # Here we check whether the episodic memory system will start a
            # new recall of an episode
            prior_index_1 = semantic_memory.last_prior_index[layer]
            last_mean = semantic_memory.M[layer][prior_index_1].mean
            last_var = semantic_memory.M[layer][prior_index_1].var
            # NOTE: We need this second term here as there is no info in episodic
            # memory (or here) about the actual neuron size of a layer!!
            # So this is the scaling factor of the variance
            last_var *= semantic_memory.av_variance[layer][0]

            recalled_root = episodic_memory.check_for_recall_root(layer = layer,
                                            current_prior_index = prior_index_1,
                                   prior_mean = last_mean, prior_var = last_var)

            if recalled_root >=0:
                # NOTE: last_mean/var etc just got values from above..
                episodic_memory.recall(root_layer = layer, root_index = recalled_root,
                                       semantic_memory = semantic_memory,
                                       effort = self.recall_effort)
                # Pass the list of recalled nodes to this class
                for i in range(len(episodic_memory.step4_frames)):
                   if len(episodic_memory.step4_frames[i]) > 0:
                       # NOTE: list() is super important here so we make a copy of
                       # this list of indeces. Otherwise, when we pop later,
                       # it will remove the actual list inside semantic_memory..
                       #self.recalled_list[i] = list(episodic_memory.step4_frames[i])
                       # NOTE: The list() is removed on purpose because we do
                       # actually need to remove the elements from the episodic
                       # memory module too. We just take a pointer to each layer's list
                       self.recalled_list[i] = episodic_memory.step4_frames[i]

        ## XX. CALCULATE THE MASK STUFF
        if self.spatial_attention:
            if np.shape(attention_mask) != np.shape(atn):
                print("ERROR: Dimensionality mismatch", np.shape(attention_mask),np.shape(self.Mmean[layer]), np.shape(atn))
                exit()
            # Else, we just keep the attention mask that we received as input!
        else:
            attention_mask = 0.5

        if layer == 0:
            if self.blindspot:
                attention_mask = np.ones(np.shape(atn))*0.5
                middle_up = int(1.1*np.shape(attention_mask)[1]/2)
                middle_down = int(0.9*np.shape(attention_mask)[1]/2)
                attention_mask[:,middle_down:middle_up,:] = 0.0
            elif self.dreaming:
                attention_mask = 0.0

        self.attention_mask[layer] = attention_mask # For visualization..

        if self.timing:
            time_before3 = time.time()
            time2 = time_before3 - time_before2

        # 4. CALCULATE ERROR (an - Mn)
        if type(self.Mmean[layer]) == np.ndarray:
            #self.error_vector[layer] = new_activations - self.Mmean[layer]
            self.error_vector[layer] = atn - self.Mmean[layer]
        else:
            self.error_vector[layer] = np.array([])

        if self.timing:
            time_before4 = time.time()
            time3 = time_before4 - time_before3

        # 5. UPDATE SEMANTIC MEMORY
        # episodic memory is passed to update indices when priors merge
        semantic_memory.update_priors(atn, layer, attention_mask)
        if self.timing:
            time_before5 = time.time()
            time4 = time_before5 - time_before4

        did_merge = semantic_memory.check_for_merging(layer, episodic_memory)
        if self.timing:
            time_before6 = time.time()
            time5 = time_before6 - time_before5
        if self.timing: print("Layer:", layer, "This is how long it takes:", time1, time2, time3, time4, time5)

        # The error has been updated here..
        if self.fw_for_pred:
            semantic_memory.last_atn[layer] = np.copy(atn)
        return did_merge























#
