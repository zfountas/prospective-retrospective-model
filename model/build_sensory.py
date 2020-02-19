#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This file provides the methods that initialize the neural networks used for
inference.

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

from model.sensory_system2 import *
from numpy import array, ones
import json

class IdentityNet:
    best_label = 'We are at identity net'
    def __init__(self):
        self.activations = array([])
        self.gradients = array([])
        self.generative_model_prediction = array([])
        self.raw_mask = []
        self.name = 'IdentityNet'
    def update(self, activations, save_raw_mask=False):
        self.activations = activations

class ContextNet:
    best_label = 'We have the same context'
    def __init__(self):
        self.activations = array([])
        # Has to have the dimensionality of the previous layer
        self.gradients = ones((1000))
        self.generative_model_prediction = array([])
        self.raw_mask = []
        self.name = 'ContexNet'
    def update(self, activations, save_raw_mask=False):
        self.activations = array([1.0, 0.0, 0.0, 0.0, 0.0])

def build_sensory(parameters_filename, spatial_attention):
    params = json.load(open(parameters_filename))
    list_of_layers = params["network"]["layers_to_use"]

    sensory_system = {}
    for i in range(len(list_of_layers)-2):
        sensory_system[list_of_layers[i+1]] = AlexNet(list_of_layers[i], list_of_layers[i+1],
                                                      spatial_attention=spatial_attention,
                                                      params = dict(params))

    if "input" in list_of_layers:
        sensory_system["input"] = IdentityNet()
    if "context" in list_of_layers:
        sensory_system["context"] = ContextNet()

    return sensory_system
