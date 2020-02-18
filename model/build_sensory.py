"""
This is the method that

"""

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
