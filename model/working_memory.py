# -*- coding: utf-8 -*-
"""
Here we add accummulators etc

TODO: UPDATE THE DESCRIPTION...

This file reads a timeseries of distances between snapshots of alexnet activation
(one timeseries for each layer for each trial) and calculates accumulators for
each layer.
"""

import json
import numpy as np

class WorkingMemory:
    def __init__(self, parameters_filename):
        params = json.load(open(parameters_filename))
        self.accummulators = []
        for i in range(len(params["network"]["layers_to_use"])):
            # It gets initialized with 1 because the episodic memory get's
            # initialized with 1 node too..!
            self.accummulators.append(1.0)


    def update(self, new_salient_features, n):
        if new_salient_features > 0.0:
            self.accummulators[n] += 1.0 # TO THINK: add PRED_ERROR instead of 1? ;)
            print("WM: LAYER",n,"BECOMING",self.accummulators[n], self.accummulators)

    def update_from_episodic_memory(self, tree=[]):
        # If accummulators were used for something else, just completely reset them

        # Count how many elements exist in each layer of the recalled tree and
        # assign this value to the corresponding accummulator
        for L in range(len(tree)):
            self.accummulators[L] = float(len(tree[L]))

    def to_string(self):
        return ",".join([str(x) for x in self.accummulators])

    def reset(self):
        for i in range(len(self.accummulators)):
            # It gets initialized with 1 because the episodic memory get's
            # initialized with 1 node too..!
            self.accummulators = [1.0] * len(self.accummulators)
