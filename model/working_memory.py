#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This file reads a timeseries of distances between snapshots of alexnet
activation (one timeseries for each layer for each trial) and calculates
accumulators for each layer.

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
            self.accummulators[n] += 1.0
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
