#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This file provides an attention mechanism, implemented as a stochastic
threshold, that is able to detect salient events in a given timeseries.

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

import numpy as np
import json

class AttentionSystem:

    def __init__(self, parameter_file):
        self.reset(parameter_file)

    def reset(self, parameter_file):
        # Load attention parameters
        params = json.load(open(parameter_file))
        self.N = len(params['network']['layers_to_use'])
        self.diff = params['attention']['max']-params['attention']['min']
        self.Max = params['attention']['max']
        self.tau = params['attention']['tau']
        self.r_walk_mu = params['attention']['r_walk_mu']
        self.r_walk_sigma = params['attention']['r_walk_sigma']
        self.surprises = []
        self.last_t = []
        self.thresholds = []
        self.time = []

        for i in range(self.N):
            self.surprises.append(0.0)
            self.last_t.append(0.0)
            self.thresholds.append(0.0)
            self.time.append(0.0)

    # Use the attention mechanism to calculate thresholds
    # NOTE: This method does not reset threshold
    def update(self, new_surprise, L):
        self.surprises[L] = 0.0

        # Calculate attention threshold
        D = np.abs(self.time[L] - self.last_t[L])

        self.thresholds[L] -= (self.diff/self.tau)*np.exp(-D/self.tau)
        self.thresholds[L] += np.random.normal(self.r_walk_mu, self.r_walk_sigma)

        # Reset feature accumulators
        if  new_surprise >= self.thresholds[L] :
            self.last_t[L] = self.time[L]
            self.thresholds[L] = self.Max
            # By doing this we can pass the suprise (or prediction erro error)
            # of the layers where a threshold was passed!
            self.surprises[L] = new_surprise

        self.time[L] += 1.0

# For unit tests
if __name__ == "__main__":
    ac = attention_system()
    distances = []
    accumulators, thresholds = ac.calculate(distances=distances)
    print ("Acc", accumulators)
    import matplotlib.pyplot as plt
    plt.subplot(414)
    plt.plot(thresholds[0], c='g')
    plt.plot(distances[0], c='b')
    plt.subplot(413)
    plt.plot(thresholds[1], c='g')
    plt.plot(distances[1], c='b')
    plt.subplot(412)
    plt.plot(thresholds[2], c='g')
    plt.plot(distances[2], c='b')
    plt.subplot(411)
    plt.plot(thresholds[3], c='g')
    plt.plot(distances[3], c='b')
    plt.show()
