#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This file is used to create real time visualizations based on openCV.

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

#import matplotlib
#matplotlib.use('TkAgg')
import numpy as np
import cv2
import matplotlib.pyplot as plt

class PlotEngine():

    def __init__(self):
        self.fig = plt.figure(figsize=(16,9))

        max_y = 1.2
        plt.subplot(421)
        self.line1, = plt.plot(np.zeros(100), np.array(range(100)), lw=2.0)
        self.line1t, = plt.plot(np.zeros(100), np.array(range(100)), c='g', lw=2.0)
        plt.ylim(0.0,max_y)
        plt.xlim(0.0,100.0)

        plt.subplot(423)
        self.line2, = plt.plot(np.zeros(100), np.array(range(100)), lw=2.0)
        self.line2t, = plt.plot(np.zeros(100), np.array(range(100)), c='g', lw=2.0)
        plt.ylim(0.0,max_y)
        plt.xlim(0.0,100.0)

        plt.subplot(425)
        self.line3, = plt.plot(np.zeros(100), np.array(range(100)), lw=2.0)
        self.line3t, = plt.plot(np.zeros(100), np.array(range(100)), c='g', lw=2.0)
        plt.ylim(0.0,max_y)
        plt.xlim(0.0,100.0)

        plt.subplot(427)
        self.line4, = plt.plot(np.zeros(100), np.array(range(100)), lw=2.0)
        self.line4t, = plt.plot(np.zeros(100), np.array(range(100)), c='g', lw=2.0)
        plt.ylim(0.0,max_y)
        plt.xlim(0.0,100.0)

        max_y = 100.0

        plt.subplot(422)
        self.line1a, = plt.plot(np.zeros(100), np.array(range(100)), lw=2.0)
        plt.ylim(0.0,max_y)
        plt.xlim(0.0,100.0)

        plt.subplot(424)
        self.line2a, = plt.plot(np.zeros(100), np.array(range(100)), lw=2.0)
        plt.ylim(0.0,max_y)
        plt.xlim(0.0,100.0)

        plt.subplot(426)
        self.line3a, = plt.plot(np.zeros(100), np.array(range(100)), lw=2.0)
        plt.ylim(0.0,max_y)
        plt.xlim(0.0,100.0)

        plt.subplot(428)
        self.line4a, = plt.plot(np.zeros(100), np.array(range(100)), lw=2.0)
        plt.ylim(0.0,max_y)
        plt.xlim(0.0,100.0)


        self.surprise_curves = []
        self.threshold_curves = []
        self.accummulator_curves = []

    def reset(self):
        self.surprise_curves = []
        self.threshold_curves = []
        self.accummulator_curves = []

    def update(self, surprise, thresholds, accummulators):
        # update data
        if self.surprise_curves == []:
            for i in range(len(surprise)):
                self.surprise_curves.append([])
            for i in range(len(thresholds)):
                self.threshold_curves.append([])
            for i in range(len(accummulators)):
                self.accummulator_curves.append([])
        for i in range(len(surprise)):
            self.surprise_curves[i].append(surprise[i])
        for i in range(len(thresholds)):
            self.threshold_curves[i].append(thresholds[i])
        for i in range(len(accummulators)):
            self.accummulator_curves[i].append(accummulators[i])

        # update plots
        if len(self.surprise_curves[7]) > 100:
            self.line1.set_ydata( np.array(self.surprise_curves[7][-100:]) )
            self.line1t.set_ydata( np.array(self.threshold_curves[7][-100:]) )
            self.line1a.set_ydata( np.array(self.accummulator_curves[7][-100:]) )
        else:
            self.line1.set_xdata( np.array(range(len(self.surprise_curves[7]))) )
            self.line1.set_ydata( np.array(self.surprise_curves[7]) )
            self.line1t.set_xdata( np.array(range(len(self.threshold_curves[7]))) )
            self.line1t.set_ydata( np.array(self.threshold_curves[7]) )
            self.line1a.set_xdata( np.array(range(len(self.accummulator_curves[7]))) )
            self.line1a.set_ydata( np.array(self.accummulator_curves[7]) )

        if len(self.surprise_curves[5]) > 100:
            self.line2.set_ydata( np.array(self.surprise_curves[5][-100:]) )
            self.line2t.set_ydata( np.array(self.threshold_curves[5][-100:]) )
            self.line2a.set_ydata( np.array(self.accummulator_curves[5][-100:]) )
        else:
            self.line2.set_xdata( np.array(range(len(self.surprise_curves[5]))) )
            self.line2.set_ydata( np.array(self.surprise_curves[5]) )
            self.line2t.set_xdata( np.array(range(len(self.threshold_curves[5]))) )
            self.line2t.set_ydata( np.array(self.threshold_curves[5]) )
            self.line2a.set_xdata( np.array(range(len(self.accummulator_curves[5]))) )
            self.line2a.set_ydata( np.array(self.accummulator_curves[5]) )

        if len(self.surprise_curves[3]) > 100:
            self.line3.set_ydata( np.array(self.surprise_curves[3][-100:]) )
            self.line3t.set_ydata( np.array(self.threshold_curves[3][-100:]) )
            self.line3a.set_ydata( np.array(self.accummulator_curves[3][-100:]) )
        else:
            self.line3.set_xdata( np.array(range(len(self.surprise_curves[3]))) )
            self.line3.set_ydata( np.array(self.surprise_curves[3]) )
            self.line3t.set_xdata( np.array(range(len(self.threshold_curves[3]))) )
            self.line3t.set_ydata( np.array(self.threshold_curves[3]) )
            self.line3a.set_xdata( np.array(range(len(self.accummulator_curves[3]))) )
            self.line3a.set_ydata( np.array(self.accummulator_curves[3]) )

        if len(self.surprise_curves[1]) > 100:
            self.line4.set_ydata( np.array(self.surprise_curves[1][-100:]) )
            self.line4t.set_ydata( np.array(self.threshold_curves[1][-100:]) )
            self.line4a.set_ydata( np.array(self.accummulator_curves[1][-100:]) )
        else:
            self.line4.set_xdata( np.array(range(len(self.surprise_curves[1]))) )
            self.line4.set_ydata( np.array(self.surprise_curves[1]) )
            self.line4t.set_xdata( np.array(range(len(self.threshold_curves[1]))) )
            self.line4t.set_ydata( np.array(self.threshold_curves[1]) )
            self.line4a.set_xdata( np.array(range(len(self.accummulator_curves[1]))) )
            self.line4a.set_ydata( np.array(self.accummulator_curves[1]) )

        # redraw the canvas
        self.fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        cv2.imshow("plot",img)
