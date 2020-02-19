#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This script uses the file produced in step 4 to produce the final figures
with model reports using linearSVR.

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

import matplotlib.pyplot as plt
import numpy as np
import pickle,csv

all_experiment_types = ['prosp_low', 'prosp_high', 'retro_low', 'retro_high']

print('---- LOADING VIDEO TRIAL SEQUENCE ----')
video_scene_types = {}
for i in [1,2,3,4,5,16,17,18, 19]: video_scene_types[i] = 'city'
for i in [6,7,8,9,10,11,12,13]:    video_scene_types[i] = 'campus_outside'
for i in [14,15]:                  video_scene_types[i] = 'office_cafe'
for i in range(20,34):             video_scene_types[i] = 'office_cafe'
video_trial_seq_keys = ['Trial','Video','Test_duration_secs','Test_duration_frames','Start_frame','End_frame']
video_trial_seq_temp = csv.reader(open('Video_trial_sequence.csv'))
video_trial_seq_temp = [row for row in video_trial_seq_temp]
if video_trial_seq_keys != video_trial_seq_temp[0]: exit()
else: video_trial_seq_temp = video_trial_seq_temp[1:]
video_ids = {}
scene_type_per_trial = {}
video_durations = {}
for i in range(len(video_trial_seq_temp)):
    video_ids[int(float(video_trial_seq_temp[i][0]))] = int(float(video_trial_seq_temp[i][1])) # Just for making sure this works..!
    scene_type_per_trial[int(float(video_trial_seq_temp[i][0]))] = video_scene_types[int(float(video_trial_seq_temp[i][1]))]
    video_durations[int(float(video_trial_seq_temp[i][0]))] = float(video_trial_seq_temp[i][2]) # Just for double-checking..!
all_scene_types = ['city', 'campus_outside', 'office_cafe']

with open("model_accumulators_dataset.pkl", "rb") as file:
    data = pickle.load(file)

print([x + ': ' + str(np.shape(np.array(data[x]))) for x in data.keys()])

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from scipy import stats

#Y = np.array(data['durations'])
XX = {'prosp_low' :np.array(data['prosp_low']).transpose(),
      'prosp_high':np.array(data['prosp_high']).transpose(),
      'retro_low' :np.array(data['retro_low']).transpose(),
      'retro_high':np.array(data['retro_high']).transpose()}
YY = {'prosp_low' :np.array(data['durations']['prosp_low']).transpose(),
      'prosp_high':np.array(data['durations']['prosp_high']).transpose(),
      'retro_low' :np.array(data['durations']['retro_low']).transpose(),
      'retro_high':np.array(data['durations']['retro_high']).transpose()}
TRIALS = {'prosp_low' :np.array(data['trial_no']['prosp_low']).transpose(),
          'prosp_high':np.array(data['trial_no']['prosp_high']).transpose(),
          'retro_low' :np.array(data['trial_no']['retro_low']).transpose(),
          'retro_high':np.array(data['trial_no']['retro_high']).transpose()}

Xtrain = {'city':[], 'office_cafe':[], 'campus_outside':[]}
Ytrain = {'city':[], 'office_cafe':[], 'campus_outside':[]}
minimum_length = min([len(YY[x]) for x in ['prosp_low', 'prosp_high']])
for i,mode in enumerate(['prosp_low', 'prosp_high']):
    for i in range(minimum_length): #len(XX[mode])):
        trial = TRIALS[mode][i]
        scene_type = scene_type_per_trial[trial]
        Xtrain[scene_type].append(XX[mode][i])
        Ytrain[scene_type].append(YY[mode][i])

xx = np.array([1.0,1.5,2.0,3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0])

while True:
    SEED = np.random.randint(1,100)
    SEED = 27 #awesome!
    np.random.seed(SEED)

    alg_all = {'city':LinearSVR(), 'office_cafe':LinearSVR(), 'campus_outside':LinearSVR()}

    for scene_type in all_scene_types:
        alg_all[scene_type].fit(Xtrain[scene_type],Ytrain[scene_type])

    x_overall = {'prosp_low':[], 'prosp_high':[], 'retro_low':[], 'retro_high':[]}
    y_overall = {'prosp_low':[], 'prosp_high':[], 'retro_low':[], 'retro_high':[]}
    e_overall = {'prosp_low':[], 'prosp_high':[], 'retro_low':[], 'retro_high':[]}

    for mode in ['prosp_low', 'prosp_high', 'retro_low', 'retro_high']:

        for scene_type in all_scene_types:
            X = []
            Y = []
            for x in range(len(XX[mode])):
                trial = TRIALS[mode][x]
                if scene_type == scene_type_per_trial[trial]:
                    X.append(XX[mode][x])
                    Y.append(YY[mode][x])
            X = np.array(X)
            Y = np.array(Y)
            Yest = alg_all[scene_type].predict(X)
            for y in range(len(Yest)):
                x_overall[mode].append(Y[y])
                y_overall[mode].append(Yest[y])
                e_overall[mode].append((Yest[y]-Y[y])/Y[y])

    # Now x,y,e_overall have the results for all scene_types!
    # So scene doesn't need to be used from now on..
    plt.figure(figsize=(16,6))
    for m,mode in enumerate(['prosp_low', 'prosp_high', 'retro_low', 'retro_high']):
        Y_binned = {}
        for y in range(len(x_overall[mode])):
            if x_overall[mode][y] not in Y_binned:
                Y_binned[x_overall[mode][y]] = []
            Y_binned[x_overall[mode][y]].append(y_overall[mode][y])
        y_mean = np.array([np.mean(Y_binned[k]) for k in xx])
        y_med = np.array([np.median(Y_binned[k]) for k in xx])
        y_std = np.array([np.std(Y_binned[k]) for k in xx]) #y_sem = np.array([stats.sem(Y_binned[k]) for k in xx])
        print('We ended up having:',sum([len(Y_binned[x]) for x in xx]),'datapoints for', mode)

        plt.subplot(2,5,1+m)
        plt.scatter(x_overall[mode][:],y_overall[mode],c='gray')
        plt.plot(xx,y_mean,'k')
        plt.plot(xx,y_med,'r')
        plt.plot(xx,y_mean+y_std,'gray')
        plt.plot(xx,y_mean-y_std,'gray')
        plt.xlim(1,65)
        plt.ylim(1,65)
        plt.plot([1,64],[1,64],'k--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(mode+' ')

    plt.subplot(2,5,5)
    plt.plot([1.10, 1.10],[np.mean(e_overall['retro_low'])-stats.sem(e_overall['retro_low']), np.mean(e_overall['retro_low'])+stats.sem(e_overall['retro_low'])],c='k',lw=1.0)
    plt.plot([2.10, 2.10],[np.mean(e_overall['retro_high'])-stats.sem(e_overall['retro_high']), np.mean(e_overall['retro_high'])+stats.sem(e_overall['retro_high'])],c='k',lw=1.0)
    plt.plot([1.10,2.10], [np.mean(e_overall['retro_low']), np.mean(e_overall['retro_high'])],'k--',label='Retrospective')
    plt.scatter([1.10,2.10], [np.mean(e_overall['retro_low']), np.mean(e_overall['retro_high'])],c='k')
    plt.plot([1, 1],[np.mean(e_overall['prosp_low'])-stats.sem(e_overall['prosp_low']), np.mean(e_overall['prosp_low'])+stats.sem(e_overall['prosp_low'])],c='k',lw=1.0)
    plt.plot([2, 2],[np.mean(e_overall['prosp_high'])-stats.sem(e_overall['prosp_high']), np.mean(e_overall['prosp_high'])+stats.sem(e_overall['prosp_high'])],c='k',lw=1.0)
    plt.plot([1,2], [np.mean(e_overall['prosp_low']), np.mean(e_overall['prosp_high'])],'k',label='Prospective')
    plt.scatter([1,2], [np.mean(e_overall['prosp_low']), np.mean(e_overall['prosp_high'])],c='k')
    plt.xlim(0.5,2.5)
    plt.xticks([],[])

    print('SEED',SEED)
    for scene_type in all_scene_types:
        print('Parameters for', scene_type, alg_all[scene_type].get_params(),'weights:',alg_all[scene_type].coef_)

    print('OK')
    plt.tight_layout()
    plt.savefig('figures/linear_1_only_pro_tr.png')
    plt.savefig('figures/linear_1_only_pro_tr.svg')
    plt.show()
    exit()












#
