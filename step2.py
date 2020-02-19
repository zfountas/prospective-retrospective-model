#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to generate the timeseries of surprises for each trial.

The model runs for a short episode in each video (frames 1 to XXX) to train its
semantic memory.
Runs with: python step2.py python timestorm_dataset

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


import os, csv
from subprocess import call
from sys import argv

semantic_folder = '../../www/timestorm_dataset/semantic_memory.pkl'

if len(argv) != 5:                            exit('Error: Give right number of arguments!')
if not os.path.exists(argv[2]):               exit('Error: Dataset path '+argv[2]+' does not exist!')
if not os.path.exists(semantic_folder): exit('Error: semantic_memory.pkl does not exist yet!')

FIRST = int(argv[3])
LAST = int(argv[4])

python_execution_command = argv[1] # 'py', 'python' or 'python3'...
dataset_path = argv[2]+'/'
for i in range(1,34):
    filename = dataset_path+'video_'+str(i)+'.mp4'
    if not os.path.exists(filename): exit('Error: File '+filename+' does not exist!!')

if not os.path.exists("trials_em"):
    os.makedirs("trials_em")
if not os.path.exists("trials_surpr"):
    os.makedirs("trials_surpr")

listan_me_ta_labels = ['Trial', 'Video', 'Test_duration_secs', 'Test_duration_frames', 'Start_frame', 'End_frame']
VTS = {'Trial':[], 'Video':[], 'Test_duration_secs':[], 'Test_duration_frames':[], 'Start_frame':[], 'End_frame':[]}
filename_vts = dataset_path+'Video_trial_sequence.csv'
with open(filename_vts, mode='r') as ff:
    reader = csv.reader(ff)
    for i,row in enumerate(reader):
        if i > 0:
            for j in range(len(row)):
                if j != 2: VTS[listan_me_ta_labels[j]].append(int(float(row[j])))
                else:      VTS[listan_me_ta_labels[j]].append(row[j])

print([k+' -> '+str(len(VTS[k])) for k in VTS.keys()])
print('')

for i in range(FIRST, LAST):
    print([k+' -> '+str(VTS[k][i]) for k in VTS.keys()])
    filename = dataset_path+'video_'+str(VTS['Video'][i])+'.mp4'
    first_frame = VTS['Start_frame'][i]
    last_frame = VTS['End_frame'][i]
    trial = int(VTS['Trial'][i])
    duration = int(float(VTS['Test_duration_secs'][i])*30)
    if duration != (last_frame - first_frame):
        exit('Error! duration ' + str(duration)+' is not equal to '+str(last_frame - first_frame))
    command = [python_execution_command, 'run.py', '-i', filename,'-p','parameters.json',
          '-ff', str(first_frame),'-lf',str(last_frame),'-sm', semantic_folder, '-em',
          'trials_em/ep_mem_'+str(trial)+'.pkl','-sm_nsm','-ss','trials_surpr/surprises_'+str(trial)+'.pkl']
    print(command)
    call(command)




#
