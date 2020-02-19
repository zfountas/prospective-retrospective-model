#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This script runs the model for a short episode in each video (frames 1 to
XXX) to train its semantic memory. It creates the file semantic_memory.pkl.

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

import os
from subprocess import call
from sys import argv

if len(argv) != 3:                          exit('Error: Give right number of arguments!')
if not os.path.exists(argv[2]):             exit('Error: Dataset path '+argv[2]+' does not exist!')
if os.path.exists('semantic_memory.pkl'):   exit('Error: semantic_memory.pkl already exists!')

python_execution_command = argv[1] # 'py', 'python' or 'python3'...
dataset_path = argv[2]+'/' # G:\timestorm_dataset
for i in range(1,34):
    filename = dataset_path+'video_'+str(i)+'.mp4'
    if not os.path.exists(filename): exit('Error: File '+filename+' does not exist!!')

#if not os.path.exists("recalls"): os.makedirs("recalls")

for i in range(1, 34):
    filename = dataset_path+'video_'+str(i)+'.mp4'
    first_frame = 0
    last_frame = 200
    call([python_execution_command, 'run.py', '-i', filename,'-p','parameters.json',
          '-ff', str(first_frame),'-lf',str(last_frame),'-sm','semantic_memory.pkl','-smr']) # ,'-v'])
