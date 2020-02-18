""" -------------------------------- STEP 1 --------------------------------
    The model runs for a short episode in each video (frames 1 to XXX) to train
    its semantic memory.
    OUTCOME:
        The file 'semantic_memory.pkl'
"""
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




#
