""" -------------------------------- STEP 2 --------------------------------
    The model runs for a short episode in each video (frames 1 to XXX) to train
    its semantic memory.
    OUTCOME:
        ...

    RUN:
    python step2.py python timestorm_dataset
"""
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
dataset_path = argv[2]+'/' # G:\timestorm_dataset
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

for i in range(FIRST, LAST): #(len(VTS['End_frame'])):
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
