#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This is the default script to run a demo of the model.

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

# Arguments
import argparse
parser = argparse.ArgumentParser(description='Script to run Jikan Nashi for production of prospective/retrospective data.')
parser.add_argument('-i', '--input', type=str, default='', dest='video_input', help='Path to the folder that contains the video to use as input. If digit, it defines the id of the webcam that will be used instead. NOTE: This field cannot be empty!')
parser.add_argument('-ff', '--first_frame', type=int, default=0, dest='first_frame', help='First frame of the video..')
parser.add_argument('-lf', '--last_frame', type=int, default=0, dest='last_frame', help='Last frame of the video..')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_const', const=True, help='Print more things..')
parser.add_argument('-sm', '--sm_filename', type=str, default=None, dest='sm_filename', help='The path (or filename) of a pickle file that contains a previously saved semantic memory. If file doesn\'t exist new memory will be saved!')
parser.add_argument('-smr', '--sm_reuse', action='store_const', const=True, default=False, dest='sm_reuse', help='If a file is detected, are we going to rewrite it?')
parser.add_argument('-sm_nsm', '--sm_no_split_merge', action='store_const', const=True, default=False, dest='sm_no_split_merge', help='Disabling splitting and merging of components in semantic memory')
parser.add_argument('-ss', '--surprises_filename', type=str, default=None, dest='surprises_filename', help='Save surprises (ss) under this filename..')
parser.add_argument('-em', '--em_filename', type=str, default=None, dest='em_filename', help='The path (or filename) of a pickle file that contains a previously saved episodic memory.')
parser.add_argument('-p', '--parameters_filename', type=str, default='parameters.json', dest='parameters_filename', help='The path (or filename) of a json file that contains all parameters of the current simulation.')
# Visualizations
parser.add_argument('--plots', dest='rt_plots', action='store_const', const=True, default=False,help='Enable real time plots.')
parser.add_argument('--graphs', dest='rt_graphs', action='store_const', const=True, default=False,help='Enable real time graphs.')
parser.add_argument('--show_p0', dest='show_p0', action='store_const', const=True, default=False, help='Enable real time plots of priors in layer 0.')
parser.add_argument('--show_p0_single', dest='show_p0_single', action='store_const', const=True, default=False, help='Enable real time plots of (only) the selected prior in layer 0.')
parser.add_argument('--show_p', dest='show_p', type=int, default=-1, help='Enable real time graphs of priors in a given layer.')
parser.add_argument('--timing', dest='timing', action='store_const', const=True, help='Debugging feature: Timing each component of the network to figure out where is the bottleneck!')
parser.add_argument('--show_episodic', dest='show_episodic', action='store_const', const=True, help='Visualize episodic memory!')
parser.add_argument('--show_episodic_L0', dest='show_episodic_L0', action='store_const', const=True, help='Visualize episodic memory in layer 0!')
parser.add_argument('--absolutely_no_plots', dest='absolutely_no_plots', action='store_const', const=True, help='Disable plots even in the cases that by default are used!')
parser.add_argument('--no_debugging_figs', dest='no_debugging_figs', action='store_const', const=True, help='Disable saving all stats of the experiment in the debugging figs folder, mainly to save up space..')
# Features
parser.add_argument('--episodic_recall', dest='episodic_recall', action='store_const', const=True, help='Activate feature: Episodic memory recall')
parser.add_argument('--spatial_attention', dest='spatial_attention', action='store_const', const=True, help='Activate feature: Spatial attention in the system.')
args = parser.parse_args()

import matplotlib as mpl
import numpy as np
import cv2, time, json, pickle, shutil
from os import path, makedirs
import os

if not args.rt_plots:   mpl.use('Agg')
else:                   mpl.use('TkAgg')
if os.name == 'nt':     separator = '\\'
else:                   separator = '/'

if args.video_input.isdigit():
    DEVICE = int(args.video_input)
    video_capture = cv2.VideoCapture(DEVICE)
    print('\nCapturing video from device',DEVICE,'\n')
else:
    video_capture = cv2.VideoCapture(args.video_input)

ff = args.first_frame
all_frame_counter = -1
while ff >= 0:
    ff -= 1
    all_frame_counter += 1
    success, frame = video_capture.read()

from model.attention_system import *
from model.semantic_memory import *
from model.episodic_memory import *
from model.working_memory import *
from model.sensory_system2 import *
from model.predictive_coding_system import *
from model.plot_engine import *
from model.build_sensory import build_sensory

if args.sm_filename is not None and os.path.isfile(args.sm_filename):
    semantic_memory = pickle.load(open(args.sm_filename,'rb'))
else:
    semantic_memory = SemanticMemory(None, args.parameters_filename)
sensory_system = build_sensory(args.parameters_filename, spatial_attention = args.spatial_attention)
attention = AttentionSystem(args.parameters_filename)
episodic_memory = EpisodicMemory(args.parameters_filename)
working_memory = WorkingMemory(args.parameters_filename)
working_memory_retrospective = WorkingMemory(args.parameters_filename)
pred_coding = PredictiveCodingSystem(args.parameters_filename, episodic_recall = args.episodic_recall, spatial_attention = args.spatial_attention)

semantic_memory.printing_priors = args.show_p
if args.sm_no_split_merge:
    semantic_memory.merging_disabled = True
    semantic_memory.splitting_disabled = True

if args.surprises_filename != None: all_surprises = []

if not path.exists("recalls"): makedirs("recalls")

slow_down = 1; show_details = True; DEMO_RESOLUTION = 450; DEMO_RESOLUTION_PRIORS = 200; NET_RESOLUTION = 227
font = cv2.FONT_HERSHEY_SIMPLEX; font_size = 0.8; fps = 0; time_secs = 0.0; show_details = True; print_details = False; time_in_sec = 0.0
overall_start_time = 0; INITIALISE_TREE = True
layer0 = []
layer0_mask = []
FPS = [] # Keep fps of last 10 secs and average it to show real fps

if args.rt_graphs: plot_engine = PlotEngine()

params = json.load(open(args.parameters_filename))
list_of_layers_to_use = params["network"]["layers_to_use"]

if not args.no_debugging_figs:
    debugging_figs_folder = 'debugging_figs/'
    s_index = 0
    while path.exists(debugging_figs_folder+str(s_index)):
        s_index += 1
    makedirs(debugging_figs_folder+str(s_index))
    makedirs(debugging_figs_folder+str(s_index)+"/frames")
    makedirs(debugging_figs_folder+str(s_index)+"/recalls")
    print('Folder to be used:',debugging_figs_folder+str(s_index))

frame_counter = -1
while success:
    frame_counter += 1
    all_frame_counter += 1
    if args.verbose: print("----------- Frame:",frame_counter, all_frame_counter,"-----------")
    print('Frame:',args.video_input, frame_counter, all_frame_counter, str(int(100.0*frame_counter/(args.last_frame-args.first_frame)))+'% done' )

    frame_to_show = cv2.resize(np.copy(frame), (DEMO_RESOLUTION, DEMO_RESOLUTION))
    frame_to_show_priors = cv2.resize(np.copy(frame), (DEMO_RESOLUTION_PRIORS, DEMO_RESOLUTION_PRIORS))
    frame = cv2.resize(frame, (NET_RESOLUTION,NET_RESOLUTION))
    input_frame = np.copy(frame)

    # Initialize the model during the first time we get a new frame.
    if INITIALISE_TREE:
        list_of_activations = []
        list_of_activations.append(np.copy(input_frame))
        print('A', 0, np.shape(sensory_system[list_of_layers_to_use[0]].activations),np.mean(sensory_system[list_of_layers_to_use[0]].activations))
        for layer in range(1, len(list_of_layers_to_use)):
            sensory_system[list_of_layers_to_use[layer]].update(input_frame)
            print('A', layer, np.shape(sensory_system[list_of_layers_to_use[layer]].activations),np.mean(sensory_system[list_of_layers_to_use[layer]].activations))
            input_frame = np.copy(np.squeeze(sensory_system[list_of_layers_to_use[layer]].activations))
            list_of_activations.append(np.copy(np.squeeze(sensory_system[list_of_layers_to_use[layer]].activations)))
        episodic_memory.initialise_tree(list_of_activations)
        if args.sm_filename == None or not os.path.isfile(args.sm_filename):
            semantic_memory.initialise_memory(list_of_activations)
        input_frame = np.copy(frame)

        # Initialize windows!
        if args.rt_plots:
            cv2.imshow('Raw input', frame_to_show)
            cv2.moveWindow('Raw input',1620-DEMO_RESOLUTION-5, 0)
            cv2.imshow('Activation of first layer', frame_to_show)
            cv2.moveWindow('Activation of first layer', 1620-2*DEMO_RESOLUTION-5, 0)
            if args.spatial_attention:
                cv2.imshow('Attention', frame_to_show)
                cv2.moveWindow('Attention', 1620-2*DEMO_RESOLUTION-5, DEMO_RESOLUTION+100)
        if args.show_p0:
            cv2.imshow('Priors L0', frame_to_show_priors)
            cv2.moveWindow('Priors L0', 100, 0)
        if args.show_p0_single:
            cv2.imshow('Selected Mslow (L0)', frame_to_show_priors)
            cv2.moveWindow('Selected Mslow (L0)', 100, 0)
        if args.show_episodic:
            cv2.imshow('Episodic', frame_to_show_priors)
            cv2.moveWindow('Episodic', 100, DEMO_RESOLUTION_PRIORS + 100)
        if args.show_episodic_L0:
            cv2.imshow('Episodic L0', frame_to_show_priors)
            cv2.moveWindow('Episodic L0', 100, DEMO_RESOLUTION_PRIORS + 100)

        INITIALISE_TREE = False

    # Iterate through all hierarchical layers and run the model
    for layer in range(len(list_of_layers_to_use)):
        sensory_system[list_of_layers_to_use[layer]].update(input_frame, save_raw_mask=False)
        layer_activation = np.squeeze(sensory_system[list_of_layers_to_use[layer]].activations)

        # Run steps of predictive coding to update both activation patterns and priors..!
        # NOTE: It updates: 'semantic_memory.last_surprise' and 'pred_coding.prev_activations' which now has become the last prior of semantic_memory..
        if layer != (len(list_of_layers_to_use) - 1): attention_mask = sensory_system[list_of_layers_to_use[layer+1]].gradients
        else: attention_mask = np.ones((5))

        did_merge = pred_coding.update(layer_activation, semantic_memory, episodic_memory, layer, attention_mask)

        # Pass latest prediciton to tensorflow
        sensory_system[list_of_layers_to_use[layer]].generative_model_prediction = pred_coding.Mmean[layer]

        # Use the attention mechanism to calculate thresholds
        # NOTE: It updates: 'attention.thresholds' and 'attention.surprises'
        attention.update(semantic_memory.last_surprise[layer], layer)

        # Update episodic memory: If interesting, save the current activation pattern!
        episodic_memory.update(new_novelty = attention.surprises[layer], activation_pattern = semantic_memory.get_last_prior_mean(layer),
                               semantic_memory = semantic_memory, pred_coding = pred_coding, layer = layer)

        # Update accummulators that keep track of
        working_memory.update(attention.surprises[layer], layer)

        # Propagate updated activations (instead of pred. error)
        input_frame = np.copy(np.squeeze(semantic_memory.get_last_prior_mean(layer)))

    if args.surprises_filename != None: all_surprises.append(np.copy(semantic_memory.last_surprise))

    prev = time.time()


    # -- VISUALIZATIONS --------------------------------------------------------
    fps += 1.0
    details = 'FPS: '+str(round(np.mean(FPS),0))
    if print_details:
        print(details)
    if show_details :
        cv2.putText(frame_to_show, details, (20,DEMO_RESOLUTION-10), font, font_size, (255,255,255),1)
        cv2.putText(frame_to_show, sensory_system[list_of_layers_to_use[-2]].best_label, (20,DEMO_RESOLUTION-40), font, font_size,(255,255,255),2)
        AA = 'Time: ' + str(round( float(frame_counter-overall_start_time)/30.0, 1))
        ##AA += ' Estim: ' + str(round(t/params.test['FPS']+4.0*np.random.rand()-2.0,2))
        ##AA += ' +- ' + str(round(upNet.bayes['my_likelihood']['sigma'],2))
        cv2.putText(frame_to_show, AA, (20,DEMO_RESOLUTION-70), font, font_size, (255,255,255), 2)

    if time_secs != round(time.time()):
        FPS.append(fps)
        if len(FPS) > 10:
            FPS = FPS[1:]
        fps = 0.0
    time_secs = round(time.time())

    if args.rt_plots:
        #if args.human_gaze[1].isdigit():
        #    hgl.draw_box(frame_to_show, frame_counter, scale_x=float(DEMO_RESOLUTION), scale_y=float(DEMO_RESOLUTION))
        cv2.imshow('Raw input', frame_to_show)
        # NOTE: cv2.imshow expects values from 0-255 if integer or 0.0-1.0 if float!
        frame_to_show2 = cv2.resize(np.copy(semantic_memory.get_last_prior_mean(0)/255.0), (DEMO_RESOLUTION,DEMO_RESOLUTION))
        layer0.append(np.copy(semantic_memory.get_last_prior_mean(0)))
        cv2.imshow('Activation of first layer', frame_to_show2)

        if len(np.shape(pred_coding.attention_mask[0])) == 3:
            frame_to_show4 = np.copy(pred_coding.attention_mask[0])
            frame_to_show4 = cv2.resize(frame_to_show4, (DEMO_RESOLUTION,DEMO_RESOLUTION))
            cv2.putText(frame_to_show4, "Fm: "+str(round(sensory_system[list_of_layers_to_use[1]].gradients.mean(),2))+\
                        " co " + str(round(sensory_system[list_of_layers_to_use[1]].grad_corrector,1))+\
                        " er " + str(round(sensory_system[list_of_layers_to_use[1]].grad_error,1)), (20,30), font, font_size, (1.0,0.0,0.0), 2)
            cv2.putText(frame_to_show4, "Min: "+str(sensory_system[list_of_layers_to_use[1]].gradients.min()), (20,60), font, font_size, (1.0,0.0,0.0), 2)
            cv2.putText(frame_to_show4, "Max: "+str(sensory_system[list_of_layers_to_use[1]].gradients.max()), (20,90), font, font_size, (1.0,0.0,0.0), 2)
            cv2.putText(frame_to_show4, "Cor: "+str(sensory_system[list_of_layers_to_use[1]].grad_corrector), (20,120), font, font_size, (1.0,0.0,0.0), 2)
            cv2.imshow('Attention', frame_to_show4)
            layer0_mask.append(np.copy(pred_coding.attention_mask[0]))

    if args.show_episodic:
        new_frame = episodic_memory.get_visualization()
        for i in range(len(episodic_memory.step2_tree)):
            layer = len(episodic_memory.all_nodes)-i-1
            for node_index in episodic_memory.step2_tree[layer]:
                new_frame[i,node_index,0] = 0.0
                new_frame[i,node_index,1] = 0.0
                new_frame[i,node_index,2] = 1.0
        new_frame = cv2.resize(new_frame, (500,200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Episodic', new_frame)

    if args.show_episodic_L0:
        frame_to_show_priors = np.zeros((DEMO_RESOLUTION_PRIORS,DEMO_RESOLUTION_PRIORS))
        if len(episodic_memory.all_nodes[0]) > 8:
            start = -9
            end = -1
        else:
            start = 0
            end = len(episodic_memory.all_nodes[0])
        for e_index in range(start, end):
            new_frame = cv2.resize(np.copy(episodic_memory.all_nodes[0][e_index].activation/255.0),
                                   (DEMO_RESOLUTION_PRIORS,DEMO_RESOLUTION_PRIORS))
            if not np.any(frame_to_show_priors): frame_to_show_priors = new_frame
            else: frame_to_show_priors = np.concatenate( (frame_to_show_priors, new_frame), axis=1)
        cv2.imshow('Episodic L0', frame_to_show_priors)


    if args.rt_graphs: plot_engine.update(semantic_memory.last_surprise, attention.thresholds, working_memory.accummulators)

    k = cv2.waitKey(slow_down)
    if k==27:                               break
    elif k==ord('q'):                       break
    elif k==32:                             semantic_memory.run_debugging() # space key to debug
    elif k==ord('p'):                       slow_down += 1000
    elif k==ord('o') and slow_down > 1000:  slow_down -= 1000
    elif k==ord('s'):                       semantic_memory.save()
    elif k == ord('p'):                     args.rt_graphs = not args.rt_graphs # plots
    elif k == ord('d'):                     print_details = not print_details


    if args.last_frame > 0 and args.last_frame <= all_frame_counter:
        break
    success, frame = video_capture.read()


if args.sm_filename != None and args.sm_reuse:
    #semantic_memory.save(args.sm_filename)
    pickle.dump(semantic_memory, open(args.sm_filename,'wb'))

if args.em_filename != None:
    #semantic_memory.save(args.sm_filename)
    for layer in range(len(episodic_memory.all_nodes)):
        for node in range(len(episodic_memory.all_nodes[layer])):
            episodic_memory.all_nodes[layer][node].activation = None
    pickle.dump(episodic_memory, open(args.em_filename,'wb'))

if args.surprises_filename != None:
    pickle.dump(all_surprises, open(args.surprises_filename,'wb'))






































#
