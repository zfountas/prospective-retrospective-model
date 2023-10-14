# A deep predictive processing model of episodic memory and time perception

This source code release accompanies the manuscript:

Z. Fountas, A. Sylaidi, K. Nikiforou, A. Seth, M. Shanahan, W. Roseboom, "[A predictive processing model of episodic memory and time perception]([https://www.biorxiv.org/content/10.1101/2020.02.17.953133v1](https://direct.mit.edu/neco/article-abstract/34/7/1501/111336/A-Predictive-Processing-Model-of-Episodic-Memory)" Neural Computation 34.7 (2022): 1501-1544,

which can also be found in a [pre-print](https://www.biorxiv.org/content/10.1101/2020.02.17.953133v1 "In bioRxiv") version.

---

### Requirements

* Programming language: *Python 3*
* Libraries: *tensorflow >= 1.12.0, matplotlib, scipy, sklearn pip PrettyTable networkx*
* Video dataset: ***To be uploaded soon***

### Instructions

###### Install Python dependencies
Assuming you have a working version of Python 3, open a terminal and type:
```bash
sudo pip install tensorflow-gpu==1.15 sklearn matplotlib scipy sklearn PrettyTable networkx
```

###### Run the model using any video or webcam
Assuming you have a video named ```video.mp4```, open a terminal and type:
```bash
python run.py -i video.mp4
```
To run the model using your webcam, open a terminal and simply type:
```bash
python run.py
```

###### Reproduce paper figures
To reproduce the results in the computational experiments presented in the paper, please follow this sequence:

1. To run the model for a short episode in each video (frames 1 to XXX) to train its semantic memory.
```bash
python step1.py
```
This will create the file ```semantic_memory.pkl```.

2. Then, to generate the timeseries of surprises for each trial. The output files go to the folder ```/trials_surpr```.
```bash
python step2.py python timestorm_dataset
```
3. Then we need to run a large number of trials in order to get an estimation of the number of average children stored in episodic memory per layer. To do this, type
```bash
python step3_save_av_children.py
```
This will create a file called ```all_av_children.pkl```.
4. To produce a figure similar to Fig.1 of the manuscript purely based on the accumulators, type
```bash
python step4_accumulators_fig_different_att_only_used_trials.py
```
This uses different attention levels for prospective and retrospective and different effort for high and low cognitive load.
5. Finally, to produce the final figures with model reports based on linear regression, type
```bash
python step5_linear.py
```
This uses the file produced in step 4 and sklearn's linearSVR.

# License
This project is licensed under the GLUv3 License - see the LICENSE file for details.
