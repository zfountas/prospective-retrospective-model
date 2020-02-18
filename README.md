# A deep predictive processing model of episodic memory and time perception

Computational model and analysis scripts for the manuscript:

Z. Fountas, A. Sylaidi, K. Nikiforou, A. Seth, M. Shanahan, W. Roseboom, "[A predictive processing model of episodic memory and time perception](https://www.biorxiv.org/content/10.1101/2020.02.17.953133v1 "In bioRxiv")
", bioRxiv 2020.02.17.953133; doi: https://doi.org/10.1101/2020.02.17.953133

---

### Requirements

Python 3, tensorflow >= 1.12.0, matplotlib, scipy, sklearn

### Scripts to run experiments in the sequence presented in the paper
- ```python step1.py``` Runs the model for a short episode in each video (frames 1 to XXX) to train its semantic memory. It creates the file ```semantic_memory.pkl```.
- ```python step2.py python timestorm_dataset```  Generates the timeseries of surprises for each trial. The output files go to the folder ```/trials_surpr```.
- ```python step3_save_av_children.py``` Runs all trials to save the number of average children stored in episodic memory. It creates the file ```all_av_children.pkl```.
 - ```python step4_accumulators_fig_different_att_only_used_trials.py``` Produces a Block-like figure purely based on the accumulators. It uses different attention levels for prospective and retrospective and different effort for high and low cognitive load.
 - ```python step5_linear.py``` Uses the file produced in step 4 to produce the final figures with model reports using linearSVR.


# License
This project is licensed under the GLUv3 License - see the LICENSE file for details.
