# A deep predictive processing model of episodic memory and time perception

Computational model and analysis scripts for the manuscript:

Z. Fountas, A. Sylaidi, K. Nikiforou, A. Seth, M. Shanahan, W. Roseboom, "[A predictive processing model of episodic memory and time perception](https://www.biorxiv.org/content/10.1101/2020.02.17.953133v1 "In bioRxiv")
", bioRxiv 2020.02.17.953133; doi: https://doi.org/10.1101/2020.02.17.953133

---

### Scripts to run in the sequence presented in the paper
- ```./run.py``` This file is a general script to run the complete model.
- ```python step1.py``` Runs the model for a short episode in each video (frames 1 to XXX) to train its semantic memory. It creates the file ```semantic_memory.pkl```.
- ```python step2.py python timestorm_dataset```  Generates the timeseries of surprises for each trial. The output files go to the folder ```/trials_surpr```.
- ```python step3_save_av_children.py``` Runs all trials to save the number of average children stored in episodic memory. It creates the file ```all_av_children.pkl```.
- Step 4:
 - ```python step4.1_optimize_parameters.py``` (*Can be skipped*) Tries to optimize the parameters of the model based on human behaviour. However, **it fails** for reasons explained in the paper!
 - ```python step4.2_accumulators_fig.py``` Produces a Block-like figure purely based on the accumulators.
 - ```python step4.2_accumulators_fig_different_att.py``` Produces a Block-like figure purely based on the accumulators. It uses different attention levels for prospective and retrospective!
 - ```python step4.3_save_accumulators_dataset.py``` Produces a Block-like figure purely based on the accumulators. It uses different attention levels for prospective and retrospective!

For results using extra regression algorithms run:
 - ```python step5.py svm``` Uses the file produced in step 4.3 to produce the final figures with model reports using SVR.
 - ```python step5.py mlp``` Uses the file produced in step 4.3 to produce the final figures with model reports using a multi-layer perceptron.
 - ```python step5.py gp``` Uses the file produced in step 4.3 to produce the final figures with model reports using Gaussian processes.



# License
This project is licensed under the GLUv3 License - see the LICENSE file for details.
