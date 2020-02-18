# -*- coding: utf-8 -*-
"""
This class creates instances of a semantic memory.
It maintains a list of labels and corresponding mean activation patterns (M),
variance of activation patterns (Mvar) anda list (actually python dictionary)
of conditional probabilities (P) for each prior and set of conditions that we've
seen before.

Example of M:
M[3][0] = np.array((0.23, 0.3, 0.56, 0.56, 0.56, 0.56, 0.56))
M[2][0] = np.array((0.2, 0.3, 0.56))
M[2][1] = np.array((0.23, 0.3, 0.56))
M[2][2] = np.array((0.5, 0.9, 0.1))
M[1][0] = np.array((0.2, 0.3))

Example of Mvar:
Mvar[3][0] = 0.4
Mvar[2][0] = 0.1

# Example of P
P[(2,0,2,0)] = 0.2 # P(M^2_0|M^2_2,M^3_0)

TODO: UPDATE THE DESCRIPTION...
"""
from __future__ import print_function
import numpy as np
import pickle
import json
from prettytable import PrettyTable
import os.path
from alexnet.classes import class_names

class SemanticMemory:


    # Class of prior objects with Gaussian strucutre and short term plasticity
    class Prior:
        def __init__(self, mean, var):

            # Mean act. pattern of the prior
            self.mean = mean.astype(float) # np.array()
            self.mean_long = mean.astype(float) # np.array() # Kalman runs here!

            # Variance of act. pattern of the prior (applied to both means)
            # M[layer number][index of prior].var = variance
            self.var = var # = 0.0

            self.old_vars = [var]

            # Counter that keeps track of all occurences of the priors in order
            # to produce the unconditional probability P(M_i)
            self.Pall = 0.0

            # Keeps track if a prior has been reused (for debugging)
            self.reused = False

            # List with the conditional probabilities
            # NOTE: It's actually the counter of how many times the pattern of
            # this prior has occur instead of the actual probabilities.
            # We can use this to update activation pattern as well..
            # P[ (index of conditional prior at this layer,
            #    index of conditional prior at the next layer ) ] = 53
            # Therefore: P[( e_{n,-1}, a_{n+1,t} )]
            self.P = {}

             # This flag helps to reduce calculations if short & long very close
            self.updating = False

    def __init__(self, memory_filename='', parameters_filename='', parameters_dict={}):

        if parameters_filename != '':
            params = json.load(open(parameters_filename))
        elif parameters_dict != {}:
            params = parameters_dict
        else:
            print("Semantic memory: Parameters not given. Exiting..")
            exit()
        self.list_of_layers_to_use = params["network"]["layers_to_use"]
        self.no_layers = len(self.list_of_layers_to_use)

        # List with the labels and corresponding priors
        self.M = []

        # List to maintain the maximum index that has been saved in the memory.
        # NOTE: This is important because during merging, some priors are lost
        # (merged) so their indexes remain empty. As a result, we cannot add a new
        # prior by checking len(self.M[n]).
        self.nextMindx = []

        # Index of last prior (per layer) used by the prediction system
        self.last_prior_index = []

        self.last_atn = []

        self.last_similarity_with_an = []

        # Indices of the 2 labels required for the conditions of self.P from the
        # last time that a prior was found (per layer)
        self.last_conditions = []

        # To keep track of the average variance in the distances we calculate for
        # each layer. Structure: (variance, counter of samples so far)
        self.av_variance = []

        # -- Kalman parameters --
        # Kalman gain
        self.K = []
        # Process variance
        self.Q = None
        # Estimate of measurement variance
        self.R = None

        self.last_surprise = []

        self.merging_disabled = False
        self.splitting_disabled = False

        # Just for illustration purposes!
        self.last_dist = []
        self.last_Pcon = [] # This is actually now used again
        self.last_Pcon_all = []
        self.all_priors_indx = []
        self.w = {} # How much the current activation pattern affects the selected prior
        self.check_labels = []
        # The maximum similarity that was recorded last time we checked for merging
        self.maximum_sim = []
        self.last_num_priors = []
        self.last_num_reused_priors = []
        self.last_num_merged_priors = []
        self.ratio_of_priors_being_updated = []
        self.printing_priors = -1
        self.debugging = False
        self.fw_for_pred = False

        for i in range(self.no_layers):
            self.M.append({})
            self.nextMindx.append(0)
            self.last_prior_index.append(0)
            self.last_atn.append(np.array([]))
            self.last_similarity_with_an.append(0)
            self.last_conditions.append((-1, -1))
            self.av_variance.append([0.0, 0.0])
            self.last_surprise.append(0.0)
            self.last_dist.append(0.0)
            self.last_Pcon.append(0.0)
            self.last_Pcon_all.append([])
            self.all_priors_indx.append([])
            self.last_num_priors.append(0)
            self.last_num_reused_priors.append(0)
            self.last_num_merged_priors.append(0)
            self.ratio_of_priors_being_updated.append(0.0)
            self.maximum_sim.append(0.0)
            self.K.append(0.0)
        self.last_prior_index.append(0)
        self.last_atn.append(np.array([]))

        self.initialMvar = params['pred_coding']['initialMvar']
        self.new_prior_threshold = params['semantic_memory']['new_prior_threshold']
        self.merging_threshold = params['semantic_memory']['merging_threshold']
        self.Pcont_noise = params['semantic_memory']['Pcont_noise']

        self.k_fast = params['semantic_memory']['k_fast']
        self.k_back_fast = self.k_fast*params['semantic_memory']['k_back_fast_r']
        self.k_back_slow = self.k_back_fast*params['semantic_memory']['k_back_slow_r']
        self.stpThres = params['semantic_memory']['plasticity_threshold']

        self.Q = params['semantic_memory']['kalman_Q']
        self.R = params['semantic_memory']['kalman_R']

        if memory_filename is not None and os.path.isfile(memory_filename):
            self.load(memory_filename)
        #else:
        #    print("Semantic memory starts from scratch!")

    def save(self, filename):
        if filename != None:
            with open(filename,'wb') as ff:
                data_to_save = {}
                data_to_save['list_of_layers'] = self.list_of_layers_to_use
                data_to_save['M'] = self.M
                data_to_save['nextMindx'] = self.nextMindx
                data_to_save['last_prior_index'] = self.last_prior_index
                data_to_save['last_similarity_with_an'] = self.last_similarity_with_an
                data_to_save['last_conditions'] = self.last_conditions
                data_to_save['av_variance'] = self.av_variance
                data_to_save['last_num_reused_priors'] = self.last_num_reused_priors
                data_to_save['last_num_merged_priors'] = self.last_num_merged_priors
                pickle.dump(data_to_save, ff)
            print("Done")

    def load(self, filename):
        if filename != None:
            with open(filename,'rb') as ff:
                data_to_load = pickle.load(ff)
                self.M = data_to_load['M']
                self.nextMindx = data_to_load['nextMindx']
                self.last_prior_index = data_to_load['last_prior_index']
                self.last_similarity_with_an = data_to_load['last_similarity_with_an']
                self.last_conditions = data_to_load['last_conditions']
                self.av_variance = data_to_load['av_variance']
                self.list_of_layers_to_use = data_to_load['list_of_layers']
                self.last_num_reused_priors = data_to_load['last_num_reused_priors']
                self.last_num_merged_priors = data_to_load['last_num_merged_priors']
                self.no_layers = len(self.list_of_layers_to_use)

            for j in range(self.nextMindx[self.no_layers - 2]):
                self.check_labels.append([])
            print("Semantic memory loaded from", filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    # Implementation of the RBF kernel
    def RBFvar(self, x, y, variance, layer):
        dist = np.linalg.norm(x-y)
        self.last_dist[layer] = dist
        return self.RBF_from_dist(dist, variance, layer)

    def RBF_from_dist(self, dist, variance, layer):
        # Iterative average of squared distances, i.e. the variance of the distances
        self.av_variance[layer][0] = self.av_variance[layer][0]*self.av_variance[layer][1] + dist**2.0
        self.av_variance[layer][1] += 1.0
        self.av_variance[layer][0] /= self.av_variance[layer][1]

        # use normalized version of variance
        if self.av_variance[layer][0] == 0.0: return 0.0
        else: return np.exp( - (dist**2.0) / (2.0 * self.av_variance[layer][0]*variance) )

    def add_prior(self, layer, activation_pattern):
        self.M[layer][self.nextMindx[layer]] = self.Prior(activation_pattern, self.initialMvar)
        self.last_prior_index[layer] = self.nextMindx[layer]
        self.nextMindx[layer] += 1
        if layer == self.no_layers - 2:
            self.check_labels.append([])

    # Once
    def initialise_memory(self, activations):
        # Initialization
        # --------------
        # If tree is empty, then add a new element/node to all layers
        # This loop will run only the first time we call update
        for L in range(len(activations)):
            self.add_prior(L, activations[L])
            self.M[L][0].P[(0,0)] = 1.0
            self.M[L][0].Pall = 1.0

    def should_we_make_new(self, n):
        if self.splitting_disabled:
            return False
        if self.new_prior_threshold > self.last_similarity_with_an[n]:
            return True
        return False

    def update_context_probabilities_and_create_new_priors(self, layer, activation_pattern, print_me=True):
        # If prior does not exist so far, create one..
        str_actions = ""
        #if self.last_prior_index[layer] < 0:
        if self.should_we_make_new(layer):
            #print("No prior found in layer",layer,"Let's create one!")
            str_actions += "Create prior"
            self.M[layer][self.last_prior_index[layer]].mean = np.copy(self.M[layer][self.last_prior_index[layer]].mean_long)
            self.add_prior(layer, activation_pattern)

        # Create/update conditional probability associated with this prior..
        cond = self.last_conditions[layer]
        if layer < 0 or self.last_prior_index[layer] < 0:
            print("ERROR in semantic memory!!")
            exit()
        elif cond[0] < 0 or cond[1] < 0:
            str_actions += "NOT SURE :)" + str(cond)
            #print("Not sure what is happening here!! :P")
        elif cond not in self.M[layer][self.last_prior_index[layer]].P:
            if str_actions == "":
                str_actions += " Create prob"
            else:
                str_actions += "/prob"
            #print("No probability found in layer",layer,"Let's create one!")
            self.M[layer][self.last_prior_index[layer]].P[cond] = 1.0
        else:
            str_actions += " Update prob"
            #print("We have a probability!! Layer:",layer, "Let's update it!")
            self.M[layer][self.last_prior_index[layer]].P[cond] += 1.0

        # Create/update unconditional probability of this prior..
        self.M[layer][self.last_prior_index[layer]].Pall += 1.0

        # Print overview of semantic memory..
        if print_me:
            self.print_me(layer, str_actions)
        return str_actions

    def run_debugging(self):
        self.debugging = True

    def print_me(self, layer, str_actions):
        print("\n")
        print(" SEMANTIC MEMORY")

        title = ['Layer', 'No of Ms', 'Re-used', 'Merged',\
                 'Mer. sim', 'Pr.sim', 'Av. variance',\
                 'Last dist', 'Pcont', 'Kalman gain',\
                 'N', 'Last M', 'Updating', 'Actions']
        table = PrettyTable(title)

        for i in range(len(self.M)):
            n = len(self.M)-i-1 # Layer

            self.last_num_priors[n] = len(self.M[n])
            self.last_num_reused_priors[n] = len([1 for x in self.M[n] if self.M[n][x].reused])
            self.last_num_merged_priors[n] = self.nextMindx[n] - len(self.M[n])

            new_row = [n, self.last_num_priors[n]]
            new_row.append(self.last_num_reused_priors[n])
            new_row.append(self.last_num_merged_priors[n])
            new_row.append(round(self.maximum_sim[n],2))    # Maximum similarity
            new_row.append(round(self.last_similarity_with_an[n],2))    # Last recorded max similarity from priors
            new_row.append(round(self.av_variance[n][0],2)) # Average variance
            new_row.append(round(self.last_dist[n],2))
            new_row.append(round(self.last_Pcon[n],2))
            new_row.append(round(self.K[n],5))
            new_row.append(int(self.M[n][self.last_prior_index[n]].Pall))
            new_row.append(self.last_prior_index[n])
            new_row.append(self.ratio_of_priors_being_updated[n])
            if layer == n: new_row.append(str_actions)
            else: new_row.append('-')
            table.add_row(new_row)
        print(table)

        if self.printing_priors >= 0:
            self.print_priors(self.printing_priors)

    def print_priors(self,layer):
        print(' ')
        if len(self.M[layer]) > 0:
            print(' -- PRIORS OF LAYER '+str(layer)+' -- ')
            for p_index in self.M[layer].keys():
                inds = np.argsort(self.M[layer][p_index].mean)
                print(p_index, "\t", class_names[inds[-1]],"|", class_names[inds[-2]][:7], "|", class_names[inds[-3]][:7])

    def get_last_prior_mean(self, layer):
        if self.last_prior_index[layer] >= 0:
            return np.copy(self.M[layer][self.last_prior_index[layer]].mean)
        else:
            return -1

    def get_last_prior(self, layer):
        if self.last_prior_index[layer] >= 0:
            return self.M[layer][self.last_prior_index[layer]]
        else:
            return None

    """
    NOTE: Here we also update our surprise for this timestep!!
          It's important to happen here because here is the first time that
          the semantic memory has access to the current atn (through the error)!
    """
    def update_priors(self, x_tn, n, attention_mask):
        # If there was no prior selected, there is a problem so exit
        if self.last_prior_index[n] < 0:
            print("ERROR: No prior was selected in layer",n,". Prior index:",self.last_prior_index[n])
            exit()

        error_vector = x_tn - self.M[n][self.last_prior_index[n]].mean

        # 0. PROCESS SURPRISE
        # Save this for comparison later to see if we need a new prior..
        self.last_similarity_with_an[n] = self.RBF_from_dist(np.linalg.norm(error_vector),
                                                self.M[n][self.last_prior_index[n]].var, n)

        # Suprise: 1-P[z|z old e old..]
        #self.last_surprise[n] = 1.0 - self.last_similarity_with_an[n] - self.WW[n])*(1.0-self.last_Pcon[n]
        #if False: # As it was with ww = 0
        #    self.last_surprise[n] = 1.0-self.last_Pcon[n]*self.last_similarity_with_an[n]
        #else:
        self.last_surprise[n] = 1.0
        for i in range(len(self.last_Pcon_all[n])):
            #print('aaaaa', np.shape(x_tn), np.shape(self.M[n][self.all_priors_indx[n]].mean))
            #print('bbbbb', x_tn, self.M[n][self.all_priors_indx[n]].mean)
            dist = np.linalg.norm(x_tn - self.M[n][self.all_priors_indx[n][i]].mean)
            similarity = self.RBF_from_dist(dist, self.M[n][self.all_priors_indx[n][i]].var, n)
            self.last_surprise[n] -= self.last_Pcon_all[n][i]*similarity

        if n == len(self.M)-1:
            self.last_surprise[n] = 0.0

        # Process variance
        #Q = 1e-5

        # Estimate of measurement variance
        # How accurate we expect the measurement (in this case atn) to be.
        # Zero would mean zero error while a high number will mean that we don't
        # trust the measurement that much!
        # NOTE: 'atn' is indeed the measurement (Zt) of the KF algorithm (as
        # opposed to using alexnet's incoming activation - atn) because the
        # error here comes from activations[n]-M[n][k] (in KF it's Zt-Xt)
        #R = 0.9**2

        # Nice explanation: https://github.com/akshaychawla/1D-Kalman-Filter

        # NOTE: Kalman Gain does not depend on the state of the system or on the
        # measurements. It depends only from A, H, P0, R, Q, which are all fixed.
        # At a certain point, and under certain conditions, Kalman Gain reaches
        # (most probably in asymptotical sense) an equilibrium point; this means
        # that after a certain amount of time, Kalman Gain should become
        # constant, as covariance matrix P.

        # 1. Bring short- and long- term components of inactive priors closer to each other..
        counter_active = 0.0 # This is defined only for illustration purposes
        for index in self.M[n].keys():

            if index != self.last_prior_index[n]: # If this prior is not currently active
                if self.M[n][index].updating:
                    counter_active += 1.0

                    # Attract each other
                    # Short-term component
                    self.M[n][index].mean += self.k_back_fast*(self.M[n][index].mean_long - self.M[n][index].mean)
                    # Long-term component
                    self.M[n][index].mean_long += self.k_back_slow*(self.M[n][index].mean - self.M[n][index].mean_long)

                    if self.RBFvar(self.M[n][index].mean, self.M[n][index].mean_long, self.M[n][index].var, n) > self.stpThres:
                        self.M[n][index].mean = np.copy(self.M[n][index].mean_long)
                        self.M[n][index].updating = False

        # For illustration purposes..
        self.ratio_of_priors_being_updated[n] = round(100.0*counter_active/len(self.M[n]),0)

        # Otherwise, let's continue..
        index = self.last_prior_index[n]

        # 2. Set the flag 'updating' on to the prior that is currently active
        self.M[n][index].updating = True

        # 3. Apply two forces to the short-term component, one that moves
        #    it towards the sensation and one that moves it towards the long-term prior
        self.M[n][index].mean += self.k_fast*attention_mask*error_vector + self.k_back_fast*(self.M[n][index].mean_long - self.M[n][index].mean)

        # 4. Run Kalman filter for the long-term component to bring it closer to the short-term one
        a_priori_state_m = self.M[n][index].mean_long
        a_priori_state_cov = self.M[n][index].var + self.Q

        # Calculate Kalman gain
        self.K[n] = a_priori_state_cov/(a_priori_state_cov + self.R**2)

        # SOS: This is how it is described in some neuroscience papers
        #mean_long = (1.0 - K[n]) * a_priori_state_m + K[n] * (mean - mean_long)
        self.M[n][index].mean_long = a_priori_state_m + self.K[n] * (self.M[n][index].mean - self.M[n][index].mean_long)
        self.M[n][index].var = (1.0 - self.K[n]) * a_priori_state_cov

        self.M[n][index].old_vars.append(self.M[n][index].var)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def generative_model(self, n, e1n_index):
        conditions = (e1n_index, self.last_prior_index[n+1])
        all_counters_of_conditions = []
        self.all_priors_indx[n] = []
        # For each prior of this layer (n)
        for prior_index in self.M[n].keys():
            # If this prior has occured before under these conditions,
            # retrieve probability (counter)
            noise = np.random.rand()*self.Pcont_noise
            if conditions in self.M[n][prior_index].P:
                # https://www.desmos.com/calculator/hvzzrlm3qy
                """
                p_contx = self.M[n][prior_index].P[conditions]
                if p_contx == 1:
                    all_counters_of_conditions.append(1.0 + noise)
                elif p_contx > 1:
                    all_counters_of_conditions.append(1.0 + np.log(p_contx) + noise)
                else:
                    print("ERROR: What kind of value is this?? Exiting..")
                    exit()
                """
                all_counters_of_conditions.append(np.log(self.M[n][prior_index].P[conditions]) + noise)
            else:
                all_counters_of_conditions.append(noise)     # NOTE: This is just noise!! If counter only 1 in other cases this will be important!
            self.all_priors_indx[n].append(prior_index)

        # Normailize contextual prior probabilities
        self.last_Pcon_all[n] = self.softmax(all_counters_of_conditions)

        # Find the index in memory that gives us the highest probability!
        # SOS NOTE: This is the index of the current list, it DOES NOT correspond
        # to the index (key/label) of the prior. This would be self.all_priors_indx[n][indx]
        indx = np.argmax(self.last_Pcon_all[n])

        # Save this for illustration purposes
        self.last_Pcon[n] = self.last_Pcon_all[n][indx] # This is actually now used again

        if self.debugging and n == 0:
            print("----------------------------LAYER:", n, "------------------")
            print("Current conditions:", conditions)
            print("ALL PROB CONTEXT:",len(all_counters_of_conditions),[x1 for x1 in all_counters_of_conditions if x1 > 0.0])
            #print("P(index,cont,dis):",[ (i1,round(x1,2),round(y1,2)) for i1,x1,y1 in zip(range(len(all_Pcon)),all_Pcon,all_Pdis) if x1 > 0.0 or y1 > 0.0 ])
            #print("Chose:",(self.all_priors_indx[n][indx], all_Pdis[indx], all_Pcon[indx]),"which makes:",all_terms[indx])
            print("Previously chosen k:", self.last_prior_index[n])
            print("Keys of priors in L0:", self.M[n].keys())

            import matplotlib.pyplot as plt
            import cv2
            columns = 6
            if len(self.M[n]) < 8:
                plt.figure(figsize=(16,9))
            else:
                plt.figure(figsize=(16,int(1.5*len(self.M[n]))))
            for i,pindex in enumerate(self.M[n].keys()):
                plt.subplot(len(self.M[n]),columns,1+i*columns)
                plt.imshow(cv2.cvtColor(np.uint8(self.M[n][pindex].mean_long), cv2.COLOR_BGR2RGB))
                plt.xticks([])
                plt.yticks([])
                mylabel = 'M_'+str(i)
                if pindex == self.all_priors_indx[n][indx]:
                    mylabel += " (new k)"
                if pindex == self.last_prior_index[n]:
                    mylabel += " (prev k)"
                plt.ylabel(mylabel)
                plt.subplot(len(self.M[n]),columns,2+i*columns)
                plt.imshow(cv2.cvtColor(np.uint8(self.M[n][pindex].mean), cv2.COLOR_BGR2RGB))
                plt.xticks([])
                plt.yticks([])
                plt.ylabel("var: "+str(round(self.M[n][pindex].var,2)))

                plt.subplot(len(self.M[n]),columns,3+i*columns)
                plt.plot(self.M[n][pindex].old_vars)
                plt.scatter(range(len(self.M[n][pindex].old_vars)), self.M[n][pindex].old_vars)
                plt.xticks([])
                plt.yticks([0.0,1.0])
                plt.ylim(0.0,1.0)

            plt.subplot(len(self.M[n]),columns,1+(len(self.M[n])-1)*columns)
            plt.xlabel('M_slow')
            plt.subplot(len(self.M[n]),columns,2+(len(self.M[n])-1)*columns)
            plt.xlabel('M_fast')

            plt.subplot(3,2,2)
            xs = range(len(self.last_Pcon_all[n]))
            ys = self.last_Pcon_all[n]
            plt.bar(xs, ys)#, 1.0, align='center')
            plt.xticks(xs)
            plt.yticks(ys)
            plt.ylabel('P_context')

            """
            plt.subplot(3,2,4)
            xs = range(len(all_Pdis))
            ys = all_Pdis
            plt.bar(xs, ys)#, 1.0, align='center')
            plt.xticks(xs)
            plt.yticks(ys)
            plt.ylabel('P_distance')

            plt.subplot(3,2,6)
            xs = range(len(all_terms))
            ys = all_terms
            plt.bar(xs, ys)#, 1.0, align='center')
            plt.xticks(xs)
            plt.yticks(ys)
            plt.ylabel('The combination (surprise)')
            """

            #plt.tight_layout()
            plt.savefig('single_instance_debugging_fig.png')
            plt.show()
            self.debugging = False

        return self.all_priors_indx[n][indx]

    def check_for_merging(self, n, episodic_memory = None, verbose = False):
        if self.merging_disabled:
            return []


        self.maximum_sim[n] = 0.0 # For illustration
        # Threshold: similarity above which it's considered to be the same
        index1 = self.last_prior_index[n]
        to_check = list(self.M[n].keys())
        did_merge = []
        if index1 in to_check:
            to_check.remove(index1)
        for index2 in to_check:
                dist = np.linalg.norm(self.M[n][index1].mean_long-self.M[n][index2].mean_long)
                sim1 = np.exp( - (dist**2.0) / (2.0 * self.M[n][index1].var*self.av_variance[n][0]) )
                sim2 = np.exp( - (dist**2.0) / (2.0 * self.M[n][index2].var*self.av_variance[n][0]) )
                sim = (sim1+sim2)/2.0
                if sim > self.maximum_sim[n]:
                    self.maximum_sim[n] = sim
                if sim > self.merging_threshold:
                    if verbose:
                        print('Priors', index1, 'and', index2, 'of layer',
                              n, 'are very similar ('+str(sim)+'). Merging initiated...!')
                        if n == 0:
                            import matplotlib.pyplot as plt
                            import cv2
                            print("VARIANCES:",self.M[n][index1].var,self.M[n][index2].var)
                            print("Pall:",self.M[n][index1].Pall,self.M[n][index2].Pall)
                            print("all variances:",np.mean([self.M[n][aaa].var for aaa in self.M[n].keys()]))
                            print("All priors there:",len(self.M[n]))
                            plt.title("SIMILARITY: "+str(sim))
                            plt.subplot(221)
                            plt.imshow(cv2.cvtColor(np.uint8(self.M[n][index1].mean), cv2.COLOR_BGR2RGB))
                            plt.xlabel('short 1')
                            plt.subplot(222)
                            plt.imshow(cv2.cvtColor(np.uint8(self.M[n][index1].mean_long), cv2.COLOR_BGR2RGB))
                            plt.xlabel('long 1')
                            plt.subplot(223)
                            plt.imshow(cv2.cvtColor(np.uint8(self.M[n][index2].mean), cv2.COLOR_BGR2RGB))
                            plt.xlabel('short 2')
                            plt.subplot(224)
                            plt.imshow(cv2.cvtColor(np.uint8(self.M[n][index2].mean_long), cv2.COLOR_BGR2RGB))
                            plt.xlabel('long 2')
                            plt.show()

                    # Always keeping the smallest index
                    remaining_index = index1 if index1 < index2 else index2
                    forgetting_index = index1 if index1 > index2 else index2

                    did_merge.append((forgetting_index,remaining_index))

                    to_check.clear() # We don't want to continue searching!
                    # Merging activation pattern of the remaining!
                    # We use the product of two Gaussians:
                    # https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
                    var1 = self.M[n][index1].var
                    var2 = self.M[n][index2].var

                    self.M[n][remaining_index].mean = self.M[n][index1].mean*(var2/(var1+var2)) + self.M[n][index2].mean * (var1/(var1+var2))
                    self.M[n][remaining_index].mean_long = self.M[n][index1].mean_long * (var2/(var1+var2)) + self.M[n][index2].mean_long * (var1/(var1+var2))
                    self.M[n][remaining_index].var = 1.0 / ( (1.0/var1) + (1.0/var2) )

                    self.M[n][remaining_index].old_vars.append(self.M[n][remaining_index].var)

                    # Merging probabilities!!!
                    self.M[n][remaining_index].Pall = self.M[n][index1].Pall + self.M[n][index2].Pall
                    self.M[n][remaining_index].reused = True if self.M[n][index1].reused or self.M[n][index2].reused else False

                    for conditions in self.M[n][forgetting_index].P.keys():
                        if conditions in self.M[n][remaining_index].P:
                            self.M[n][remaining_index].P[conditions] += self.M[n][forgetting_index].P[conditions]
                        else:
                            self.M[n][remaining_index].P[conditions] = self.M[n][forgetting_index].P[conditions]

                    # Updating episodic memory!
                    if episodic_memory != None:
                        episodic_memory.merge(n, remaining_index, forgetting_index)

                    # Update last_prior_index[n]
                    if self.last_prior_index[n] == forgetting_index:
                        self.last_prior_index[n] = remaining_index

                    # Update last_conditions[n]
                    if n > 0 and self.last_conditions[n-1][1] == forgetting_index:
                        self.last_conditions[n-1] = (self.last_conditions[n-1][0], remaining_index)

                    # Updating probabilities of this layer!             # Reminder: Structure of conditions: ( e_{n,-1}, a_{n+1,t} )
                    for index3 in self.M[n].keys():
                        # Find all conditions from this layer where the prior with forgetting_index was used as e[-1]
                        all_relevant_conditions = [cond for cond in self.M[n][index3].P.keys() if cond[0] == forgetting_index]
                        # Move those counders to the conditional probabilities that use the remaining_index
                        for conditions in all_relevant_conditions:
                            target_conditions = (remaining_index, conditions[1])
                            # If already exists, just update it
                            if target_conditions in self.M[n][index3].P:
                                self.M[n][index3].P[target_conditions] += self.M[n][index3].P[conditions]
                            # Else, just move it there
                            else:
                                self.M[n][index3].P[target_conditions] = self.M[n][index3].P[conditions]
                            # It is a single number so we can use pop() instead of 'del'
                            self.M[n][index3].P.pop(conditions)

                    # Updating probabilities of layer below!
                    if n > 0:
                        for index_below in self.M[n-1].keys():
                            # Find all conditions from layer below where the prior with forgetting_index was used as a[n-1][t]
                            all_relevant_conditions = [cond for cond in self.M[n-1][index_below].P.keys() if cond[1] == forgetting_index]
                            # Move those counders to the conditional probabilities that use the remaining_index
                            for conditions in all_relevant_conditions:
                                target_conditions = (conditions[0], remaining_index)
                                # If already exists, just update it
                                if target_conditions in self.M[n-1][index_below].P:
                                    self.M[n-1][index_below].P[target_conditions] += self.M[n-1][index_below].P[conditions]
                                # Else, just move it there
                                else:
                                    self.M[n-1][index_below].P[target_conditions] = self.M[n-1][index_below].P[conditions]
                                # It is a single number so we can use pop() instead of 'del'
                                self.M[n-1][index_below].P.pop(conditions)

                    # Deletes both the key and the memory!!
                    del self.M[n][forgetting_index]
        return did_merge

    def predict_layer(self, n, e1n_index, forced_prior_index=-1):
        """
        Returns Mprob, self.M[n] mean/var and sets self.last_prior_index!
        """

        if forced_prior_index in self.M[n]:
            predicted_indx = forced_prior_index
        else:
            predicted_indx = self.generative_model(n, e1n_index)

        # Remember which prior and conditions they were...
        if predicted_indx < len(self.M[n])-1:
            self.M[n][predicted_indx].reused = True
        self.last_prior_index[n] = predicted_indx
        self.last_conditions[n] = (e1n_index, self.last_prior_index[n+1]) # ( e_{n,-1}, a_{n+1,t} )

        # If we found a prior:
        if self.last_prior_index[n] >= 0:
            return self.M[n][predicted_indx].mean, self.M[n][predicted_indx].var
        else:
            print("ERORR: No prior found:",actual_prob, -1, -1)
            exit()
























#
