# -*- coding: utf-8 -*-
"""

tree structure stuff for particular experiences..

TODO: UPDATE THE DESCRIPTION...
"""
from __future__ import print_function
import numpy as np
import os
import json
from sys import stdout
# For graph visualization
import networkx as nx
import matplotlib.pyplot as plt
from random import shuffle, randint

class EpisodicMemory:
    class Node:
        def __init__(self, activ, prior_index, novelty):
            # activation pattern of the current layer (not to be confused with
            # activations of all layers defined in class Epis.Memory
            self.activation = activ
            self.prior_index = prior_index
            self.novelty = novelty
            self.recency = 1.0
            self.children = list() # list of tupples of the structure (Node index, novelty weight)
            self.parent = None
            self.previous = None

        def __repr__(self):
            return str(round(self.novelty*self.recency,1))#"N"#"Node: [%s]" % str(np.shape(self.activation))

    def __init__(self, parameters_filename):
        params = json.load(open(parameters_filename))
        self.root = self.Node(np.array([]), -1, -1.0)
        # List of pointers of all nodes divided into different layers.
        # It is not necessary but it simplifies our computations!
        self.all_nodes = []

        # List of nodes that have been recalled during the last time something was
        # recalled. We want to know that in order to be able to visualize them!
        self.step2_tree = []
        self.step3_tree = []
        self.step4_frames = []

        # An image which will contain
        self.visualization = np.zeros((len(params["network"]["layers_to_use"]),1000,3))
        # The size of the layer with the most episodic instances. Used for visualization purposes..
        self.memory_size = 0
        # Parameter to set the number of recalls over time (frequency of recalls)
        self.recall_frequency = params["episodic_memory"]["recall_frequency"]

        self.sigma = params["episodic_memory"]["sigma"]

        # Statistics kept for the recalling process
        self.av_children = [] # Average number of children per layer so far

        for i in range(len(params["network"]["layers_to_use"])):
            self.all_nodes.append([])
            self.av_children.append(0.0)

        # Visualization of the lattice as a graph!
        self.G = nx.Graph()
        self.Gpos = {}
        self.Gcolormap = []
        self.Glabels = {}


    # Always adds a node as the last child of the last node in a layer
    def add_node(self, layer, activation, prior_index, novelty):
        new_node = self.Node(activation, prior_index, novelty)

        if len(self.all_nodes) <= layer:
            print("Episodic memory: Layer size mismatch ("+str(layer)+")")
            exit()

        # Calculate average number of children for this layer including the last
        # recorded node (before we change to a new..)
        if len(self.all_nodes[layer]) > 0:
            self.av_children[layer] = len(self.all_nodes[layer]) * self.av_children[layer]
            self.av_children[layer] += float(len(self.all_nodes[layer][-1].children))
            self.av_children[layer] /= float(len(self.all_nodes[layer])+1.0)

        self.all_nodes[layer].append(new_node)
        my_index = len(self.all_nodes[layer])-1
        if len(self.all_nodes[layer]) > self.memory_size:
            self.memory_size = len(self.all_nodes[layer])

        # Visualizations
        self.G.add_node((layer,my_index))
        self.Gpos[(layer,my_index)] = (self.memory_size, layer)
        self.Glabels[(layer,my_index)] = str(prior_index) # my_index
        self.Gcolormap.append((novelty,novelty,novelty))

        if len(self.all_nodes) > layer+1 and len(self.all_nodes[layer+1]) > 0:
            self.all_nodes[layer+1][-1].children.append(my_index)
            self.G.add_edge((layer+1,len(self.all_nodes[layer+1])-1),(layer, my_index))


    def initialise_tree(self, activations):
        '''
        Initialization
        --------------
        If tree is empty, then add a new element/node to all layers
        This loop will run only the first time we call update
        '''
        for L in range(len(activations)):
            layer = len(activations)-L-1
            self.add_node(layer, activations[layer], 0, 0.0001)

    def draw_tree(self, with_labels = True):
        nx.draw(self.G, labels = self.Glabels, node_color = self.Gcolormap,
                pos = self.Gpos, with_labels = with_labels)
        plt.show()

    def update(self, new_novelty = -1.0, activation_pattern = np.array([]),
               semantic_memory = None, pred_coding = None, layer = -1):
        '''
        activation_pattern: An np.array() of current activations in the current layer
        new_novelty: Indicates if the current layer exceeded the att.
                              threshold e.g. 0 or 1
        '''

        self.decay(layer)

        # Main loop
        # --------------
        # If a new salient feature is found and memory needs to be updated:
        if new_novelty > 0.0 and layer != (len(self.all_nodes)-1):

            # Add the new nodes
            self.add_node(layer, activation_pattern, semantic_memory.last_prior_index[layer], new_novelty)

            # Print the whole tree
            self.print_me()

            # Update the visualization of the whole tree
            self.plot_me()

            # Update the 'links' between areas of the brain :P
            semantic_memory.update_context_probabilities_and_create_new_priors(layer, activation_pattern)


    def merge(self, layer, index1, index2):
        if 0 > layer >= len(self.all_nodes):
            print("WRONG LAYER INDEX IN EPISODIC MEMORY!!", layer, index1, index2)
            exit()
        for node in self.all_nodes[layer]:
            if node.prior_index == index2:
                node.prior_index = index1

    def get_last_in_layer(self, layer):
        if layer < len(self.all_nodes) and len(self.all_nodes[layer]) > 0:
            return self.all_nodes[layer][-1]
        else:
            return False

    def print_me(self):
        #os.system('cls' if os.name == 'nt' else 'clear')
        print("-- Adding node to the tree --")

        for L in range(len(self.all_nodes)):
            layer = len(self.all_nodes)-L-1
            print(layer, len(self.all_nodes[layer]), np.shape(self.all_nodes[layer][0].activation))

        print ("")
        print (" | 1.0 (root)")
        for L in range(len(self.all_nodes)):
            layer = len(self.all_nodes)-L-1
            text = str(self.av_children[layer]) + " | "
            for N in self.all_nodes[layer][:100]:
                text += N.__repr__() + " "
            print(text)

    def plot_me(self):
        for L in range(len(self.all_nodes)):
            layer = len(self.all_nodes)-L-1
            for i in range(len(self.all_nodes[layer])):
                if i < np.shape(self.visualization)[1]:
                    self.visualization[L,i,0] = self.all_nodes[layer][i].novelty
                    self.visualization[L,i,1] = self.all_nodes[layer][i].novelty
                    self.visualization[L,i,2] = self.all_nodes[layer][i].novelty

    def decay(self, layer):
            for i in range(len(self.all_nodes[layer])-1):
                self.all_nodes[layer][i].recency *= 0.999

    # Returns the part of the visualization that has been used so far..
    def get_visualization(self):
        return self.visualization[:,:self.memory_size,:]

    def RBF(self, x, y, variance):
        if variance == 0.0:
            return 0.0
        else:
            dist = np.linalg.norm(x-y)
            return np.exp( -(dist**2.0) / (2.0*variance) )

    def Precall(self, layer, ep_indx, prior_mean, prior_var):
        P1 = self.all_nodes[layer][ep_indx].recency
        P2 = self.RBF(self.all_nodes[layer][ep_indx].activation, prior_mean, prior_var)
        P3 = self.all_nodes[layer][ep_indx].novelty
        return P1*P2*P3

    # This is the process with which our system decides whether to start recalling
    # an episode!
    # It is based on 3 recall criteria:
    #   1. Recency: How recent this node was recorded
    #   2. Similarity: How similar is this node to what is happening right now
    #   3. Novelty/surprise: How important this memory was!
    def check_for_recall_root(self, layer, current_prior_index, prior_mean, prior_var):
        # NOTE: This has the same effect with multiplying the recall_frequency
        #       with the final probability! Thing about it ;)
        #       We have to use it because otherwise we'd recall stuff all the time
        rr = np.random.rand()
        if rr >= self.recall_frequency:
            return -1

        if layer >= len(self.all_nodes):
            print("Error: Wrong index of layers in episodic memory!")
            exit()

        # Make a list of all nodes that could be recalled along with their score
        # i.e. ther probabilities to be recalled:
        possible_nodes = []
        possible_nodes_probs = []
        for i in range(len(self.all_nodes[layer])):
            if self.all_nodes[layer][i].prior_index == current_prior_index:
                possible_nodes.append(i)
                possible_nodes_probs.append(self.Precall(layer,i,prior_mean,prior_var))

        # If no node can be recalled, return -1
        if len(possible_nodes) <= 0:
            return -1

        # Select one of the filtered nodes randomly based on the distribution Precall[node]
        rand_num = np.random.rand()*float(sum(possible_nodes_probs))
        #print("rand_num",float(sum(possible_nodes_probs)), rand_num, "curr_prior_indx", current_prior_index, [node.prior_index for node in self.all_nodes[layer]])
        selected_index = 0
        A = [sum(possible_nodes_probs[:i]) for i in range(1,len(possible_nodes_probs)+1)]
        #print("A",A)
        while selected_index < len(A) and rand_num > A[selected_index]:
            selected_index += 1
            #print("sel_index",selected_index)
        if selected_index >= len(A):
            print("Something went wrong:", A, selected_index)
            exit()

        # Make final decision of whether we'll recall something now
        if rr < self.recall_frequency*possible_nodes_probs[selected_index]:
            print(" RECALL: The selected index is:", possible_nodes[selected_index], "with prob:",
                  possible_nodes_probs[selected_index])
            return possible_nodes[selected_index]
        else:
            return -1 # If no node was recalled, return -1

    # Generate a sub-tree probabilistically and return it
    def recall(self, root_layer, root_index, semantic_memory, retrospective=False, effort=1, show_plots=False, verbose=True,recency=1.0):
        # We have 4 steps to take here
        # 1. Find closest branch - list of indices
        # 2. Find recalled tree - list (layer) of lists (node) of 3-tuples (episodic index, list of children with episodic index)
        # 3. Extend tree -
        # 4. Transofrm tree into frames - list of lists of episodic indices
        current_nodes = {}
        if verbose:
            print("Recalling a tree with root:", root_layer, root_index)

        # 1. FIND FIRST FRAME
        first_frame = []
        for L in range(len(self.all_nodes)):
            current_nodes[L] = {}
            if L < root_layer:
                first_frame.append(-2) # This will be replaced depending on the first frame
            elif L == root_layer:
                first_frame.append(root_index)
            else:
                first_frame.append(-1) # This will remain empty

        if retrospective:
            if verbose:
                print("Retrospective recall:")
            current_layer = root_layer
            current_index = root_index
            while current_layer > 0 and self.all_nodes[current_layer][current_index].children:
                #print("Cur layer:", current_layer, "Cur index:", current_index, "number of children:", len(self.all_nodes[current_layer][current_index].children))
                current_index = self.all_nodes[current_layer][current_index].children[0]
                current_layer -= 1
                first_frame[current_layer] = current_index
        else:
            current_layer = root_layer
            current_index = root_index
            while current_layer > 0 and self.all_nodes[current_layer][current_index].children:
                current_children = self.all_nodes[current_layer][current_index].children
                #print(current_layer, "CHILDREN",current_children, "current_index",current_index)
                max_P = 0.0
                current_index = 0
                current_layer -= 1
                # We need the most recent state of the system (i.e. M' from
                # semantic memory) from the root_layer and bellow
                p_index_1 = semantic_memory.last_prior_index[current_layer]
                last_mean = semantic_memory.M[current_layer][p_index_1].mean
                last_var = semantic_memory.M[current_layer][p_index_1].var
                last_var *= semantic_memory.av_variance[current_layer][0]

                for child in current_children:
                    P = self.Precall(current_layer, child, last_mean, last_var)
                    if P > max_P:
                        max_P = P
                        current_index = child
                first_frame[current_layer] = current_index

        if verbose: print("RECALL STEP 1: the first_frame is", first_frame)

        # 2. FIND RECALLED TREE (and list)
        self.step2_tree = []
        self.step3_tree = []
        # Empty recalled_list - buffer that keeps track of recalled indices in each layer
        for L in range(len(self.all_nodes)):
            self.step2_tree.append([])
            self.step3_tree.append([])

        current_nodes[root_layer][root_index] = list(self.all_nodes[root_layer][root_index].children)
        self.step2_tree[root_layer] = [(root_index, list(self.all_nodes[root_layer][root_index].children) )]
        self.step3_tree[root_layer] = [(root_index, list(self.all_nodes[root_layer][root_index].children), None)]
        current_layer = root_layer
        current_index = root_index
        while current_layer > 0:
            parent_tuples = list(self.step2_tree[current_layer])
            for parent_index,parent_tuple in enumerate(parent_tuples):
                parent = parent_tuple[0]
                children = parent_tuple[1]
                for child in children:
                    if child >= first_frame[current_layer-1]:
                        P1 = recency #self.all_nodes[current_layer-1][child].recency
                        P3 = self.all_nodes[current_layer-1][child].novelty
                        if sum([1 for repeats in range(effort) if np.random.rand() < P1*P3]) > 0:
                            self.step2_tree[current_layer-1].append( (child, list(self.all_nodes[current_layer-1][child].children) ) )
                            self.step3_tree[current_layer-1].append( (child, list(self.all_nodes[current_layer-1][child].children), parent_index) )
                            current_nodes[current_layer-1][child] = list(self.all_nodes[current_layer-1][child].children)
                        else:
                            self.step3_tree[current_layer][parent_index][1].remove(child)
                            current_nodes[current_layer][parent].remove(child)
                            #print("En eperasen")
                    else:
                        #print("Etsi ena debugging na yparxei",child,first_frame[current_layer-1])
                        self.step3_tree[current_layer][parent_index][1].remove(child)
                        current_nodes[current_layer][parent].remove(child)
            current_layer -= 1

        if verbose:
            print("STEP2 tree:")
            for i in range(len(self.all_nodes)-1,-1,-1):
                print("\t",self.step2_tree[i])

            print("STEP3 tree:")
            for i in range(len(self.all_nodes)-1,-1,-1):
                print("\t",self.step3_tree[i])

        # Small trick to give unique ids to negative nodes which represent nodes
        # not taken from semantic memory (the green ones with questionmark)
        negative = -1

        current_layer = root_layer
        current_index = root_index
        while current_layer > 0:
            parent_tuples = list(self.step3_tree[current_layer])
            self.step3_tree[current_layer-1] = []
            for parent_index,parent_tuple in enumerate(parent_tuples):
                parent = parent_tuple[0]
                children = parent_tuple[1]
                added_children = len(children)

                if parent >= 0:
                    sm_index = self.all_nodes[current_layer][parent].prior_index
                    current_av_children = np.average([len(node.children) for node in self.all_nodes[current_layer] if node.prior_index == sm_index ])
                    #print("current_av_children A",current_av_children,self.av_children[current_layer])
                else:
                    current_av_children = self.av_children[current_layer]

                if not retrospective:
                    current_av_children /= 2.0

                estimated_children = int(round(np.random.normal(current_av_children,self.sigma),0))

                # NOTE: If we don't do this we'll get a tree that does not
                # necessarily reach the bottom layer all the time!
                if estimated_children < 1:
                    estimated_children = 1

                # Here we'll shuffle the current order of the order of childer
                # and insert them into the lower layer list (we have emptied it
                # when we entered this layer).
                temporary_list_of_children = []
                if children:
                    first_child = children[0]
                    # Initialise the list that will be added in the layer below
                    final_list_of_children = [(first_child, current_nodes[current_layer-1][first_child], parent_index)]
                    for i,child in enumerate(children):
                        if i != 0: # Do not put first child.
                            temporary_list_of_children.append( (child, current_nodes[current_layer-1][child], parent_index) )
                else:
                    final_list_of_children = []

                if estimated_children > added_children:
                    for i in range(estimated_children-added_children):
                        temporary_list_of_children.insert(randint(0,len(temporary_list_of_children)),(negative, [], parent_index))
                        negative -= 1

                final_list_of_children += temporary_list_of_children

                # Here we instert the list where it should be
                for child in final_list_of_children:
                    self.step3_tree[current_layer-1].append(child)

            current_layer -= 1

        if verbose:
            print("STEP3 tree:")
            for i in range(len(self.all_nodes)-1,-1,-1):
                print("\t",self.step3_tree[i])

        # 4. MAKE LIST OF FRAMES FOR PRED_CODING...
        self.step4_frames = []
        for L in range(len(self.all_nodes)):
            self.step4_frames.append([])

        for bottom_indx, bottom in enumerate(self.step3_tree[0]):
            # bottom[0]: episodic_index 1: children - should be empty 2: parent
            current = bottom[0]
            current_indx = bottom_indx
            L = 0
            while L <= root_layer:
                self.step4_frames[L].append(current)
                if L < root_layer:
                    parent_index = self.step3_tree[L][current_indx][2]
                    current = self.step3_tree[L+1][parent_index][0]
                    current_indx = parent_index
                L += 1

        if verbose:
            print("STEP4 frames:")
            for i in range(len(self.all_nodes)-1,-1,-1):
                print("\t",self.step4_frames[i])

        if show_plots:
            self.draw_recall()
            plt.figure(figsize=(17,5))
            plt.subplot(131)
            self.draw_step2(save=False, show=False, close=False)
            plt.subplot(132)
            self.draw_step3(save=False, show=False, close=False)
            plt.subplot(133)
            self.draw_step4(save=False, show=True)
            #exit()

    def draw_recall(self, save=True, show=False, close=True):
        Gcolormap = list(self.Gcolormap)
        Glabels = dict(self.Glabels)
        nodeListA = []
        for layer in range(len(self.step2_tree)):
            for node_index_tuple in self.step2_tree[layer]:
                node_index = node_index_tuple[0]
                if node_index >= 0:
                    #Glabels[(layer,node_index)] = "R"
                    nodeListA.append((layer,node_index))
        nx.draw(self.G, node_color = Gcolormap, labels = self.Glabels,
                pos = self.Gpos, with_labels = True)
        nx.draw_networkx_nodes(self.G, self.Gpos, labels = self.Glabels,
                               nodelist=nodeListA, node_color="r")

        if save:
            s_index = 0
            while os.path.exists("recalls/"+str(s_index)+'.png'):
                s_index += 1
            plt.savefig("recalls/"+str(s_index)+'.png')
        if show:
            plt.show()
        if close:
            plt.close()

    def draw_step2(self, save=True, show=False, close=True):
        G = nx.Graph()
        Gpos = {}
        Gcolormap = []
        Glabels = {}
        fig_size = 100.0
        for layer in range(len(self.step2_tree)):
            for i,node_index_tuple in enumerate(self.step2_tree[layer]):
                node_index = node_index_tuple[0]
                Gpos[(layer,node_index)] = ((i+1)*fig_size/float(len(self.step2_tree[layer])+1), layer)
                G.add_node((layer,node_index))
                if node_index >= 0:
                    Glabels[(layer,node_index)] = self.all_nodes[layer][node_index].prior_index
                    Gcolormap.append((0.99,0,0))
                else:
                    Glabels[(layer,node_index)] = "?"
                    Gcolormap.append((0.2,0.99,0.2))

        for edge in self.G.edges():
            if edge[0] in G.nodes() and edge[1] in G.nodes():
                G.add_edge(edge[0], edge[1])

        nx.draw_networkx(G, pos = Gpos, labels = Glabels, node_color = Gcolormap, arrows=True)

        if save:
            s_index = 0
            while os.path.exists("recalls/"+str(s_index)+'.png'):
                s_index += 1
            plt.savefig("recalls/"+str(s_index)+'.png')
        if show:
            plt.show()
        if close:
            plt.close()

    def draw_step3(self, save=True, show=False, close=True):
        G = nx.Graph()
        Gpos = {}
        Gcolormap = []
        Glabels = {}
        fig_size = 100.0
        for layer in range(len(self.step3_tree)):
            for i,node_index_tuple in enumerate(self.step3_tree[layer]):
                node_index = node_index_tuple[0]
                Gpos[(layer, i, node_index)] = ((i+1)*fig_size/float(len(self.step3_tree[layer])+1), layer)
                G.add_node((layer, i, node_index))
                if (layer, i, node_index) in Glabels:
                    print( (layer, i, node_index), Glabels)
                    if node_index >= 0:
                        print("prior index:", self.all_nodes[layer][node_index].prior_index)
                    print(self.step3_tree)
                    print("Yo2")
                    exit()

                if node_index >= 0:
                    Glabels[(layer, i, node_index)] = self.all_nodes[layer][node_index].prior_index
                    Gcolormap.append((0.99,0,0))
                else:
                    Glabels[(layer, i, node_index)] = node_index
                    #Glabels[(layer, i, node_index)] = "?"
                    Gcolormap.append((0.2,0.99,0.2))

        for edge in self.G.edges():
            if edge[0] in G.nodes() and edge[1] in G.nodes():
                G.add_edge(edge[0], edge[1])

        nx.draw_networkx(G, pos = Gpos, labels = Glabels, node_color = Gcolormap, arrows=True)

        if save:
            s_index = 0
            while os.path.exists("recalls/expaneded_"+str(s_index)+'.png'):
                s_index += 1
            plt.savefig("recalls/expaneded_"+str(s_index)+'.png')
        if show:
            plt.show()
        if close:
            plt.close()


    def draw_step4(self, save=True, show=False, close=True):
        G = nx.Graph()
        Gpos = {}
        Gcolormap = []
        Glabels = {}
        fig_size = 100.0
        extra_id = 0
        for layer in range(len(self.step4_frames)):
            for i,node_index in enumerate(self.step4_frames[layer]):
                Gpos[(layer,node_index,extra_id)] = ((i+1)*fig_size/float(len(self.step4_frames[0])+1), layer)
                G.add_node((layer,node_index,extra_id))
                if node_index >= 0:
                    Glabels[(layer,node_index,extra_id)] = self.all_nodes[layer][node_index].prior_index
                    Gcolormap.append((0.99,0,0))
                else:
                    Glabels[(layer,node_index,extra_id)] = node_index
                    #Glabels[(layer,node_index)] = "?"
                    Gcolormap.append((0.2,0.99,0.2))
                extra_id += 1
        nx.draw_networkx(G, pos = Gpos, labels = Glabels, node_color = Gcolormap)

        if save:
            s_index = 0
            while os.path.exists("recalls/"+str(s_index)+'_step4.png'):
                s_index += 1
            plt.savefig("recalls/"+str(s_index)+'_step4.png')
        if show:
            plt.show()
        if close:
            plt.close()







#
