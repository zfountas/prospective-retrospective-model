"""
Script to run all trials and produce the figure based on just the accummulators
(no regression).
It also saves a pickle with the resulting dataset called 'model_accumulators_dataset.pkl'
"""

# Arguments
import pickle, csv
import matplotlib.pyplot as plt
from scipy import stats
from model.attention_system import *
from model.episodic_memory import * # tree structure stuff of the particular experience
from model.working_memory import *  # accummulators etc
from model.semantic_memory import * # bag of words with trees etc
import trials_used as tu

# Generate a sub-tree probabilistically and return it
def recall(ep_mem_sigma, ep_mem_av_children, all_nodes, effort=1, verbose=False, recency=1.0):
    # We have 4 steps to take here
    # 1. Find first frame
    # 2. Find recalled tree - list (layer) of lists (node) of 3-tuples (episodic index, list of children with episodic index)
    # 3. Extend tree -
    # 4. Transofrm tree into frames - list of lists of episodic indices
    current_nodes = {}

    root_layer = len(all_nodes) - 1

    # 1. FIND FIRST FRAME
    first_frame = []
    for L in range(len(all_nodes)):
        current_nodes[L] = {}
        if L < root_layer:
            first_frame.append(-2) # This will be replaced depending on the first frame
        elif L == root_layer:
            first_frame.append(0)
        else:
            first_frame.append(-1) # This will remain empty

    current_layer = root_layer
    current_index = 0
    while current_layer > 0 and all_nodes[current_layer][current_index].children:
        #print("Cur layer:", current_layer, "Cur index:", current_index, "number of children:", len(all_nodes[current_layer][current_index].children))
        current_index = all_nodes[current_layer][current_index].children[0]
        current_layer -= 1
        first_frame[current_layer] = current_index

    if verbose: print("RECALL STEP 1: the first_frame is", first_frame)

    # 2. FIND RECALLED TREE (and list)
    step2_tree = []
    step3_tree = []
    # Empty recalled_list - buffer that keeps track of recalled indices in each layer
    for L in range(len(all_nodes)):
        step2_tree.append([])
        step3_tree.append([])

    current_nodes[root_layer][0] = list(all_nodes[root_layer][0].children)
    step2_tree[root_layer] = [(0, list(all_nodes[root_layer][0].children) )]
    step3_tree[root_layer] = [(0, list(all_nodes[root_layer][0].children), None)]
    current_layer = root_layer
    current_index = 0
    while current_layer > 0:
        parent_tuples = list(step2_tree[current_layer])
        for parent_index,parent_tuple in enumerate(parent_tuples):
            parent = parent_tuple[0]
            children = parent_tuple[1]
            for child in children:
                if child >= first_frame[current_layer-1]:
                    P1 = recency #all_nodes[current_layer-1][child].recency
                    P3 = all_nodes[current_layer-1][child].novelty
                    if sum([1 for repeats in range(effort) if np.random.rand() < P1*P3]) > 0 or current_layer == root_layer: # If it's part of the second layer from the top take it!!
                        step2_tree[current_layer-1].append( (child, list(all_nodes[current_layer-1][child].children) ) )
                        step3_tree[current_layer-1].append( (child, list(all_nodes[current_layer-1][child].children), parent_index) )
                        current_nodes[current_layer-1][child] = list(all_nodes[current_layer-1][child].children)
                    else:
                        step3_tree[current_layer][parent_index][1].remove(child)
                        current_nodes[current_layer][parent].remove(child)
                        #print("En eperasen")
                else:
                    #print("Etsi ena debugging na yparxei",child,first_frame[current_layer-1])
                    step3_tree[current_layer][parent_index][1].remove(child)
                    current_nodes[current_layer][parent].remove(child)
        current_layer -= 1

    if verbose:
        print("STEP2 tree:")
        for i in range(len(all_nodes)-1,-1,-1):
            print("\t",step2_tree[i])

        print("STEP3 tree:")
        for i in range(len(all_nodes)-1,-1,-1):
            print("\t",step3_tree[i])

    # Small trick to give unique ids to negative nodes which represent nodes
    # not taken from semantic memory (the green ones with questionmark)
    negative = -1

    current_layer = root_layer
    current_index = 0
    while current_layer > 0:
        parent_tuples = list(step3_tree[current_layer])
        step3_tree[current_layer-1] = []
        for parent_index,parent_tuple in enumerate(parent_tuples):
            parent = parent_tuple[0]
            children = parent_tuple[1]
            added_children = len(children)

            if parent >= 0:
                sm_index = all_nodes[current_layer][parent].prior_index
                current_av_children = np.average([len(node.children) for node in all_nodes[current_layer] if node.prior_index == sm_index ])
                #print("current_av_children A",current_av_children,self.av_children[current_layer])
            else:
                current_av_children = ep_mem_av_children[current_layer]

            estimated_children = int(round(np.random.normal(current_av_children,ep_mem_sigma),0))

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
                step3_tree[current_layer-1].append(child)

        current_layer -= 1

    if verbose:
        print("STEP3 tree:")
        for i in range(len(all_nodes)-1,-1,-1):
            print("\t",step3_tree[i])

    # 4. MAKE LIST OF FRAMES FOR PRED_CODING...
    step4_frames = []
    for L in range(len(all_nodes)):
        step4_frames.append([])

    for bottom_indx, bottom in enumerate(step3_tree[0]):
        # bottom[0]: episodic_index 1: children - should be empty 2: parent
        current = bottom[0]
        current_indx = bottom_indx
        L = 0
        while L <= root_layer:
            step4_frames[L].append(current)
            if L < root_layer:
                parent_index = step3_tree[L][current_indx][2]
                current = step3_tree[L+1][parent_index][0]
                current_indx = parent_index
            L += 1

    if verbose:
        print("STEP4 frames:")
        for i in range(len(all_nodes)-1,-1,-1):
            print("\t",step4_frames[i])

    return step3_tree


semantic_memory = pickle.load(open('semantic_memory.pkl','rb'))
all_av_children = pickle.load(open('all_av_children.pkl','rb'))
print('All average children:', all_av_children)

durations = {'prosp_low':[], 'prosp_high':[], 'retro_low':[], 'retro_high':[]}
current_trials = {'prosp_low':[], 'prosp_high':[], 'retro_low':[], 'retro_high':[]}
prosp_acc = {'low':[], 'high':[]}
retro_acc = {'low':[], 'high':[]}
prosp_acc_per_s = {'low':[], 'high':[]}
retro_acc_per_s = {'low':[], 'high':[]}

# SOS: For some reason, sigmas<0.5 are acting very differntly
# sigma: higher sigma shifts the retrospective curve upwards
# effort: higher effort shifts the retrospective curve upwards
# higher tau (less decay, i.e. less attention) shifts both curves downwards
params = {'low_pr_t': 46, 'high_pr_t': 52, 'low_ret_t': 90, 'high_ret_t': 90.0, 'low_eff': 1, 'high_eff': 70, 'recency': 0.1, 'sigma': 0.1}

TRIAL_FIRST = 1
TRIAL_LAST = 4290 # 4290 are all of them sparsegps in gpflow....

all_indices = list(range(TRIAL_FIRST, TRIAL_LAST))

for trials_left,trial in enumerate(all_indices):
    try:
        all_surprises = pickle.load(open('trials_surpr/surprises_'+str(trial)+'.pkl','rb'))
        # Acts as dummy activations..!
        dummy_activations = all_surprises[0]
        duration = float(len(all_surprises))/30.0

        for mode in ['low', 'high']:
            if trial in tu.trial_ids['prosp_'+mode]:
                durations['prosp_'+mode].append(duration)
                current_trials['prosp_'+mode].append(trial)

                attention_prospective = AttentionSystem('parameters.json')
                attention_prospective.tau = params[mode+'_pr_t']
                working_memory_prospective = WorkingMemory('parameters.json')
                episodic_memory_prospective = EpisodicMemory('parameters.json')
                episodic_memory_prospective.initialise_tree(dummy_activations)

                for t in range(len(all_surprises)):
                    for layer in range(len(dummy_activations)):
                        # Use the attention mechanism to calculate thresholds
                        # NOTE: It updates: 'attention.thresholds' and 'attention.surprises'
                        attention_prospective.update(all_surprises[t][layer], layer)

                        # Update episodic memory: If interesting, save the current activation pattern!
                        episodic_memory_prospective.decay(layer)
                        if attention_prospective.surprises[layer] > 0.0 and layer != (len(episodic_memory_prospective.all_nodes)-1):
                            # Add the new nodes
                            episodic_memory_prospective.add_node(layer, dummy_activations, semantic_memory.last_prior_index[layer], attention_prospective.surprises[layer])

                working_memory_prospective.update_from_episodic_memory(episodic_memory_prospective.all_nodes)

                for i in range(len(working_memory_prospective.accummulators)):
                    if len(prosp_acc_per_s[mode]) <= i:
                        prosp_acc[mode].append([])
                        prosp_acc_per_s[mode].append([])

                    prosp_acc[mode][i].append(working_memory_prospective.accummulators[i])
                    prosp_acc_per_s[mode][i].append(working_memory_prospective.accummulators[i]/duration)


            if trial in tu.trial_ids['retro_'+mode]:
                durations['retro_'+mode].append(duration)
                current_trials['retro_'+mode].append(trial)

                attention_retrospective = AttentionSystem('parameters.json')
                attention_retrospective.tau = params[mode+'_ret_t']
                working_memory_retrospective = WorkingMemory('parameters.json')
                episodic_memory_retrospective = EpisodicMemory('parameters.json')
                episodic_memory_retrospective.initialise_tree(dummy_activations)

                for t in range(len(all_surprises)):
                    for layer in range(len(dummy_activations)):
                        # Use the attention mechanism to calculate thresholds
                        # NOTE: It updates: 'attention.thresholds' and 'attention.surprises'
                        attention_retrospective.update(all_surprises[t][layer], layer)

                        # Update episodic memory: If interesting, save the current activation pattern!
                        episodic_memory_retrospective.decay(layer)
                        if attention_retrospective.surprises[layer] > 0.0 and layer != (len(episodic_memory_retrospective.all_nodes)-1):
                            # Add the new nodes
                            episodic_memory_retrospective.add_node(layer, dummy_activations, semantic_memory.last_prior_index[layer], attention_retrospective.surprises[layer])

                step3_tree = recall(ep_mem_sigma=params['sigma'], ep_mem_av_children=all_av_children, # episodic_memory.av_children
                                    all_nodes=episodic_memory_retrospective.all_nodes, effort = int(params[mode+'_eff']),
                                    verbose = False, recency=params['recency'])
                working_memory_retrospective.update_from_episodic_memory(step3_tree)

                for i in range(len(working_memory_retrospective.accummulators)):
                    if len(retro_acc_per_s[mode]) <= i:
                        retro_acc[mode].append([])
                        retro_acc_per_s[mode].append([])

                    retro_acc[mode][i].append(working_memory_retrospective.accummulators[i])
                    retro_acc_per_s[mode][i].append(working_memory_retrospective.accummulators[i]/duration)

            left = str(round(100.0*float(trials_left)/float(len(all_indices)),1))+'%       '
            print('\r',duration,'errors so far: prosp',np.mean(prosp_acc[mode]),'retro',np.mean(retro_acc[mode]),left,end='')

    except:
        pass

print("\n -- RESULT --")

print('Saving in a pickle file..!')
data_for_pkl = {'prosp_low' : prosp_acc['low'], 'prosp_high' : prosp_acc['high'],
                'retro_low' : retro_acc['low'], 'retro_high' : retro_acc['high'],
                'durations' : durations, 'trial_no' : current_trials}
with open("model_accumulators_dataset.pkl", "wb") as file:
    pickle.dump(data_for_pkl, file)


plt.figure(figsize=(18,12))
layers = len(retro_acc_per_s['low']) - 1

plt.subplot(4,layers,1)
plt.ylabel("Mean dur.judg.ratio")

ax1 = plt.subplot(4,layers,layers+1)

#ax1 = plt.axes(frameon=False)
ax1.set_frame_on(False)
#ax1.get_xaxis().tick_bottom()
#ax1.axes.get_yaxis().set_visible(False)


for i in range(layers):
    ax1 = plt.subplot(4,layers,1+i)

    #ax1 = plt.axes(frameon=False)
    ax1.set_frame_on(False)
    #ax1.get_xaxis().tick_bottom()
    #ax1.axes.get_yaxis().set_visible(False)


    plt.plot([1.10, 1.10],[np.mean(retro_acc_per_s['low'][i])-stats.sem(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['low'][i])+stats.sem(retro_acc_per_s['low'][i])],c='k',lw=1.0)
    plt.plot([2.10, 2.10],[np.mean(retro_acc_per_s['high'][i])-stats.sem(retro_acc_per_s['high'][i]), np.mean(retro_acc_per_s['high'][i])+stats.sem(retro_acc_per_s['high'][i])],c='k',lw=1.0)
    plt.plot([1.10,2.10], [np.mean(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['high'][i])],'k--',label='Retrospective')
    plt.scatter([1.10,2.10], [np.mean(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['high'][i])],c='k')

    plt.plot([1, 1],[np.mean(prosp_acc_per_s['low'][i])-stats.sem(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['low'][i])+stats.sem(prosp_acc_per_s['low'][i])],c='k',lw=1.0)
    plt.plot([2, 2],[np.mean(prosp_acc_per_s['high'][i])-stats.sem(prosp_acc_per_s['high'][i]), np.mean(prosp_acc_per_s['high'][i])+stats.sem(prosp_acc_per_s['high'][i])],c='k',lw=1.0)
    plt.plot([1,2], [np.mean(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['high'][i])],'k',label='Prospective')
    plt.scatter([1,2], [np.mean(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['high'][i])],c='k')

    #plt.legend()
    #plt.xlabel('Layer: '+str(i))
    #plt.ylabel("Mean duration judgment ratio"+str(params))
    #plt.xlabel("Cog. load ("+str(len(retro_acc_per_s['low'][i]))+" vs "+str(len(retro_acc_per_s['high'][i]))+" & "+str(len(prosp_acc_per_s['low'][i]))+" vs "+str(len(prosp_acc_per_s['high'][i]))+") non-c.")
    #plt.ylim(0.9,1.4)

    plt.xlim(0.5,2.5)
    #plt.xticks([1,2],['Low', 'High'])
    plt.xticks([],[])
    #plt.yticks([1.5,2.0],[1.5, 2])
    #plt.axis('off')


    ax2 = plt.subplot(4,layers,layers+1+i)
    ax2.set_frame_on(False)

    plt.plot([1.10, 1.10],[np.mean(retro_acc_per_s['low'][i])-stats.sem(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['low'][i])+stats.sem(retro_acc_per_s['low'][i])],c='k',lw=1.0)
    plt.plot([2.10, 2.10],[np.mean(retro_acc_per_s['high'][i])-stats.sem(retro_acc_per_s['high'][i]), np.mean(retro_acc_per_s['high'][i])+stats.sem(retro_acc_per_s['high'][i])],c='k',lw=1.0)
    plt.plot([1.10,2.10], [np.mean(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['high'][i])],'k--',label='Retrospective')
    plt.scatter([1.10,2.10], [np.mean(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['high'][i])],c='k')
    plt.plot([1, 1],[np.mean(prosp_acc_per_s['low'][i])-stats.sem(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['low'][i])+stats.sem(prosp_acc_per_s['low'][i])],c='k',lw=1.0)
    plt.plot([2, 2],[np.mean(prosp_acc_per_s['high'][i])-stats.sem(prosp_acc_per_s['high'][i]), np.mean(prosp_acc_per_s['high'][i])+stats.sem(prosp_acc_per_s['high'][i])],c='k',lw=1.0)
    plt.plot([1,2], [np.mean(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['high'][i])],'k',label='Prospective')
    plt.scatter([1,2], [np.mean(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['high'][i])],c='k')
    plt.xlabel('Layer: '+str(i))

    plt.xlim(0.5,2.5)
    #plt.ylim(0.5,3.5)
    plt.xticks([],[])
    if i == 0:
        pass
    else:
        pass #plt.yticks([],[])


plt.subplot(212)
for i in range(layers):
    plt.plot([1.10+2*i, 1.10+2*i],[np.mean(retro_acc_per_s['low'][i])-stats.sem(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['low'][i])+stats.sem(retro_acc_per_s['low'][i])],c='k',lw=1.0)
    plt.plot([2.10+2*i, 2.10+2*i],[np.mean(retro_acc_per_s['high'][i])-stats.sem(retro_acc_per_s['high'][i]), np.mean(retro_acc_per_s['high'][i])+stats.sem(retro_acc_per_s['high'][i])],c='k',lw=1.0)
    plt.plot([1.10+2*i,2.10+2*i], [np.mean(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['high'][i])],'k--',label='Retrospective')
    plt.scatter([1.10+2*i,2.10+2*i], [np.mean(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['high'][i])],c='k')
    plt.plot([1+2*i, 1+2*i],[np.mean(prosp_acc_per_s['low'][i])-stats.sem(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['low'][i])+stats.sem(prosp_acc_per_s['low'][i])],c='k',lw=1.0)
    plt.plot([2+2*i, 2+2*i],[np.mean(prosp_acc_per_s['high'][i])-stats.sem(prosp_acc_per_s['high'][i]), np.mean(prosp_acc_per_s['high'][i])+stats.sem(prosp_acc_per_s['high'][i])],c='k',lw=1.0)
    plt.plot([1+2*i,2+2*i], [np.mean(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['high'][i])],'k',label='Prospective')
    plt.scatter([1+2*i,2+2*i], [np.mean(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['high'][i])],c='k')
#plt.xlabel('Layer: '+str(i))

plt.xlim(0.5,layers*2+0.5)
#plt.ylim(1.1,3.3)
plt.xticks([],[])
#plt.yticks([],[])


plt.tight_layout()
plt.savefig('figures/step4_AccsBlock_diff_att.png')
plt.savefig('figures/step4_AccsBlock_diff_att.svg')

plt.figure(figsize=(16,4))
for i in range(layers):
    plt.plot([1.10+2*i, 1.10+2*i],[np.mean(retro_acc_per_s['low'][i])-stats.sem(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['low'][i])+stats.sem(retro_acc_per_s['low'][i])],c='k',lw=1.0)
    plt.plot([2.10+2*i, 2.10+2*i],[np.mean(retro_acc_per_s['high'][i])-stats.sem(retro_acc_per_s['high'][i]), np.mean(retro_acc_per_s['high'][i])+stats.sem(retro_acc_per_s['high'][i])],c='k',lw=1.0)
    plt.plot([1.10+2*i,2.10+2*i], [np.mean(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['high'][i])],'k--',label='Retrospective')
    plt.scatter([1.10+2*i,2.10+2*i], [np.mean(retro_acc_per_s['low'][i]), np.mean(retro_acc_per_s['high'][i])],c='k')
    plt.plot([1+2*i, 1+2*i],[np.mean(prosp_acc_per_s['low'][i])-stats.sem(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['low'][i])+stats.sem(prosp_acc_per_s['low'][i])],c='k',lw=1.0)
    plt.plot([2+2*i, 2+2*i],[np.mean(prosp_acc_per_s['high'][i])-stats.sem(prosp_acc_per_s['high'][i]), np.mean(prosp_acc_per_s['high'][i])+stats.sem(prosp_acc_per_s['high'][i])],c='k',lw=1.0)
    plt.plot([1+2*i,2+2*i], [np.mean(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['high'][i])],'k',label='Prospective')
    plt.scatter([1+2*i,2+2*i], [np.mean(prosp_acc_per_s['low'][i]), np.mean(prosp_acc_per_s['high'][i])],c='k')
#plt.xlabel('Layer: '+str(i))

plt.xlim(0.5,layers*2+0.5)
#plt.ylim(1.1,3.3)
plt.xticks([],[])
plt.yscale('log')
#plt.yticks([],[])

plt.tight_layout()
plt.savefig('figures/step4_AccsBlock_diff_att_fig2.png')
plt.savefig('figures/step4_AccsBlock_diff_att_fig2.svg')


plt.show()


print('OK')





#
