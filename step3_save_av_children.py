# Arguments
import pickle, csv
from model.attention_system import *
from model.episodic_memory import * # tree structure stuff of the particular experience
from model.semantic_memory import * # bag of words with trees etc

all_average_children = []

semantic_memory = pickle.load(open('semantic_memory.pkl','rb'))

params = {'low':(9.0, 1), 'high':(11.0, 130)} # tau, effort

verbose = False
TRIAL_FIRST = 1
TRIAL_LAST = 4290

for trial in range(TRIAL_FIRST,TRIAL_LAST):
    try:
        for mode in ['low', 'high']:
            attention = AttentionSystem('parameters.json')
            episodic_memory = EpisodicMemory('parameters.json')
            all_surprises = pickle.load(open('trials_surpr/surprises_'+str(trial)+'.pkl','rb'))

            attention.tau = params['low'][0] +(params['high'][0] - params['low'][0])*np.random.rand()

            # Acts as dummy activations..!
            dummy_activations = all_surprises[0]
            duration = float(len(all_surprises))/30.0

            episodic_memory.initialise_tree(dummy_activations)

            for t in range(len(all_surprises)):
                for layer in range(len(dummy_activations)):
                    # Use the attention mechanism to calculate thresholds
                    # NOTE: It updates: 'attention.thresholds' and 'attention.surprises'
                    attention.update(all_surprises[t][layer], layer)

                    # Update episodic memory: If interesting, save the current activation pattern!
                    episodic_memory.decay(layer)
                    if attention.surprises[layer] > 0.0 and layer != (len(episodic_memory.all_nodes)-1):
                        # Add the new nodes
                        episodic_memory.add_node(layer, dummy_activations, semantic_memory.last_prior_index[layer], attention.surprises[layer])

            if len(all_average_children) == 0:
                for i in range(len(episodic_memory.av_children)):
                    all_average_children.append([])
            for i in range(len(episodic_memory.av_children)):
                all_average_children[i].append(episodic_memory.av_children[i])
    except:
        pass

all_av_children = np.zeros(len(all_average_children))

for i in range(len(all_av_children)):
    all_av_children[i] = np.mean(all_average_children[i])

with open('all_av_children.pkl','wb') as ff:
    pickle.dump(all_av_children, ff)
