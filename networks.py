import multiprocessing as mp
import random as python_random
import sys
import logging
import regressors
logger = logging.getLogger('MLP_training')

max_depth = float('inf')
max_layer = float('inf')
overdetermined_level = 1
training_seqs = 0
threads = 1



def builder(generations,num_per_generation,depth,level,n_input,train,layer,par,seed):
	"""Build MLP topologies based on maximum width, depth, overdetermined level"""
	global max_depth 
	max_depth = depth
	global overdetermined_level
	overdetermined_level = level
	global training_seqs
	training_seqs = train
	global max_layer
	max_layer = layer 
	global threads
	threads = par
	python_random.seed(seed)

	logger.info('The theoretical maximum number of MLP topologies to build: '+str(float((2*(max_layer-1))**max_depth)))
	logger.info('The maximum number of MLP topologies to train over all generations: '+str(generations*num_per_generation))

	NNs =[]
	if ((2*(max_layer-1))**max_depth<generations*num_per_generation and (2*(max_layer-1))**max_depth<536870912):
		#build brute force if theoretical number is less than max to be sampled by genetic algorithm AND less than the maximum possible list size in python
		logger.info('Generating all possible MLPs')
		NNs = [[2]]
		while not(NNs[-1][-1]==max_layer and len(NNs[-1])==max_depth):
			new_NN = [x for x in NNs[-1]]
			if new_NN[-1]<max_layer:
				new_NN[-1] = new_NN[-1]+1
			else:
				new_NN = new_NN+[2]	
			NNs.append(new_NN)
		NNs = [NN for NN in NNs if training_seqs/(overdetermined_level*regressors.connections(to_add))>=1.0]
		return (True,NNs)
	else:
		#build random subset
		logger.info('Generating a random subset of MLPs')
		if (training_seqs/(overdetermined_level*regressors.connections([2]))<1.0):
			return(True,[])
		max_layer = int(((training_seqs/overdetermined_level)-n_input*2-2)/3) #assuming the max is a two-layer network with 2 as the first layer
		logger.info('The maximum number of nodes in any layer and still within the overdetermined level is: '+str(max_layer))
		while len(NNs)<num_per_generation:
			num_layers = python_random.randint(1,max_depth)
			to_add =[]
			for _ in range(0,num_layers,1):
				to_add.append(python_random.randint(2,max_layer))
			if training_seqs/(overdetermined_level*regressors.connections(to_add))>=1.0:
				if not(to_add in NNs):
					NNs.append(to_add)
		return (False,NNs)
			
#implement a genetic algorithm to search for an optimal topology
def recombiner(input):
	"""Randomly recombine two topologies"""	
	(net1,net2) = input
	joint1 = python_random.randint(0,len(net1)-1)
	joint2 = python_random.randint(0,len(net2)-1)
	new_network = net1[:joint1]+net2[joint2:]
	if training_seqs/(overdetermined_level*regressors.connections(new_network))>=1.0: 
		if max_depth >= len(new_network):
			return (True,new_network)
	return (False,None)

def recombine(possible):
	"""Recombine topologies"""
	pairs = [(possible[x],possible[y]) for x in range(0,len(possible),1) for y in range(0,len(possible),1) if not(x == y)]
	p = mp.Pool(threads)
	results = p.map(recombiner,pairs,1)		
	p.close()
	p.join()
	results = [x[1] for x in results if x[0]]
	return results

def mutater(network):
	"""Randomly mutate a topology"""
	new_network =[x for x in network]
	layer = python_random.randint(0,len(new_network)-1)
	new_network[layer] = python_random.randint(2,max_layer)
	if training_seqs/(overdetermined_level*regressors.connections(new_network))>=1.0:
		return (True,network)
	return (False,None)

def mutate(possible):
	"""Mutate topologies"""
	p = mp.Pool(threads)
	results = p.map(mutater,possible,1)		
	p.close()
	p.join()
	results = [result[1] for result in results if result[0]]
	return results	
