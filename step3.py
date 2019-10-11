#!/usr/bin/env python3
import logging
logger = logging.getLogger('MLP_training')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('Step3.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import sys
logger.info('Python version: '+sys.version)
import platform
logger.info('Platform: '+platform.platform())
del(platform)
import Bio
logger.info('Biopython version: '+Bio.__version__)
del Bio
import multiprocessing as mp
logger.info('Number of logical CPU cores: '+str(mp.cpu_count()))
from Bio import AlignIO
import converter
import random
import os
import matplotlib
logger.info('MatPlotLib version: '+matplotlib.__version__)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
logger.info('NumPy version: '+np.__version__)
import balancer
import shutil
import math
from tqdm import tqdm
import biserial
import tophat
import argparse
from Bio.Align import MultipleSeqAlignment
import tfMLP as MLP
from scipy.stats import wilcoxon
from scipy.stats import chisquare
import networks
import identity as identity_calc


#defaults
overdetermined_level =1
max_depth = 5
max_num = 500
G = 10
balanced = False
th_threshold = None
bs_threshold = None
parallel = False
max_layer = float('inf')
identity = False
to_keep = 0.2
unique = False
identity_test = False
unit = 'Tg'
efficient = False

parser = argparse.ArgumentParser(description='Step 3. One-hot encode the protein sequences. Optionally remove less significant columns and/or rebalance the data. Calculate a linear regression and all possible MLPs.')
parser.add_argument("-tr","--train",action='store', type=str, help="The MSA training file in FASTA format.",dest='train_file',default=None)
parser.add_argument("-te","--test",action='store', type=str, help="The MSA testing file in FASTA format.",dest='test_file',default=None)
parser.add_argument("-v", "--validation", type=str,help="The MSA validation file in FASTA format.",action="store",dest='val_file',default=None)
parser.add_argument("-o", "--overdetermined",action='store', type=float, help="The overdetermined level required of the MLPs. Default is "+str(overdetermined_level)+'.',dest='overdetermined',default=overdetermined_level)
parser.add_argument("-ld", "--max_depth",action='store', type=int, help="The maximum number of hidden layers (depth) in the MLPs. Default is "+str(max_depth)+'.',dest='max_depth',default=max_depth)
parser.add_argument("-lw", "--max_width",action='store', type=int, help="The maximum number of nodes (width) in a layer of the MLPs. Default is "+str(max_layer)+'.',dest='max_layer',default=max_layer)
parser.add_argument("-n", "--max_MLP",action='store', type=int, help="The maximum number of MLPs to train per generation. Default is "+str(max_num)+'.',dest='max_num',default=max_num)
parser.add_argument("-b", "--balance",action='store_true', help="The balance the training MSA. Default is "+str(balanced)+'.',dest='balance',default=balanced)
threshes = parser.add_mutually_exclusive_group()
threshes.add_argument("-th", "--th_threshold",action='store', type=float, help="Threshold for the tophat fit correlation coefficients. Used to exclude columns in the MSA. Default is "+str(th_threshold)+'.',dest='th_threshold',default=th_threshold)
threshes.add_argument("-pb", "--pb_threshold",action='store', type=float, help="Threshold for the point-biserial correlation coefficients. Used to exclude columns in the MSA. Default is "+str(bs_threshold)+'.',dest='bs_threshold',default=bs_threshold)
parser.add_argument("-p", "--parallel",help="Run parallel where-ever possible. Avoid using if there are errors (sometimes seen in the labels of plots). Default is "+str(parallel),action="store_true",dest='parallel',default=parallel)
parser.add_argument("-g", "--generations",type=int,help="Number of generations to run the genetic algorithm. Default is "+str(G),dest='G',default=G)
parser.add_argument("-k", "--keep",type=float,help="Fraction of the MLPs to keep every generation. Default is "+str(to_keep),dest='to_keep',default=to_keep)
parser.add_argument("-i", "--identity",help="Train regressions with an identity activation function for comparison. Default is "+str(identity),action="store_true",dest='identity',default=identity)
parser.add_argument("-uo", "--unique_overdetermined",help="Use the number of unique training sequences in calculating the overdetermined level. Default is "+str(unique),action="store_true",dest='unique',default=unique)
parser.add_argument("-ti", "--train_identity",help="Compare the test prediction residual versus maximum identity to the training dataset. Default is "+str(identity_test),action="store_true",dest='identity_test',default=identity_test)
parser.add_argument("-u", "--unit",help="Description/unit for the regression target. Default is "+str(unit),action="store",dest='unit',default=unit)
parser.add_argument("-e", "--efficient",help="Use unsigned 8-bit intergers for the one-hot encoded protein sequence, rather than standard 32-bit floats. This will reduced memory use ~75%. Default is "+str(efficient),action="store_true",dest='efficient',default=efficient)


args = parser.parse_args()
training_file = args.train_file
testing_file = args.test_file
val_file = args.val_file
overdetermined_level = args.overdetermined
max_depth = args.max_depth
max_num = args.max_num
balanced = args.balance
th_threshold = args.th_threshold
bs_threshold = args.bs_threshold
parallel = args.parallel
max_layer = args.max_layer
G = args.G
to_keep = args.to_keep
identity = args.identity
unique = args.unique
identity_test = args.identity_test
unit = args.unit
efficient = args.efficient

#log the run parameters
logger.info('Description/unit of the regression target: '+str(unit))
logger.info('Fold Overdetermined: '+str(overdetermined_level))
logger.info('Maximum MLP depth: '+str(max_depth))
logger.info('Maximum number of MLPs per generation: '+str(max_num))
logger.info('Fraction of the MLPs to keep per generation: '+str(to_keep))
logger.info('Maximum number of nodes in a layer: '+str(max_layer))
logger.info('Number of generations: '+str(G))
logger.info('Balancing: '+str(balanced))
logger.info('Tophat r-score Z-score column removal threshold: '+str(th_threshold))
logger.info('Biserial r-score Z-score column removal threshold: '+str(bs_threshold))
logger.info('Running parallel: '+str(parallel))
logger.info('Training MLP with identity activation function: '+str(identity))
logger.info('The training file is: '+str(training_file))
logger.info('The test file is: '+str(testing_file))
logger.info('The validation file is: '+str(val_file))
logger.info('Use unique training sequences to calculate overdetermined level: '+str(unique))
logger.info('Calculate the pairwise percent identity versus difference in '+unit+' between the training and test sets: '+str(identity_test))
logger.info('Use unsigned 8-bit intergers for the one hot encoded sequence: '+str(efficient))

files = {}
for file in [('train',training_file),('test',testing_file),('valid',val_file)]:
	try:
		files[file[0]] = AlignIO.read(file[1],'fasta')
	except:
		logger.info('Unreadable '+file[0]+' MSA file, provided as '+file[1])
		print('Unreadable '+file[0]+' MSA file, provided as '+file[1])
		sys.exit()
	else:
		pass	
if os.path.isdir('./results'):
	shutil.rmtree('./results')
if not(os.path.isdir('./results')):
	os.makedirs('./results')

#calculate total number of inputs possible
length = files['train'].get_alignment_length()
input_values = 0
absolutely_conserved =0
for y in range(0,length,1):
	used = {}
	for x in files['train']:
		used[x.seq[y]]=True
	if len(used.keys())>1:
		input_values = input_values + len(used.keys())
	else:
		absolutely_conserved = absolutely_conserved +1

training_seqs = len(files['train'])
logger.info('Training sequences: '+str(training_seqs))
logger.info('Test sequences: '+str(len(files['test'])))
logger.info('Validation sequences: '+str(len(files['valid'])))
refs = set([x.seq for x in files['train']]+[x.seq for x in files['valid']])
logger.info('Number of test sequences in the train or validation datasets: '+str(len([y.seq for y in files['test'] if y.seq in refs])))
logger.info('Unique training sequences: '+str(len(set([x.seq for x in files['train']]))))
logger.info('Unique test sequences: '+str(len(set([x.seq for x in files['test']]))))
logger.info('Unique validation sequences: '+str(len(set([x.seq for x in files['valid']]))))
logger.info('Number of unique test sequences in the train or validation datasets: '+str(len(set([y.seq for y in files['test'] if y.seq in refs]))))
del(refs)

if unique:
	logger.info('Using the number of unique training sequences in calculating the overdetermined level')
	training_seqs = len(set([x.seq for x in files['train']]))


#some basic information about the input sequences
logger.info('Alignment length: '+str(length))
logger.info('Absolutely conserved positions: '+str(absolutely_conserved))
logger.info('Traning MSA input values: '+str(input_values))
logger.info('Mean number of AA seen per variable position: '+str(input_values/(float(length)-absolutely_conserved)))
logger.info('Mean number of sequences per variable position: '+str(training_seqs/(float(length)-absolutely_conserved)))

#testif AAs outside of the training set are in the testing or testing sets (possible, especially if the total dataset is small)
logger.info('Finding AAs in testing/validation datasets, but not training datasets.')
for y in range(0,length,1):
	used = {}
	warned = {}
	for x in files['train']:
		used[x.seq[y]]=True
	for x in files['test']:
		if not(x.seq[y] in used.keys()):
			if not(x.seq[y] in warned.keys()):
				logger.info('Warning: '+x.seq[y]+' at column '+str(y+1)+' is in the testing set, but not in the training set.')
				warned[x.seq[y]] = True
	for x in files['valid']:
		if not(x.seq[y] in used.keys()):
			if not(x.seq[y] in warned.keys()):
				logger.info('Warning: '+x.seq[y]+' at column '+str(y+1)+' is in the validation set, but not in the training set.')
				warned[x.seq[y]] = True	

#convert the MSAs to binary lists of AA sequences (excluding invariant positions)
logger.info('One-hot encoding the protein sequences')
onehot_data = converter.convert_to_pd(files,parallel,efficient)
logger.info('The mean training '+unit+': '+str(np.mean(onehot_data['train']['target'])))

#calculate biserial r-values only if threshold given
if not(th_threshold == None): 
	onehot_data = tophat.calc(th_threshold,onehot_data,parallel,unit)
elif not(bs_threshold == None):
	onehot_data = biserial.biserial(bs_threshold,onehot_data,parallel,unit)

#make a figure of OGT values
n,bins,patches=plt.hist(onehot_data['train']['target'],50,density=False,label='Training')
chi_sq = chisquare(n)[0]
logger.info('The chi-square of the '+unit+' distribution versus equal values across all 50 bins: '+str(chi_sq))
plt.hist(onehot_data['test']['target'],bins,density=False,label='Testing')
plt.hist(onehot_data['valid']['target'],bins,density=False,label='Validation')
plt.legend()
plt.title('Chi_sq = '+str(chi_sq))
plt.xlabel(unit+' values')
plt.ylabel('Number of Sequences')
plt.savefig('./results/'+unit+'_histogram.png')
plt.cla()
plt.clf()
plt.close()

#make a figure of OGT values
n,bins,patches=plt.hist(onehot_data['train']['target'],50,density=False,label='Training')
plt.hist(onehot_data['test']['target'],bins,density=False,label='Testing')
plt.hist(onehot_data['valid']['target'],bins,density=False,label='Validation')
plt.yscale('log')
plt.legend()
plt.title('Chi_sq = '+str(chi_sq))
plt.xlabel(unit+' values')
plt.ylabel('Number of Sequences')
plt.savefig('./results/'+unit+'_histogram_log.png')
plt.cla()
plt.clf()
plt.close()

if balanced:
	onehot_data = balancer.overweight(onehot_data)

	#make a figure of balanced OGT values
	n,bins,patches=plt.hist(onehot_data['train']['target'],50,density=False,label='Balanced Training')
	plt.hist(onehot_data['test']['target'],bins,density=False,label='Testing')
	plt.hist(onehot_data['valid']['target'],bins,density=False,label='Validation')
	plt.legend()
	plt.xlabel(unit+' values')
	plt.ylabel('Number of Sequences')
	plt.savefig('./results/'+unit+'_histogram_balanced.png')
	plt.cla()
	plt.clf()
	plt.close()

	#make a figure of balanced OGT values
	n,bins,patches=plt.hist(onehot_data['train']['target'],50,density=False,label='Balanced Training')
	plt.hist(onehot_data['test']['target'],bins,density=False,label='Testing')
	plt.hist(onehot_data['valid']['target'],bins,density=False,label='Validation')
	plt.yscale('log')
	plt.legend()
	plt.xlabel(unit+' values')
	plt.ylabel('Number of Sequences')
	plt.savefig('./results/'+unit+'_histogram_balanced_log.png')
	plt.cla()
	plt.clf()
	plt.close()

	logger.info('Number of training sequences after balancing: '+str(onehot_data['train']['target'].shape[0]))
	logger.info('The mean training '+unit+' after balancing: '+str(np.mean(onehot_data['train']['target'])))


logger.info('Training sequences minimum: '+str(onehot_data['train']['target'].min()))
logger.info('Training sequences maximum: '+str(onehot_data['train']['target'].max()))
logger.info('Testing sequences minimum: '+str(onehot_data['test']['target'].min()))
logger.info('Testing sequences maximum: '+str(onehot_data['test']['target'].max()))
logger.info('Validation sequences minimum: '+str(onehot_data['valid']['target'].min()))
logger.info('Validation sequences maximum: '+str(onehot_data['valid']['target'].max()))
logger.info('Total size of data (in MB): '+str((sum(onehot_data['train'].memory_usage())+sum(onehot_data['test'].memory_usage())+sum(onehot_data['valid'].memory_usage()))/(1024**2)))

logger.info('Setting up common training parameters')
MLP.setup(onehot_data,unit)

n_input = onehot_data['train'].shape[1]-2

if (max_layer > 2*(n_input)):
	logger.info('Limiting the layer width based on the width of the input layer')
	max_layer = 2*n_input
	logger.info('New maximum number of nodes per layer: '+str(max_layer))

logger.info('Maximum number of nodes in a layer: '+str(max_layer))

#find all valid NNs
logger.info('Building out MLP topologies to train')
(brute,population)= networks.builder(G,max_num,max_depth,overdetermined_level,n_input,training_seqs,max_layer,parallel)

logger.info('Calculating linear regression')
#run linear regression
if (onehot_data['train'].shape[1]-1)>onehot_data['train'].shape[0]:
	logger.info('Warning! Linear regression will be underdetermined. More values to fit than sequences in training set.')
lin = MLP.trainer((None,'Linear'))
logger.info('Linear regression gives validation MSE of: '+str(lin['MSE']))
lin = MLP.final_eval((None,'Linear'))
logger.info('Linear regression gives MSE of: '+str(lin['MSE'])+', RMSE of: '+str(lin['RMSE'])+', r-value: '+str(lin['r'])+', data/param: '+str(lin['data_param']))
f = open('./results/NN_results.tsv','w')
f.write('Network\tLayers\tConnections\tdata/param\tMSE\tRMSE\tr-value')
if identity:
	f.write('\tident_MSE\tident_RMSE\tident_r\n')
else:
	f.write('\n')
f.write('linear\t1\t'+str(onehot_data['train'].shape[1]-1)+'\t'+str(lin['data_param'])+'\t'+str(lin['MSE'])+'\t'+str(lin['RMSE'])+'\t'+str(lin['r'])+'\n')

if max_num == 0:
	#catch if not training any MLPs
	logger.info('Not training any MLPs.')

elif len(population)>0:
	all_results_relu =[]
	all_results_ident =[]
	if not(brute):
		print('Training generation: 0')
		#train
		results = [MLP.trainer((NN,'ReLu')) for NN in tqdm(population,unit='MLP')]
		all_results_relu.append(results)
		if identity:
			all_results_ident.append([MLP.trainer((NN,'Identity')) for NN in tqdm(population,unit='MLP')])
		sorted_results = sorted(results, key=lambda k: k['MSE'])
		best = sorted_results[0]
		#move best MLP
		os.mkdir('./results/MLP_best')
		shutil.copyfile('./results/MLP_'+'-'.join([str(x) for x in best['NN']])+'/ReLu/model.h5', './results/MLP_best/model.h5')
		logger.info('The best MLP of generation 0 has a topology of '+'-'.join([str(x) for x in best['NN']])+' validation dataset MSE of '+str(sorted_results[0]['MSE']))
		for nn in results:
			shutil.rmtree('./results/MLP_'+'-'.join([str(x) for x in nn['NN']]))
		for g in range(1,G,1):
			print('Training generation: '+str(g))
			#select the top to_keep%, recombine and mutate
			population = sorted_results[:int(len(sorted_results)*to_keep)+1]
			population = [x['NN'] for x in population]
			population = population + networks.recombine(population) + networks.mutate(population)
			#remove identical MLPs
			old_population = population
			population=[]
			for x in old_population:
				if not(x in population):
					population.append(x)
			#generate a population of neural networks.
			population = random.sample(population,min([len(population),max_num]))
			#train
			results = [MLP.trainer((NN,'ReLu')) for NN in tqdm(population,unit='MLP')]
			all_results_relu.append(results)
			if identity:
				all_results_ident.append([MLP.trainer((NN,'Identity')) for NN in tqdm(population,unit='MLP')])
			sorted_results = sorted(results, key=lambda k: k['MSE'])

			#find the new best if it exists
			if sorted_results[0]['MSE']<best['MSE']:
				best= sorted_results[0]
				os.remove('./results/MLP_best/model.h5')
				shutil.copyfile('./results/MLP_'+'-'.join([str(x) for x in best['NN']])+'/ReLu/model.h5', './results/MLP_best/model.h5')
			logger.info('The best MLP of generation '+str(g)+' has a topology of '+'-'.join([str(x) for x in sorted_results[0]['NN']])+' validation dataset MSE of '+str(sorted_results[0]['MSE']))

			#remove non-best MLPs
			for nn in results:
				shutil.rmtree('./results/MLP_'+'-'.join([str(x) for x in nn['NN']]))

	else:
		print('Using brute force to predict all possible MLP topologies.')
		#brute force
		results = results = [MLP.trainer((NN,'ReLu')) for NN in tqdm(population,unit="MLP")]
		all_results_relu = results
		if identity:
			all_results_ident = [MLP.trainer((NN,'Identity')) for NN in tqdm(population,unit='MLP')]
		sorted_results = sorted(results, key=lambda k: k['MSE']) 
		best = sorted_results[0]
		#move best MLP
		os.mkdir('./results/MLP_best')
		shutil.copyfile('./results/MLP_'+'-'.join([str(x) for x in best['NN']])+'/ReLu/model.h5', './results/MLP_best/model.h5')
		for nn in results:
			shutil.rmtree('./results/MLP_'+'-'.join([str(x) for x in nn['NN']]))

	#train and evaluate on best NN topology
	result = MLP.final_eval((best['NN'],'ReLu'))

	logger.info('Network of topology: '+'-'.join([str(y) for y in best['NN']])+' gives test MSE of: '+str(result['MSE'])+', RMSE of: '+str(result['RMSE'])+', r-value:'+str(result['r']))
	logger.info('Equations (sequences) to unknowns (connections) ratio: '+str(result['data_param']))

	#calculate how accuracy scales with sequence similarity
	if identity_test:
		identity_calc.accuracy(files['train'],files['test'],result['residuals'],unit,parallel)

	#plot r's by generation
	all_relu_rs = []
	all_ident_rs = []
	generation = []
	for y in range(0,len(all_results_relu),1):
		all_relu_rs = all_relu_rs + [x['r'] for x in all_results_relu[y]]
		generation = generation+ [y]*len(all_results_relu[y])
		if identity:
			all_ident_rs = all_ident_rs + [x['r'] for x in all_results_ident[y]]
	plt.axhline(y=lin['r'],color='black')
	if identity:
		plt.plot(generation,all_ident_rs,'.',label='Identity',markersize=14,alpha=0.6)
	plt.plot(generation,all_relu_rs,'.',label='ReLu',markersize=14,alpha=0.6)
	plt.xlim(-1,generation[-1]+1)
	plt.ylim(0.8*min(all_relu_rs+all_ident_rs+[lin['r']]),1.2*max(all_relu_rs+all_ident_rs+[lin['r']]))
	plt.grid()
	plt.xlabel('Generation')
	plt.ylabel('r of Validation Data')
	plt.legend()
	plt.savefig('./results/MLP_r_vs_generation.png')
	plt.cla()
	plt.clf()
	plt.close()	


	#plot mse by generation
	all_relu_mses = []
	all_ident_mses = []
	for y in range(0,len(all_results_relu),1):
		all_relu_mses = all_relu_mses + [x['MSE'] for x in all_results_relu[y]]
		if identity:
			all_ident_mses = all_ident_mses + [x['MSE'] for x in all_results_ident[y]]
	plt.axhline(y=lin['MSE'],color='black')
	if identity:
		plt.plot(generation,all_ident_mses,'.',label='Identity',markersize=14,alpha=0.6)
	plt.plot(generation,all_relu_mses,'.',label='ReLu',markersize=14,alpha=0.6)
	plt.xlim(-1,generation[-1]+1)
	plt.ylim(0.8*min(all_relu_mses+all_ident_mses+[lin['MSE']]),1.2*max(all_relu_mses+all_ident_mses+[lin['MSE']]))
	plt.grid()
	plt.xlabel('Generation')
	plt.ylabel('MSE of Validation Data')
	plt.legend()
	plt.savefig('./results/MLP_MSE_vs_generation.png')
	plt.cla()
	plt.clf()
	plt.close()	


	#unroll results for plotting
	temp_relu = []
	temp_ident = []
	for y in range(0,len(all_results_relu),1):
		temp_relu = temp_relu + all_results_relu[y]
		if identity:
			temp_ident = temp_ident + all_results_ident[y]
	all_results_relu = temp_relu
	all_results_ident = temp_ident

	#record the accuracies for all examined topologies
	for x in range(0,len(all_results_relu),1):
		net = '-'.join([str(x) for x in all_results_relu[x]['NN']])
		f.write(net+'\t'+str(len(all_results_relu[x]['NN']))+'\t'+str(MLP.connections(all_results_relu[x]['NN']))+'\t'+str(all_results_relu[x]['data_param'])+'\t'+str(all_results_relu[x]['MSE'])+'\t'+str(all_results_relu[x]['RMSE'])+'\t'+str(all_results_relu[x]['r']))
		if identity:
			f.write('\t'+str(all_results_ident[x]['MSE'])+'\t'+str(all_results_ident[x]['RMSE'])+'\t'+str(all_results_ident[x]['r'])+'\n')
		else:
			f.write('\n')

	#plot accuracies for all examined topologies
	#plot the MSE vs number of connections
	plt.axhline(y=lin['MSE'],color='black')
	if identity:
		plt.plot([MLP.connections(x['NN']) for x in all_results_ident],[x['MSE'] for x in all_results_ident],'.',label='Identity',markersize=14,alpha=0.6)
	plt.plot([MLP.connections(x['NN']) for x in all_results_relu],[x['MSE'] for x in all_results_relu],'.',label='ReLu',markersize=14,alpha=0.6)
	plt.xlim(0,max([MLP.connections(x['NN']) for x in all_results_relu])*1.05)
	plt.ylim(0.5*min([x['MSE'] for x in all_results_relu]+[x['MSE'] for x in all_results_ident]),lin['MSE']+1)
	plt.grid()
	plt.xlabel('Number of Connections')
	plt.ylabel('MSE of Validation Data')
	plt.legend()
	plt.savefig('./results/MLP_MSE_vs_Connection.png')
	plt.cla()
	plt.clf()
	plt.close()	

	#plot the MSE vs number of layers
	plt.axhline(y=lin['MSE'],color='black')
	if identity:
		plt.plot([len(x['NN']) for x in all_results_ident],[x['MSE'] for x in all_results_ident],'.',label='Identity',markersize=14,alpha=0.6)
	plt.plot([len(x['NN']) for x in all_results_relu],[x['MSE'] for x in all_results_relu],'.',label='ReLu',markersize=14,alpha=0.6)
	plt.xlim(0,max([len(x['NN']) for x in all_results_relu])+1)
	plt.ylim(0.5*min([x['MSE'] for x in all_results_relu]+[x['MSE'] for x in all_results_ident]),lin['MSE']+1)
	plt.grid()
	plt.xlabel('Number of Layers')
	plt.ylabel('MSE of Validation Data')
	plt.legend()
	plt.savefig('./results/MLP_MSE_vs_Layers.png')
	plt.cla()
	plt.clf()
	plt.close()	

	#plot the RMSE vs max width
	plt.axhline(y=lin['MSE'],color='black')
	if identity:
		plt.plot([max(x['NN']) for x in all_results_ident],[x['MSE'] for x in all_results_ident],'.',label='Identity',markersize=14,alpha=0.6)
	plt.plot([max(x['NN']) for x in all_results_relu],[x['MSE'] for x in all_results_relu],'.',label='ReLu',markersize=14,alpha=0.6)
	plt.xlim(0,max([max(x['NN']) for x in all_results_relu])+1)
	plt.ylim(0.5*min([x['MSE'] for x in all_results_relu]+[x['MSE'] for x in all_results_ident]),lin['MSE']+1)
	plt.grid()
	plt.xlabel('Max Layer Width')
	plt.ylabel('MSE of Validation Data')
	plt.legend()
	plt.savefig('./results/MLP_MSE_vs_Width.png')
	plt.cla()
	plt.clf()
	plt.close()	

	#plot the RMSE vs overdetermination
	plt.axhline(y=lin['MSE'],color='black')
	if identity:
		plt.plot([training_seqs/MLP.connections(x['NN']) for x in all_results_ident],[x['MSE'] for x in all_results_ident],'.',label='Identity',markersize=14,alpha=0.6)
	plt.plot([training_seqs/MLP.connections(x['NN']) for x in all_results_relu],[x['MSE'] for x in all_results_relu],'.',label='ReLu',markersize=14,alpha=0.6)
	plt.xlim(0.5,max([training_seqs/MLP.connections(x['NN']) for x in all_results_relu])*1.05)
	plt.ylim(0.5*min([x['MSE'] for x in all_results_relu]+[x['MSE'] for x in all_results_ident]),lin['MSE']+1)
	plt.grid()
	plt.xlabel('Overdetermination')
	plt.ylabel('MSE of Validation Data')
	plt.legend()
	plt.savefig('./results/MLP_MSE_vs_overdetermination.png')
	plt.cla()
	plt.clf()
	plt.close()	

	#plot histogram of rmse's
	rmses_relu = [all_results_relu[x]['RMSE'] for x in range(0,len(all_results_relu),1) if not(math.isnan(all_results_relu[x]['RMSE']))]
	hist_range = list(np.linspace(min(rmses_relu)*0.95,max(rmses_relu)*1.05,40))
	plt.axvline(x=lin['RMSE'],color='black')
	if identity:
		rmses_relu = [all_results_relu[x]['RMSE'] for x in range(0,len(all_results_relu),1) if (not(math.isnan(all_results_relu[x]['RMSE'])) and not(math.isnan(all_results_ident[x]['RMSE'])))]
		rmses_lin = [all_results_ident[x]['RMSE'] for x in range(0,len(all_results_ident),1) if (not(math.isnan(all_results_relu[x]['RMSE'])) and not(math.isnan(all_results_ident[x]['RMSE'])))]
		if len(rmses_relu)>0:
			hist_range = list(np.linspace(min(rmses_lin+rmses_relu)*0.95,max(rmses_lin+rmses_relu)*1.05,40))
			plt.hist(rmses_lin,hist_range,density=False,label='Identity',alpha=0.6)
	plt.hist(rmses_relu,hist_range,density=False,label='ReLu',alpha=0.6)
	plt.tick_params(axis='x', which='both',bottom=True,top=False)
	plt.legend()
	plt.xlabel("Trained MLPs RMSE")
	if len(rmses_relu)>20:
		if identity:
			(t,p) = wilcoxon(rmses_relu,rmses_lin)
			plt.title("Wilcoxon p: "+str(p))
			logger.info('The Wilcoxon p-value of the RMSEs using a ReLu versus linear activation function: '+str(p))
	plt.ylabel('Count')
	plt.savefig('./results/MLP_RMSE_histogram.png')
	plt.cla()
	plt.clf()
	plt.close()

	#plot histogram of mse's
	mses_relu = [all_results_relu[x]['MSE'] for x in range(0,len(all_results_relu),1) if not(math.isnan(all_results_relu[x]['MSE']))]
	hist_range = list(np.linspace(min(mses_relu)*0.95,max(mses_relu)*1.05,40))
	plt.axvline(x=lin['MSE'],color='black')
	if identity:
		mses_relu = [all_results_relu[x]['MSE'] for x in range(0,len(all_results_relu),1) if (not(math.isnan(all_results_relu[x]['MSE'])) and not(math.isnan(all_results_ident[x]['MSE'])))]
		mses_lin = [all_results_ident[x]['MSE'] for x in range(0,len(all_results_ident),1) if (not(math.isnan(all_results_relu[x]['MSE'])) and not(math.isnan(all_results_ident[x]['MSE'])))]
		if len(mses_relu)>0:
			hist_range = list(np.linspace(min(mses_lin+mses_relu)*0.95,max(mses_lin+mses_relu)*1.05,40))
			plt.hist(mses_lin,hist_range,density=False,label='Identity',alpha=0.6)
	plt.hist(mses_relu,hist_range,density=False,label='ReLu',alpha=0.6)
	plt.tick_params(axis='x', which='both',bottom=True,top=False)
	plt.legend()
	plt.xlabel("Trained MLPs Validation MSE")
	plt.ylabel('Count')
	if len(mses_relu)>20:
		if identity:
			(t,p) = wilcoxon(mses_relu,mses_lin)
			plt.title("Wilcoxon p: "+str(p))
			logger.info('The Wilcoxon p-value of the MSEs using a ReLu versus linear activation function: '+str(p))
	plt.savefig('./results/MLP_MSE_histogram.png')
	plt.cla()
	plt.clf()
	plt.close()

	#plot histogram of rs's
	rs_relu = [all_results_relu[x]['r'] for x in range(0,len(all_results_relu),1) if not(math.isnan(all_results_relu[x]['r']))]
	hist_range = list(np.linspace(min(rs_relu)*0.95,max(rs_relu)*1.05,40))
	if min(rs_relu) <0:
		#catch if the minimum r-value is negative
		hist_range = list(np.linspace(min(rs_relu)*1.05,max(rs_relu)*1.05,40))		

	if identity:
		rs_relu = [all_results_relu[x]['r'] for x in range(0,len(all_results_relu),1) if (not(math.isnan(all_results_relu[x]['r'])) and not(math.isnan(all_results_ident[x]['r'])))]
		rs_lin = [all_results_ident[x]['r'] for x in range(0,len(all_results_ident),1) if (not(math.isnan(all_results_relu[x]['r'])) and not(math.isnan(all_results_ident[x]['r'])))]
		if len(rs_relu)>0:
			hist_range = list(np.linspace(min(rs_lin+rs_relu)*0.95,max(rs_lin+rs_relu)*1.05,40))
			if min(rs_relu+rs_lin) <0:
				#catch if the minimum r-value is negative
				hist_range = list(np.linspace(min(rs_lin+rs_relu)*1.05,max(rs_lin+rs_relu)*1.05,40))		
	
	plt.axvline(x=lin['r'],color='black')
	if identity:
		plt.hist(rs_lin,hist_range,density=False,label='Identity',alpha=0.6)
	plt.hist(rs_relu,hist_range,density=False,label='ReLu',alpha=0.6)
	plt.tick_params(axis='x', which='both',bottom=True,top=False)
	plt.legend()
	plt.xlabel("Trained MLPs Validation r-values")
	plt.ylabel('Count')
	if len(rs_relu)>20:
		if identity:
			(t,p) = wilcoxon(rs_relu,rs_lin)
			plt.title("Wilcoxon p: "+str(p))
			logger.info('The Wilcoxon p-value of the Pearson r-values using a ReLu versus linear activation function: '+str(p))
	plt.savefig('./results/MLP_r_value_histogram.png')
	plt.cla()
	plt.clf()
	plt.close()	

else:
	logger.info('Insufficient sequences to train a MLP')
f.close()

logger.info('Finished normally')
