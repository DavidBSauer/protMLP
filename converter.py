#!/usr/bin/env python3
import logging
logger = logging.getLogger('MLP_training')
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import pandas as pd

#convert the sequences to Numpy arrays
AA_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,'-':20}
AAs = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']

def temp(txt):
	a = txt.split('|')
	b = a [-1]
	return float(b)

def plot(input):
	"""plot heatmap of AA by position"""
	(name,alignment) = input
	length = alignment.get_alignment_length()
	num = len(alignment)
	matrix = np.zeros(shape=(length,21))

	AA_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,'-':20}
	for x in alignment:
		for y in range(0, length):
			matrix[y][AA_dict[x.seq[y]]] += 1
	matrix = matrix/num

	new_matrix = matrix.transpose()
	new_matrix = np.flipud(new_matrix)
	(x,length) = new_matrix.shape
	plt.figure(figsize=(33,18), dpi=300)
	ax = plt.axes()
	plt.tick_params(axis='y', which='both',left=False,right=False)
	yticklabels=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']
	pos = np.arange(len(yticklabels))
	pos = pos[::-1]
	ax.set_yticks(pos + (1.0 / 2))
	ax.set_yticklabels(yticklabels)
	font_size = 30
	ax.tick_params(axis='both',labelsize=font_size)
	p = ax.pcolormesh(new_matrix,cmap=cm.coolwarm,vmin=0,vmax=1)
	cbar = plt.colorbar(p)
	cbar.ax.tick_params(labelsize = font_size)
	plt.tight_layout()
	plt.savefig('./results/'+name+'_AA_frequency_heatmap.png')
	plt.clf()
	plt.close()
	return None

def convert(inputs):
	"""One-hot encode sequence"""
	(AA_sequence,length) = inputs
	working_AA_sequence = str(AA_sequence.seq).upper().replace('.','-')
	full_vector = [False]*(length*21)
	for pos in range(0,length,1):
		full_vector[pos*21+AA_dict[working_AA_sequence[pos]]]=True
	return (AA_sequence.id,full_vector,temp(AA_sequence.id))
		
def convert_no_temp(inputs):
	"""One-hot encode sequence without including a reported Tg"""
	(AA_sequence,length) = inputs
	working_AA_sequence = str(AA_sequence.seq).upper().replace('.','-')
	full_vector = [False]*(length*21)
	for pos in range(0,length,1):
		full_vector[pos*21+AA_dict[working_AA_sequence[pos]]]=True
	return (AA_sequence.id,full_vector)


def convert_to_pd(files,parallel,efficient):
	"""Convert an MSA to a Pandas dataframe of one-hot encoded sequences"""
	#plot heatmaps of MSAs
	to_plot = [(key,files[key]) for key in ['train','test','valid']]
	if parallel:
		p = mp.Pool()
		p.map(plot,to_plot)
		p.close()
		p.join()
	else:
		list(map(plot,to_plot))

	#covert MSAs into a series of lists with binary data encoding the sequence of the varying positions
	length = files['train'].get_alignment_length()
	logger.info('Input alignment length: '+str(length))
	logger.info('Giving a (maximum) one-hot encoded alignment length of: '+str(length*21))
	tr_num = float(len(files['train']))

	#calculate a non-binary AA_vector reference
	ref = np.array([y+str(x) for x in range(1,length+1,1) for y in AAs])

	all_pd_data = {x:None for x in files.keys()}
	for file in files.keys():
		#create binary vectors encoding the protein sequence
		logger.info('Converting '+file+' MSA to one-hot encoded MSA')
		to_convert = [(x,length) for x in files[file]]
		if parallel:
			p = mp.Pool()
			results = p.map(convert,to_convert)
			p.close()
			p.join()
		else:
			results = map(convert,to_convert)
		results = list(results)
		random.shuffle(results)
		if efficient:
			all_pd_data[file]=pd.DataFrame(data=np.array([x[1] for x in results],dtype='uint8'),columns =ref) 
		else:
			all_pd_data[file]=pd.DataFrame(data=np.array([x[1] for x in results],dtype='float32'),columns =ref) 
		all_pd_data[file]=all_pd_data[file].assign(id = [x[0] for x in results])
		all_pd_data[file]=all_pd_data[file].assign(target = [x[-1] for x in results])

		#all_pd_data[file].to_csv('./results/'+file+'_raw_encoded_alignment.csv',index=False)
			
	#remove unseen positions in the binary alignments
	## note: this will result in NO AA for test seq sequences if it contains an AA not seen in the training set
	to_remove = [pos for pos in ref[:-1] if ((float(np.sum(all_pd_data['train'][pos])) == 0.0) or (float(np.sum(all_pd_data['train'][pos])) == float(len(files['train']))))]
	for file in files.keys():
		all_pd_data[file]=all_pd_data[file].drop(labels=to_remove,axis=1)
		#all_pd_data[file].to_csv('./results/'+file+'_alignment_only_observed_AAs.csv',index=False)
	
	#write out the NN AA template
	f = open('./results/NN_AA_template.txt','w+')
	for pos in [x for x in ref[:-1] if not(x in to_remove)]:
		f.write(pos+'\n')
	f.close()

	logger.info('After removing invariant columns, the one-hot alignment length is: '+str(len(ref)-len(to_remove)-1))
	logger.info('Training sequences per one-hot encoded length: '+str(len(files['train'])/(len(ref)-len(to_remove)-1)))
	
	return all_pd_data
	
def convert_on_template(file,template,parallel):
	"""Convert an MSA to a Pandas dataframe of one-hot encoded sequences based on a provided template of observed AAs"""
	length = file.get_alignment_length()
	logger.info('Input alignment length: '+str(length))
	logger.info('Giving a (maximum) one-hot encoded alignment length of: '+str(length*21))
	tr_num = float(len(file))

	#calculate a non-binary AA_vector reference
	ref = np.array([y+''+str(x) for x in range(1,length+1,1) for y in AAs])

	#create binary vectors encoding the protein sequence
	logger.info('Converting MSA to one-hot encoded MSA')
	to_convert = [(x,length) for x in file]
	if parallel:
		p = mp.Pool()
		results = p.map(convert_no_temp,to_convert)
		p.close()
		p.join()
	else:
		results = map(convert_no_temp,to_convert)
	results = list(results)
	random.shuffle(results)
	pd_data=pd.DataFrame(data=np.array([x[1] for x in results],dtype='float32'),columns =ref) 
	pd_data=pd_data.assign(id = [x[0] for x in results])
	#pd_data.to_csv('test_alignment.csv',index=False)
			
	#remove unseen positions in the binary alignments
	## note: this will result in NO AA for test seq sequences if it contains an AA not seen in the training set
	to_remove = [pos for pos in ref if not(pos in template)]
	pd_data=pd_data.drop(labels=to_remove,axis=1)

	logger.info('After removing unseen columns, the alignment length is: '+str(len(ref)-len(to_remove)))
	
	return pd_data

