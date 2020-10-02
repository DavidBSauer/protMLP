import logging
logger = logging.getLogger('MLP_training')
from Bio.Align import MultipleSeqAlignment
import numpy as np
import random

bins = 20

def overweight(all_data):
	"""Balance a provided dataset into 20 bins"""
	hist,bin_edges = np.histogram(all_data['train']['target'].values,bins)
	maximum = hist.max()
	logger.info('Maximum number of sequences in any bin: '+str(maximum))

	choices = []
	width = bin_edges[1]-bin_edges[0]
	for bin in range(0,len(bin_edges),1):
		if bin ==0:
			#if first bin
			valid_seqs = [index for index, row in all_data['train'].iterrows() if (bin_edges[bin]<=row['target']<bin_edges[bin]+width)]
			logger.info('Number of training sequences initially in bin '+str(bin_edges[bin])[0:5]+' to '+str(bin_edges[bin]+width)[0:5]+': '+str(len(valid_seqs)))
		elif bin == len(bin_edges)-1:
			#if last bin
			valid_seqs = [index for index, row in all_data['train'].iterrows() if (bin_edges[bin]<=row['target']<bin_edges[bin]+width+1)]
			logger.info('Number of training sequences initially in bin '+str(bin_edges[bin])[0:5]+' to '+str(bin_edges[bin]+width+1)[0:5]+': '+str(len(valid_seqs)))		
		else:
			#all other bins
			valid_seqs = [index for index, row in all_data['train'].iterrows() if (bin_edges[bin]<=row['target']<bin_edges[bin]+width)]
			logger.info('Number of training sequences initially in bin '+str(bin_edges[bin])[0:5]+' to '+str(bin_edges[bin]+width)[0:5]+': '+str(len(valid_seqs)))
		curr_pos = valid_seqs
		if len(valid_seqs)>0: #catch in case of empty bins (empty bins remain empty)	
			#randomly add additional sequences from bin (with replacement) up to the maximum for the bin
			curr_pos=curr_pos+list(np.random.choice(valid_seqs,size=maximum-len(curr_pos),replace=True))
		choices = choices+curr_pos
	#create a new pd array using the randomly selected sequences
	all_data['train']=all_data['train'].loc[choices]
	all_data['train']=all_data['train'].reset_index(drop=True)
	#all_data['train'].to_csv('./results/train_balanced.csv')

	return all_data
