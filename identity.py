import logging
import multiprocessing as mp
logger = logging.getLogger('MLP_training')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import skewnorm
import numpy as np
import random
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


def identity(inputs):
	"""Calculate identity between two sequences"""
	(A,B) = inputs
	length = 0.0
	idents = 0.0
	for z in range(0,len(A),1):
		if ((A[z] != '-') or (A[z] != '-')):
			length = length +1
			if (A[z] == B[z]):
				idents = idents+1				
	if length>0:
		return 100*idents/length
	else:
		return np.nan

def comparitor(test,train_seqs):
	"""Find the maximum identity between two sequences"""
	pairs = [(test,x) for x in train_seqs]
	result = map(identity,pairs)
	return max(list(result))	

def all_comparitor(input):
	"""Find the maximum identity between two sequences"""
	(pos,seqs) = input
	pairs = [(seqs[pos],seqs[y]) for y in range(pos+1,len(seqs),1)]
	return list(map(identity,pairs))

pairwise = {}

def compare_all(MSA,verbose):
	N=len(MSA)
	reduced = [str(MSA[x].seq) for x in range(0,N,1)]
	logger.info('Number of sequences: '+str(N))
	shuffled_positions = [(y,reduced) for y in range(0,N,1)]
	random.shuffle(shuffled_positions)
	if verbose:
		print('Calculating sequences identity')
		results = [all_comparitor(y) for y in tqdm(shuffled_positions,unit='seqs')]
	else:
		results = [all_comparitor(y) for y in shuffled_positions]
	results = [item for sublist in results for item in sublist]
	pairs = len(results)
	logger.info('The number of sequence pairs: '+str(pairs))

	#logger.info('Fitting the skew-normal distribution to the MSA pairwise identities.')
	#ae, loce, scalee = skewnorm.fit(results)
	std = np.std(results)
	mean = np.mean(results)
	logger.info('Pairwise squence identity mean: '+str(mean))
	logger.info('Pairwise squence identity stdev: '+str(std))
	#logger.info('Pairwise squence identity skew-normal skewness parameter: '+str(ae))

	plt.cla()
	plt.clf()
	plt.close()
	n,bins,patches=plt.hist(results,100,density=False)
	plt.xlim(-5,105)
	plt.xticks([0,25,50,75,100])
	plt.rc('font')
	plt.ylabel('Number of pairs')
	plt.xlabel('Identity (%)')
	#plt.title('Total sequences: '+str(N)+' pairs: '+str(pairs)+'\nmean: '+str(mean)[:6]+', stdev: '+str(std)[:6]+', skew-normal a: '+str(ae)[:6])
	plt.title('Total sequences: '+str(N)+' pairs: '+str(pairs)+'\nmean: '+str(mean)[:6]+', stdev: '+str(std)[:6])
	plt.savefig('All_sequences_identity_histogram.png')		
	plt.cla()
	plt.clf()
	plt.close()	
	#return {'mean':mean,'std':std,'skew':ae}
	return {'mean':mean,'std':std}



def accuracy(train,test,preds,unit,directory,verbose,parallel):
	"""Calculate test prediction accuracy versus maximum identity to training dataset"""
	logger.info('Calculate test prediction accuracy versus maximum identity to training dataset')
	train_reduced = set([str(x.seq) for x in train])
	test_reduced = set([str(x.seq) for x in test])
	global pairwise
	if verbose:
		print('Comparing training and test sequences')
		results = {y:comparitor(y,train_reduced) for y in tqdm(test_reduced,unit='seqs')}
	else:
		results = {y:comparitor(y,train_reduced) for y in test_reduced}
	test_keys = [test[x].id for x in range(0,len(test),1)]
	test_idents = [results[str(test[x].seq)] for x in range(0,len(test),1)]
	test_preds = [preds[x] for x in test_keys]
	test_abs_preds = [abs(preds[x]) for x in test_keys]
	regr = LinearRegression(copy_X=True,fit_intercept=True,n_jobs=parallel).fit([[x] for x in test_idents],[[x] for x in test_abs_preds])
	m = regr.coef_
	b = regr.intercept_
	logger.info('Linear relationship slope between maximum train-test identity and absolute residual: '+str(m[0][0]))
	logger.info('Linear relationship intercept between maximum train-test identity and absolute residual: '+str(b[0]))
	r = pearsonr(test_idents,test_abs_preds)[0]
	logger.info('Correlation between maximum train-test identity and absolute residual: '+str(r))
	plt.cla()
	plt.clf()
	plt.close()
	plt.plot(test_idents,test_abs_preds,'.',markersize=4,color='#222299',alpha=0.1)
	plt.xlim(-5,105)
	plt.xticks([0,25,50,75,100])
	plt.rc('font')
	plt.title('Slope: '+str(m[0][0])+' Pearson r: '+str(r)[:6])
	plt.ylabel('|d'+unit+'|')
	plt.xlabel('Maximum Identity (%)')
	plt.savefig(directory+'/train_test_identity_vs_absolute_test_accuracy.png')		
	plt.cla()
	plt.clf()
	plt.close()	

	plt.cla()
	plt.clf()
	plt.close()
	n,bins,patches=plt.hist(test_idents,50,density=False)
	plt.xlim(-5,105)
	plt.xticks([0,25,50,75,100])
	plt.rc('font')
	plt.ylabel('Number of pairs')
	plt.xlabel('Maximum Identity (%)')
	plt.savefig(directory+'/train_test_identity_histogram.png')		
	plt.cla()
	plt.clf()
	plt.close()	

	
	with open(directory+'/max_identity_vs_test_accuracy.tsv','w') as g:
		g.write('sequence_id\tmaximum_identity\td'+unit+'\n')
		for x in range(0,len(test_keys),1):
			g.write(test_keys[x]+'\t'+str(test_idents[x])+'\t'+str(test_preds[x])+'\n')
