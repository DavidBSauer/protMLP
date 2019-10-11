import logging
import multiprocessing as mp
logger = logging.getLogger('MLP_training')
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from functools import lru_cache

@lru_cache(maxsize =10**6)
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
		return idents/length
	else:
		return np.nan

def comparitor(test,train_seqs,parallel):
	"""Find the maximum identity between two sequences"""
	pairs = [(test,x) for x in train_seqs]
	if parallel:
		p = mp.Pool()
		result = p.map(identity,pairs)
		p.close()
		p.join()
	else:
		result = map(identity,pairs)
	result = max(list(result))*100
	return result	

def accuracy(train,test,preds,unit,parallel):
	"""Calculate test prediction accuracy versus maximum identity to training dataset"""
	logger.info('Calculate test prediction accuracy versus maximum identity to training dataset')
	print('Comparing training and test sequences')
	train_reduced = set([str(train[x].seq) for x in range(0,len(train),1)])
	test_reduced = set([str(test[x].seq) for x in range(0,len(test),1)])

	results = {y:comparitor(y,train_reduced,parallel) for y in tqdm(test_reduced,unit='seqs')}

	test_keys = [test[x].id for x in range(0,len(test),1)]
	test_idents = [results[str(test[x].seq)] for x in range(0,len(test),1)]
	test_preds = [preds[x] for x in test_keys]
	test_abs_preds = [abs(preds[x]) for x in test_keys]
	r = pearsonr(test_abs_preds,test_idents)[0]
	logger.info('Correlation between maximum train-test identity and absolute residual: '+str(r))
	plt.cla()
	plt.clf()
	plt.close()
	plt.plot(test_idents,test_abs_preds,'.',markersize=4,color='#222299',alpha=0.1)
	plt.xlim(-5,105)
	plt.xticks([0,25,50,75,100])
	plt.rc('font')
	plt.title('Pearson r: '+str(r)[:6])
	plt.ylabel('|d'+unit+'|')
	plt.xlabel('Maximum Identity (%)')
	plt.savefig('./results/train_test_identity_vs_absolute_test_accuracy.png')		
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
	plt.savefig('./results/train_test_identity_histogram.png')		
	plt.cla()
	plt.clf()
	plt.close()	

	
	with open('./results/max_identity_vs_test_accuracy.tsv','w') as g:
		g.write('sequence_id\tmaximum_identity\td'+unit+'\n')
		for x in range(0,len(test_keys),1):
			g.write(test_keys[x]+'\t'+str(test_idents[x])+'\t'+str(test_preds[x])+'\n')
