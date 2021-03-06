#calculate the fit of a tophat function to AA distribution (one-hot encoded) vs OGT
from scipy.stats import zscore, pearsonr
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger('MLP_training')
import multiprocessing as mp
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from tqdm import tqdm
import math


def eval(my_input):
	(values,hat_mid,hat_width,target) = my_input
	preds = np.where(hat_mid-hat_width/2<target,1,0) + np.where(target<hat_mid+hat_width/2,1,0)
	preds = np.where(preds == 2, 1, 0)
	#preds = [1 if hat_mid-hat_width/2<x<hat_mid+hat_width/2 else 0 for x in target]
	curr_r = pearsonr(values,preds)[0]
	return (abs(curr_r),hat_mid,hat_width,curr_r)

def comparison(my_input):
	"""Fit the presence or absence of a particular AA to the top-hat function"""
	(name,values,target,unit) = my_input
	best ={'hat_mid':None,'hat_width':None,'r':np.nan}
	fig, ax = plt.subplots(nrows=1,ncols=1)
	ax.set_xlabel(unit)
	ax.set_ylabel('Boolean presence of '+name)
	ax.set_yticks([0,1])
	ax.set_yticklabels(['False','True'])
	if len(list(set(values))) > 1:
		guesses = [(values,mid,width,target) for mid in np.linspace(min(target),max(target),num=100,endpoint=True) for width in np.linspace(0,max(target)-min(target),num=100,endpoint=True)]
		results = map(eval,guesses)
		results = list(results)
		results.sort()
		best ={'hat_mid':results[-1][1],'hat_width':results[-1][2],'r':results[-1][3]}
		hat_width = best['hat_width']
		hat_mid = best['hat_mid']
		preds = np.where(hat_mid-hat_width/2<target,1,0) + np.where(target<hat_mid+hat_width/2,1,0)
		preds = np.where(preds == 2, 1, 0)
		#preds =[1 if hat_mid-hat_width/2<x<hat_mid+hat_width/2 else 0 for x in target]
		target,values,preds = zip(*sorted(zip(target,values,preds)))
		ax.plot(target,preds,'-',rasterized=True,markersize=14,alpha=0.6)
	ax.plot(target,values,'.',rasterized=True,markersize=14,alpha=0.6)
	ax.set_title('Boolean presence of '+name+' vs '+unit+'\nBP r= '+format(best['r'],'.3f'))
	plt.savefig('./results/positional_correlations/'+name+'_correlation.png')
	plt.cla()
	plt.clf()
	plt.close()
	return {'AA':name,'r':best['r']}

def calc(threshold,all_data,threads,unit):
	"""Fit a top hat function to all positions of the one-hot encoded sequence"""
	if not(os.path.isdir('./results/positional_correlations/')):
		os.mkdir('./results/positional_correlations/')

	AA_ref = list(all_data['train'].columns)
	AA_ref.remove('id')
	AA_ref.remove('target')
	to_analyze = [(pos,all_data['train'][pos].values,all_data['train']['target'].values,unit) for pos in AA_ref]
	print('Calculating tophat correlation coefficient.')
	logger.info('Calculating tophat correlation coefficient.')
	#multithread for performance
	p = mp.Pool(threads)
	results = p.map(comparison,to_analyze)
	p.close()
	p.join()
	results = {x['AA']:x['r'] for x in results}
	AA_ref_valid = [x for x in AA_ref if not(np.isnan(results[x]))]
	results_minus_nan = {x:results[x] for x in AA_ref_valid}

	logger.info('Before removing columns, the alignment length is: '+str(len(results.keys())))
	#logger.info('Keeping columns from the MSA if abs(r-scores) >= '+str(threshold))
	#valid_pos = [pos for pos in AA_ref_valid if abs(results[pos])>=threshold]
	logger.info('Fraction of columns kept based on abs(r-scores): '+str(threshold))
	valid_pos = [(pos,abs(results[pos])) for pos in AA_ref_valid]
	valid_pos.sort(key=lambda x:x[1])
	valid_pos = valid_pos[-1*math.floor(threshold*len(valid_pos)):]
	threshold = valid_pos[0][1]
	logger.info('Point-biserial threshold value: '+str(threshold))
	valid_pos = [x[0] for x in valid_pos]

	#write out the results
	sorted_rs = reversed(sorted(results_minus_nan, key=lambda dict_key: abs(results_minus_nan[dict_key])))
	gg = open('./results/Sorted_positional_tophat_correlations.tsv','w')
	gg.write('position\tr-value\n')
	for x in sorted_rs:
		gg.write(x+'\t'+str(results[x])+'\n')
	gg.close()
	
	#plot r heatmap
	length = int(AA_ref[-1][1:])
	matrix = np.zeros(shape=(length,21))
	matrix.fill(np.nan)

	AA_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,'-':20}
	for x in AA_ref_valid:
		matrix[int(x[1:])-1][AA_dict[x[0]]] = results[x]
	matrix = np.ma.masked_where(np.isnan(matrix),matrix)

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
	bound = np.array([-1*np.amin(new_matrix),np.amax(new_matrix)]).max()
	p = ax.pcolormesh(new_matrix,cmap=cm.coolwarm,vmin=-1*bound,vmax=bound)
	cbar = plt.colorbar(p)
	cbar.ax.tick_params(labelsize = font_size)
	plt.tight_layout()
	plt.savefig('./results/tophat_heatmap.png')
	plt.clf()
	plt.close()
	
	positions_values = np.array([results[x] for x in AA_ref_valid])
	stdev =np.std(positions_values)
	mean = np.mean(positions_values)
	logger.info('The stdev of the r-scores is: '+str(stdev))
	logger.info('The mean of the r-scores is: '+str(mean))
	logger.info('The mean of the absolute value of the r-scores is: '+str(np.mean(np.absolute(positions_values))))
	positions = np.array([int(x[1:]) for x in AA_ref_valid])

	#plot r by pos
	fig, ax = plt.subplots(nrows=1,ncols=1)
	ax.set_xlabel('AA positions')
	ax.set_ylabel('Tophat r-value')
	#add horizontal lines a bounds of threshold
	if threshold > 0:
		whole_range = range(positions.min()-10,positions.max()+10,1)
		ax.plot(whole_range,[threshold]*len(whole_range),'-',rasterized=True,linewidth=2,color='#566573')
		ax.plot(whole_range,[-1*threshold]*len(whole_range),'-',rasterized=True,linewidth=2,color='#566573')		
	ax.plot(positions,positions_values,'.',rasterized=True,markersize=14,alpha=0.6)
	ax.set_title('r-value vs position')
	plt.savefig('./results/tophat_correlation_vs_position.png')
	plt.cla()
	plt.clf()
	plt.close()

	#plot histogram of rs	
	if threshold > 0:
		plt.axvline(x=threshold,color='#566573')
		plt.axvline(x=-1*threshold,color='#566573')
	n,bins,patches=plt.hist(positions_values,50,density=False,label='r')
	x_lim =max([abs(x) for x in positions_values])
	plt.xlim(-1.1*x_lim,1.1*x_lim)
	plt.xlabel('r')
	plt.ylabel('Number of positions')
	plt.savefig('./results/tophat_correlation_histogram.png')
	plt.cla()
	plt.clf()
	plt.close()
	

	positions_values ={pos:results[pos] for pos in AA_ref_valid}
	positions_values_z ={pos:(results[pos]-mean)/stdev for pos in AA_ref_valid}

	#write out the NN AA template
	f = open('./results/NN_AA_template.txt','w+')
	for pos in valid_pos:
		f.write(pos+'\n')
	f.close()
	
	to_remove =[pos for pos in AA_ref if not(pos in valid_pos)]

	#remove insignificant columns
	for file in all_data.keys():
		all_data[file]=all_data[file].drop(labels=to_remove,axis=1)
		#all_data[file].to_csv('./results/'+file+'_alignment_after_biserial_thresholding.csv')

	logger.info('After removing columns based on correlaton coefficient to tophat function, the alignment length is: '+str(len(valid_pos)))

	return all_data

