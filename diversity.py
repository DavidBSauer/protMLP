import multiprocessing as mp
import matplotlib.pyplot as plt
import math

def info_calc(input):
	"""calculate information content of each position in a MSA"""
	(n,values,chars) = input
	H = -1*sum([(values.count(x)/len(values))*math.log2(values.count(x)/len(values)) if not(values.count(x) == 0) else 0 for x in chars])
	en = ((len(chars)-1)/(2*len(values)))*(1/math.log(2))
	R = math.log2(len(chars))-(H+en)
	return (n,R)

def information(MSA,nucleotide,threads):
	"""calculate the information content of a MSA. Based on Schneider et al. 1985"""
	chars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
	if nucleotide:
		chars = ['A','C','G','T']
	jobs = [(n,[x.seq[n] for x in MSA],chars) for n in range(0,MSA.get_alignment_length(),1)]
	p = mp.Pool(threads)
	results = p.map(info_calc,jobs)
	p.close()
	p.join()
	results = list(results)
	results.sort()
	sum_information = sum([x[1] for x in results])
	plt.plot([x[0] for x in results],[x[1] for x in results])
	plt.xlabel('Position')
	plt.ylabel('Rseq(L) (bits)')
	plt.ylim(-0.05,4.4)
	if nucleotide:
		plt.ylim(-0.05,2.4)
	plt.title('Sequence information.\nRseq: '+str(sum_information))
	plt.tight_layout()
	plt.savefig('./results/Sequence_information.png')
	plt.clf()
	plt.close()
	return sum_information	

def divaa_calc(input):
	"""calculate the divaa of a position in a MSA. Based on Rodi et al. 2004"""
	(n,values,chars) = input
	d = 1/(len(chars)*sum([(values.count(x)/len(values))**2 for x in chars]))
	return (n,d)
	
def divaa(MSA,nucleotide,threads):
	"""calculate the mean divaa of a MSA. Based on Rodi et al. 2004"""
	chars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']
	if nucleotide:
		chars = ['A','C','G','T','-']
	jobs = [(n,[x.seq[n] for x in MSA],chars) for n in range(0,MSA.get_alignment_length(),1)]
	p = mp.Pool(threads)
	results = p.map(divaa_calc,jobs)
	p.close()
	p.join()
	results = list(results)
	results.sort()
	mean_diversity = sum([x[1] for x in results])/len(results)
	plt.plot([x[0] for x in results],[x[1] for x in results])
	plt.xlabel('Position')
	plt.ylabel('DIVAA Diversity')
	plt.ylim(-0.05,1.05)
	plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
	plt.title('DIVAA diversity.\nMean diversity: '+str(mean_diversity))
	plt.tight_layout()
	plt.savefig('./results/DIVAA_diversity.png')
	plt.clf()
	plt.close()
	return mean_diversity	
