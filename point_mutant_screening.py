import logging
logger = logging.getLogger('mutants')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('Mutant_screening.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import sys
import argparse
from Bio import SeqIO
import os
import converter
import regressors
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import AlignIO
import multiprocessing as mp
import pandas as pd
import random

seed = random.randint(0,10**9)
AA_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,'-':20}
AAs = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']
mut_num =1

parallel = 1
batch = float('inf')
parser = argparse.ArgumentParser(description='Given a trained model and input template, predict the optimal growth temperature of mutants of a provided protein sequence.')
parser.add_argument("-sq","--seq",action='store', type=str, help="The FASTA file of a sequence from which mutants will be generated. Must be aligned to the training data.",dest='file',default=None)
parser.add_argument("-t","--template",action='store', type=str, help="The template file of MLP inputs. (From step3 this will be the file AA_template.txt.)",dest='template',default=None)
parser.add_argument("-m","--model",action='store', type=str, help="The regression model (model.h5) file to use.",dest='model',default=None)
parser.add_argument("-p", "--parallel",help="Run parallel where-ever possible. Avoid using if there are errors. Default is "+str(parallel),action="store",dest='parallel',default=parallel)
parser.add_argument("-n","--mut_num",action='store',type=int,help='The maximum number of mutants to generate within the pepite sequence. Note: this will grow exponentially! Default is '+str(mut_num),dest='mut_num',default=mut_num)
parser.add_argument("-b","--batch",action='store',type=int,help='Calculate in batches. This is helpful if the number of mutations in high, which can lead to high memory use and possibly the program crashing. Optimal batch size will depend upon protein length, but 4 to 5 are typically reasonable.',dest='batch',default=batch)
parser.add_argument("-s", "--seed",type=int,help="Seed for the regression. Randomly generated value is "+str(seed),dest='seed',default=seed)


args = parser.parse_args()
parallel = int(args.parallel)
file = args.file
template = args.template
model = args.model
mut_num = args.mut_num
batch = 10**args.batch
seed = args.seed

logger.info('Target sequence file: '+str(file))
logger.info('Template file: '+str(template))
logger.info('Model file: '+str(model))
logger.info('Number of threads to run in parallel: '+str(parallel))
logger.info('Maximum number of mutants: '+str(mut_num))
logger.info('Batch size: '+str(batch))
logger.info('Seed is: '+str(seed))

if ((file == None) or (template == None) or (model == None)):
	logger.info('Need a trained model, template, and sequences to predict; which were not provided. Quitting')
	print('Need a trained model, template, and sequences to predict; which were not provided. Quitting')
	sys.exit()

#load the MSA file	
try:
	logger.info('Reading the target file: '+file)
	input_seq = SeqIO.read(file,'fasta')
except:
	logger.info('Problem reading the target file. Quitting.')
	print('Problem reading the target file. Quitting.')
	sys.exit()
else:
	pass


#read in the template
if not(os.path.isfile(template)):
	logger.info('Problem reading the template file. Quitting.')
	print('Problem reading the template file. Quitting.')
	sys.exit()
else:
	f = open(template,'r')
	template = [line.strip() for line in f.readlines()]
	f.close()

#check that the regression model is there, will load later 
if not(os.path.isfile(model)):
	logger.info('Problem finding the model directory. Quitting.')
	print('Problem finding the model directory. Quitting.')
	sys.exit()

#create a dictionary of all possible mutants (AAs which are in the training set)
valid_mutants = {}
for poss_AA in template:
	if int(poss_AA[1:])-1 in valid_mutants.keys():
		valid_mutants[int(poss_AA[1:])-1].append(poss_AA[0])
	else:
		valid_mutants[int(poss_AA[1:])-1]=[poss_AA[0]]

def mut_generator(input):
	(name,sequence,template) = input
	curr_seq = list(str(sequence))
	poss_seqs = []
	for x in valid_mutants.keys():
		for poss_AA in valid_mutants[x]:
			if not(poss_AA == curr_seq[x]): #ensure this AA isn't already in the sequence
				if not(str(x+1) in [z[1:-1] for z in name]): #ensure this position hasn't already been mutated
					working_seq = [z for z in curr_seq]
					working_seq[x]=poss_AA
					poss_seqs.append((name+[curr_seq[x]+str(x+1)+poss_AA],''.join(working_seq)))
	return poss_seqs

#generate mutant proteins
logger.info('Generating mutants, round 1')
mutants = mut_generator(([],input_seq.seq,template))

#generate mutants on previously generated mutants
for num in range(1,mut_num,1):
	logger.info('Generating mutants, round '+str(num+1))
	to_analyze = [(mutant[0],mutant[1],template) for mutant in mutants]
	p = mp.Pool(parallel)
	new_mutants = p.map(mut_generator,to_analyze,1)
	p.close()
	p.join()			
	new_mutants = list(new_mutants)
	flattened = [val for sublist in new_mutants for val in sublist]
	mutants = mutants + flattened		

logger.info('The number of mutant sequences generated: '+str(len(mutants)))

mutant_seqs = [SeqRecord(Seq(''.join(mutant[1])), '/'.join(mutant[0]),'','') for mutant in mutants]
mutant_seqs = mutant_seqs+[SeqRecord(input_seq.seq, 'WT','','')]
MSA= MultipleSeqAlignment(mutant_seqs)
#AlignIO.write(MSA,'mutant_sequences.fa','fasta')

pred_results = {}
if len(MSA) > batch:
	logger.info('Running in batch mode')
	
	#generate batches to analyze
	batches = []
	working_batch = []
	for sequence in MSA:
		if len(working_batch)==batch:
			batches.append(MultipleSeqAlignment(working_batch))
			working_batch =[]
		working_batch.append(sequence)
	batches.append(MultipleSeqAlignment(working_batch))
	
	#analyze in batches
	for z in range(0,len(batches),1):
		batch = batches[z]
		logger.info('Calculating batch: '+str(z+1))

		#covert the alignment to a PD dataframe and remove un-needed columns
		data=converter.convert_on_template_no_target(batch,template,parallel,seed)

		#predict using previously trained model
		results = regressors.infer_MLP(model,data)
		for index,row in results.iterrows():
			pred_results[row['id']]=row['prediction']
	
else:
	#covert the alignment to a PD dataframe and remove un-needed columns
	data=converter.convert_on_template_no_target(MSA,template,parallel,seed)

	#predict using previously trained model
	results = regressors.infer(model,data)
	pred_results = {row['id']:row['prediction'] for index,row in results.iterrows()}

#write out MSA of mutant sequences with predicted values
mutant_seqs ={x.id:x for x in mutant_seqs}
mutant_seqs_values =[SeqRecord(mutant_seqs[x].seq, x+'|'+str(pred_results[x]),'','') for x in pred_results]
MSA= MultipleSeqAlignment(mutant_seqs_values)
AlignIO.write(MSA,'mutant_sequences_predictions.fa','fasta')
