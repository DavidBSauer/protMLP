import logging
logger = logging.getLogger('prediction')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('Prediction.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import sys
import argparse
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
import converter
import regressors
import random

AA_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,'-':20}
AAs = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']

parallel = 1
pseudo = False
seed = random.randint(0,10**9)

parser = argparse.ArgumentParser(description='Given a trained model and input template, predict the optimal growth temperature of a FASTA file of protein sequences.')
parser.add_argument("-sq","--seq",action='store', type=str, help="The MSA file of sequences to predict in FASTA format.",dest='msa',default=None)
parser.add_argument("-t","--template",action='store', type=str, help="The template file of MLP inputs. (From step3 this will be the file NN_AA_template.txt.)",dest='template',default=None)
parser.add_argument("-m","--model",action='store', type=str, help="The regression model (model.h5) file to use.",dest='model',default=None)
parser.add_argument("-p", "--parallel",help="Number of threads to run in parallel. Avoid using if there are errors. Default is "+str(parallel),action="store",dest='parallel',default=parallel)
parser.add_argument("-pl", "--pseudolabel",help="Pseudolabel the provide MSA. Default is "+str(pseudo),action="store_true",dest='pseudo',default=pseudo)
parser.add_argument("-knn_k", "--knn_neighbors",help="Number of neighbors to use when calculating a k-Nearest-Neighbor prediction. Using kNN prediction will also require an alignment.",action="store",type=int,dest='knnk',default=None)
parser.add_argument("-rfr_t", "--rfr_trees",help="Number of trees to use when calculating a Random Forest prediction. Using RF prediction will also require an alignment.",action="store",type=int,dest='rfrt',default=None)
parser.add_argument("-a", "--alignment",help="Training alignment for calculating k-Nearest-Neighbor and Random Forest Regression in FASTA format.",action="store",type=str,dest='aln',default=None)
parser.add_argument("-s", "--seed",type=int,help="Seed for the regression. Randomly generated value is "+str(seed),dest='seed',default=seed)

args = parser.parse_args()
parallel = int(args.parallel)
MSA_name = args.msa
template = args.template
model = args.model
pseudo = args.pseudo
aln = args.aln
knnk = args.knnk
rfrt = args.rfrt
seed = args.seed

logger.info('MSA file: '+str(MSA_name))
logger.info('Template file: '+str(template))
logger.info('Model file: '+str(model))
logger.info('Number of threads to run in parallel: '+str(parallel))
logger.info('Pseudolabel the MSA: '+str(pseudo))
logger.info('kNN neighbors: '+str(knnk))
logger.info('RF trees: '+str(rfrt))
logger.info('Training alignment: '+str(aln))
logger.info('Seed is: '+str(seed))

if ((MSA_name == None) or (template == None)):
	logger.info('Need a template and sequences to predict; which were not provided. Quitting')
	print('Need a trained model, template, and sequences to predict; which were not provided. Quitting')
	sys.exit()

#load the MSA file	
try:
	logger.info('Reading the target file: '+MSA_name)
	MSA = AlignIO.read(MSA_name,'fasta')
except:
	logger.info('Problem reading the target MSA. Quitting.')
	print('Problem reading the target MSA. Quitting.')
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

#covert the alignment to a PD dataframe and remove un-needed columns
data=converter.convert_on_template_no_target(MSA,template,parallel,seed)

regressors.setup_params(parallel,seed)

if model == None:
	logger.info('No MLP model provided. Will not run MLP prediction.')
else:
	#check that the regression model is there, will load later 
	if os.path.isfile(model):
		#predict using previously trained model
		results = regressors.infer_MLP(model,data)
		results.to_csv('model_prediction_results.tsv',sep='\t',columns=['id','prediction'],index=False)
		
		#pseduolabel the MSA
		if pseudo:
			predictions = {row['id']:row['prediction'] for index, row in results.iterrows()}
			pred_MSA = [] 
			for x in MSA:
				pred_MSA.append(SeqRecord(Seq(str(x.seq),x.seq.alphabet), x.id+'|'+str(predictions[x.id]),'',''))
			pred_MSA = MultipleSeqAlignment(pred_MSA)
			AlignIO.write(pred_MSA,MSA_name.split('.')[0]+"_model_pseudolabel.fa", "fasta")
	else:	
		logger.info('Problem finding the model directory.')

if ((aln != None) and (knnk != None)):
	try:
		logger.info('Will try kNN prediction.')
		MSA = AlignIO.read(aln,'fasta')
		MSA = converter.convert_on_template(MSA,template,parallel,seed)
	except:
		logger.info('Problem reading the training alignment file. Quitting.')
		print('Problem reading the training alignment file. Quitting.')
		sys.exit()
	else:
	#run the knn prediction
		#predict using previously trained model
		results = regressors.infer_knn(knnk,MSA,data,parallel)
		results.to_csv('kNN_prediction_results.tsv',sep='\t',columns=['id','prediction'],index=False)
		
		#pseduolabel the MSA
		if pseudo:
			predictions = {row['id']:row['prediction'] for index, row in results.iterrows()}
			pred_MSA = [] 
			for x in MSA:
				pred_MSA.append(SeqRecord(Seq(str(x.seq),x.seq.alphabet), x.id+'|'+str(predictions[x.id]),'',''))
			pred_MSA = MultipleSeqAlignment(pred_MSA)
			AlignIO.write(pred_MSA,MSA_name.split('.')[0]+"_kNN_pseudolabel.fa", "fasta")
else:
	logger.info('Missing either the training alignment or number of neighbors (k) necessary for kNN. Will not run kNN prediction.')


if ((aln != None) and (rfrt != None)):
	try:
		logger.info('Will try Random Forest prediction.')
		MSA = AlignIO.read(aln,'fasta')
		MSA = converter.convert_on_template(MSA,template,parallel,seed)
	except:
		logger.info('Problem reading the training alignment file. Quitting.')
		print('Problem reading the training alignment file. Quitting.')
		sys.exit()
	else:
	#run the knn prediction
		#predict using previously trained model
		results = regressors.infer_RFR(rfrt,MSA,data,parallel)
		results.to_csv('RF_prediction_results.tsv',sep='\t',columns=['id','prediction'],index=False)
		
		#pseduolabel the MSA
		if pseudo:
			predictions = {row['id']:row['prediction'] for index, row in results.iterrows()}
			pred_MSA = [] 
			for x in MSA:
				pred_MSA.append(SeqRecord(Seq(str(x.seq),x.seq.alphabet), x.id+'|'+str(predictions[x.id]),'',''))
			pred_MSA = MultipleSeqAlignment(pred_MSA)
			AlignIO.write(pred_MSA,MSA_name.split('.')[0]+"_RFR_pseudolabel.fa", "fasta")
else:
	logger.info('Missing either the training alignment or number of trees necessary for Random Forest regression. Will not run Random Forest prediction.')
