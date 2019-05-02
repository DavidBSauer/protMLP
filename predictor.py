#!/usr/bin/env python3
import logging
logger = logging.getLogger('prediction')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('Prediction.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import sys
import argparse
from Bio import AlignIO
import os
import converter
import tfMLP as MLP

AA_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,'-':20}
AAs = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']

parallel = False
parser = argparse.ArgumentParser(description='Step 4. Given a trained model and input template, predict the optimal growth temperature of a FASTA file of protein sequences.')
parser.add_argument("-s","--seq",action='store', type=str, help="The MSA file of sequences to predict in FASTA format.",dest='msa',default=None)
parser.add_argument("-t","--template",action='store', type=str, help="The template file of MLP inputs. (From step3 this will be the file NN_AA_template.txt.)",dest='template',default=None)
parser.add_argument("-m","--model",action='store', type=str, help="The regression model (model.h5) file to use.",dest='model',default=None)
parser.add_argument("-p", "--parallel",help="Run parallel where-ever possible. Avoid using if there are errors. Default is "+str(parallel),action="store_true",dest='parallel',default=parallel)

args = parser.parse_args()
parallel = args.parallel
MSA = args.msa
template = args.template
model = args.model

logger.info('MSA file: '+str(MSA))
logger.info('Template file: '+str(template))
logger.info('Model file: '+str(model))
logger.info('Running parallel: '+str(parallel))

if ((MSA == None) or (template == None) or (model == None)):
	logger.info('Need a trained model, template, and sequences to predict; which were not provided. Quitting')
	print('Need a trained model, template, and sequences to predict; which were not provided. Quitting')
	sys.exit()

#load the MSA file	
try:
	logger.info('Reading the target file: '+args.msa)
	MSA = AlignIO.read(MSA,'fasta')
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

#check that the regression model is there, will load later 
if not(os.path.isfile(model)):
	logger.info('Problem finding the model directory. Quitting.')
	print('Problem finding the model directory. Quitting.')
	sys.exit()

#covert the alignment to a PD dataframe and remove un-needed columns
data=converter.convert_on_template(MSA,template,parallel)

#predict using previously trained model
results = MLP.infer(model,data)
results.to_csv('prediction_results.csv',sep='\t',columns=['id','prediction'],index=False)
