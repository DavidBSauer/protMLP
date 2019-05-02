#!/usr/bin/env python3
import logging
logger = logging.getLogger('species_assignment')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('Step1.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import Bio
logger.info('Biopython version: '+Bio.__version__)
import sys
logger.info('Python version: '+sys.version)
import platform
logger.info('Platform: '+platform.platform())
del(platform)
import cleanup
from Bio.Align import MultipleSeqAlignment
import random
import argparse

parser = argparse.ArgumentParser(description='Step 1. Take in an MSA file and assign the species from Uniprot (locally or via web). Will remove fragments and gap incuding sequences and generate train, test, and optionally validation MSAs')
files = parser.add_argument_group('Required files')

threshold = 0.995
validation = False

files = parser.add_argument_group('Required file')
files.add_argument("-f","--fasta",action='append', type=str, help="The MSA file in FASTA format. Can be invoked multiple times for multiple files.",dest='input_fasta',default=[])
files.add_argument("-s","--stock",action='append', type=str, help="The MSA file in Stockholm format. Can be invoked multiple times for multiple files.",dest='input_stock',default=[])
parser.add_argument("-t", "--threshold",action='store', type=float, help="Removing sequences which cause gaps at a frequency greater than provided frequency. Default is "+str(threshold)+'.',dest='threshold',default=threshold)
getters = parser.add_mutually_exclusive_group()
getters.add_argument("-w", "--web",action='store_true', help="Access the uniprot website for Uniprot information on each sequence. This is the default behavior.",dest='getters_web',default=False)
getters.add_argument("-lx", "--local_xml",action='append', type=str, help="Provide a local XML Uniprot database file. Will use this file for retrieving information on each sequence. Can be invoked repeatedly for multiple files.",dest='getters_local_xml',default=None)
getters.add_argument("-ld", "--local_dat",action='append', type=str, help="Provide a local DAT Uniprot database file. Will use this file for retrieving information on each sequence. Can be invoked repeatedly for multiple files.",dest='getters_local_dat',default=None)

args = parser.parse_args()

MSA_files = args.input_fasta+args.input_stock
if len(MSA_files)==0:
	logger.info('No MSA files provided. Quitting.')
	sys.exit()

MSA_files= {}
for MSA_file in args.input_fasta:
	logger.info('Loading the FASTA MSA file: '+MSA_file)
	try:
		MSA_files[MSA_file] = Bio.AlignIO.read(MSA_file,'fasta')
	except:
		logger.info('Problem reading the file: '+MSA_file+'. Skipping.')
	else:	
		pass

for MSA_file in args.input_stock:
	logger.info('Loading the Stockholm MSA file: '+MSA_file)
	try:
		MSA_files[MSA_file] = Bio.AlignIO.read(MSA_file,'stockholm')
	except:
		logger.info('Problem reading the file: '+MSA_file+'. Skipping.')
	else:	
		pass

if not(args.getters_local_xml == None):
	logger.info('Using a local xml copy of the Uniprot XML databases.')
	logger.info('Uniprot database files: '+' '.join(args.getters_local_xml))
	import getter_local as getter
	getter.setup_xml(args.getters_local_xml)
elif not(args.getters_local_dat == None):
	logger.info('Using a local copy of the Uniprot DAT databases.')
	logger.info('Uniprot database files: '+' '.join(args.getters_local_dat))
	import getter_local as getter
	getter.setup_dat(args.getters_local_dat)
else:
	logger.info('Using the web to access Uniprot.')
	import getter

def spec(txt):
	a = txt.split('|')
	b = a[1]
	return b.lower()

for MSA_file in MSA_files.keys():
	#read in the input MSAs
	logger.info('Working on the MSA file: '+MSA_file)
	MSA = MSA_files[MSA_file]
	logger.info('Initial sequences: '+str(len(MSA)))
	logger.info('Initial alignment length: '+str(MSA.get_alignment_length()))
	MSA_Xed = cleanup.removeXs(MSA)
	Bio.AlignIO.write(MSA_Xed,MSA_file.split('.')[0]+"_Xed.fa", "fasta")
	del MSA
	logger.info('Sequences without X\'s or other non-cannonical AAs: '+str(len(MSA_Xed)))
	logger.info('Creating non-redundant names')
	MSA_nonredundant = cleanup.rendundant_names(MSA_Xed)
	Bio.AlignIO.write(MSA_nonredundant,MSA_file.split('.')[0]+"_nonredundant.fa", "fasta")
	del MSA_Xed

	#retain only those sequences with species, non-fragment
	logger.info('Finding non-fragement sequences with species')	
	MSA_assigned = getter.non_frag_spec(MSA_nonredundant)
	del MSA_nonredundant
	logger.info('Non-fragment sequences, with species information: '+str(len(MSA_assigned)))
	species_list = list(set([x.id.split('|')[-1] for x in MSA_assigned]))
	f = open(MSA_file.split('.')[0]+'_assigned_species.txt','w')
	f.write('\n'.join(species_list))
	f.close()

	#remove gap inducing sequences which are deleterious to data-parameter ratio
	logger.info('Removing sequences which cause gaps at a frequency greater than '+str(threshold))
	MSA_degapped = cleanup.gaps(MSA_assigned,threshold)
	Bio.AlignIO.write(MSA_degapped,MSA_file.split('.')[0]+"_degapped.fa", "fasta")
	logger.info('Sequences after degapping: '+str(len(MSA_degapped)))

	#generate training/testing/validation datasets\
	logger.info('Generating training/testing/validation datasets')
	MSA_training =[]
	MSA_testing =[]
	MSA_validation =[]
	for x in MSA_degapped:
		rand = random.random()
		if rand<=0.7:
			MSA_training.append(x)
		elif rand<=0.8:
			MSA_validation.append(x)
		else:
			#assign 20% to testing set
			MSA_testing.append(x)	

	MSA_training=MultipleSeqAlignment(MSA_training)
	Bio.AlignIO.write(MSA_training, MSA_file.split('.')[0]+"_training.fa", "fasta")
	MSA_testing=MultipleSeqAlignment(MSA_testing)
	Bio.AlignIO.write(MSA_testing, MSA_file.split('.')[0]+"_testing.fa", "fasta")
	MSA_validation=MultipleSeqAlignment(MSA_validation)
	Bio.AlignIO.write(MSA_validation, MSA_file.split('.')[0]+"_validation.fa", "fasta")	

	logger.info('Sequences in training dataset: '+str(len(MSA_training)))
	logger.info('Sequences in testing dataset: '+str(len(MSA_testing)))
	logger.info('Sequences in validation dataset: '+str(len(MSA_validation)))


logger.info('finished successfully')

