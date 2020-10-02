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
import subprocess
import os

threshold = 0.995
cluster = 'None'
nucleotide = False
parallel = 1
cdhit = None
verbose = False

parser = argparse.ArgumentParser(description='Step 1. Take in an MSA file and assign the species from Uniprot (locally or via web). Will remove fragments and gap incuding sequences and generate train, test, and validation MSAs')
files = parser.add_argument_group('Required file')
files.add_argument("-f","--fasta",action='append', type=str, help="The MSA file in FASTA format. Can be invoked multiple times for multiple files.",dest='input_fasta',default=[])
files.add_argument("-s","--stock",action='append', type=str, help="The MSA file in Stockholm format. Can be invoked multiple times for multiple files.",dest='input_stock',default=[])
parser.add_argument("-t", "--threshold",action='store', type=float, help="Removing sequences which cause gaps at a frequency greater than provided frequency. Default is "+str(threshold)+'.',dest='threshold',default=threshold)
parser.add_argument("-c", "--cluster",action='store', type=str, help="Cluster sequences to this fraction identity using CD-HIT (range 0.0-1.0)",dest='cluster',default=cluster)
parser.add_argument("-cdhit",action='store', type=str, help="CD-Hit executable",dest='cdhit',default=cdhit)
parser.add_argument("-w", "--web",action='store_true', help="Access the uniprot website for Uniprot information on each sequence. This is the default behavior.",dest='getters_web',default=False)
parser.add_argument("-lx", "--local_xml",action='append', type=str, help="Provide a local XML Uniprot database file. Will use this file for retrieving information on each sequence. Can be invoked repeatedly for multiple files.",dest='getters_local_xml',default=[])
parser.add_argument("-ld", "--local_dat",action='append', type=str, help="Provide a local DAT Uniprot database file. Will use this file for retrieving information on each sequence. Can be invoked repeatedly for multiple files.",dest='getters_local_dat',default=[])
parser.add_argument("-nuc", "--nucleotide",help="The provided sequences are nucleotide sequences. Does not change regression behavior, but is used for diversity calculation and making neater plots. Default is "+str(nucleotide),action="store_true",dest='nuc',default=nucleotide)
parser.add_argument("-p", "--parallel",help="Number of threads to run in parallel. Default is "+str(parallel),action="store",dest='parallel',default=parallel)
parser.add_argument("-v", "--verbose",help="Verbose: show progress printed in console. Default is "+str(verbose),action="store_true",dest='verbose',default=verbose)

args = parser.parse_args()

cluster = args.cluster
nucleotide = args.nuc
threshold = args.threshold
parallel = int(args.parallel)
cdhit = args.cdhit
verbose = args.verbose

logger.info('Sequences are nucleotide sequences: '+str(nucleotide))
logger.info('Clustering tsv file: '+str(cluster))
logger.info('Degapping threshold: '+str(threshold))
logger.info('Number of threads to run in parallel: '+str(parallel))
logger.info('CD-Hit executable: '+str(cdhit))
logger.info('Verbose: '+str(verbose))

MSA_files = args.input_fasta+args.input_stock
if len(MSA_files)==0:
	logger.info('No MSA files provided. Quitting.')
	sys.exit()

if not(cluster == 'None'):
	if cdhit == None:
		logger.info('No CD-Hit executable provided. Required for clustering.')
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

if len(MSA_file)==0:
	logger.info('No MSA files could be loaded. Quitting.')
	sys.exit()

species_assign = False
if len(args.getters_local_xml) > 0:
	logger.info('Using a local xml copy of the Uniprot XML databases.')
	logger.info('Uniprot database files: '+' '.join(args.getters_local_xml))
	import getter_local as getter
	getter.setup_xml(args.getters_local_xml)
	species_assign = True
elif len(args.getters_local_dat) > 0:
	logger.info('Using a local copy of the Uniprot DAT databases.')
	logger.info('Uniprot database files: '+' '.join(args.getters_local_dat))
	import getter_local as getter
	getter.setup_dat(args.getters_local_dat)
	species_assign = True
elif args.getters_web:
	logger.info('Using the web to access Uniprot.')
	import getter
	species_assign = True

for MSA_file in MSA_files.keys():
	#read in the input MSAs
	logger.info('Working on the MSA file: '+MSA_file)
	MSA = MSA_files[MSA_file]
	logger.info('Initial sequences: '+str(len(MSA)))
	logger.info('Initial alignment length: '+str(MSA.get_alignment_length()))
	MSA_Xed = cleanup.removeXs(MSA,nucleotide)
	Bio.AlignIO.write(MSA_Xed,MSA_file.split('.')[0]+"_Xed.fa", "fasta")
	del MSA
	
	logger.info('Sequences without X\'s or other non-cannonical AAs: '+str(len(MSA_Xed)))
	logger.info('Creating non-redundant names')
	MSA_nonredundant = cleanup.rendundant_names(MSA_Xed)
	Bio.AlignIO.write(MSA_nonredundant,MSA_file.split('.')[0]+"_nonredundant.fa", "fasta")
	del MSA_Xed

	if species_assign:
		#retain only those sequences with species, non-fragment
		logger.info('Finding sequences with species')	
		MSA_species = getter.spec(MSA_nonredundant,verbose)
		logger.info('Sequences with species assigned: '+str(len(MSA_species)))	
		Bio.AlignIO.write(MSA_species,MSA_file.split('.')[0]+"_assigned_species.fa", "fasta")
		del MSA_nonredundant
		logger.info('Removing sequence fragments')	
		MSA_nonfrag = getter.non_frag(MSA_species,verbose)
		logger.info('Non-fragment sequences: '+str(len(MSA_nonfrag)))
		Bio.AlignIO.write(MSA_nonfrag,MSA_file.split('.')[0]+"_nonfragment.fa", "fasta")
		del MSA_species
	else:
		#skip species assignment and fragment removal
		MSA_nonfrag = MSA_nonredundant

	#remove gap inducing sequences which are deleterious to data-parameter ratio
	logger.info('Removing sequences which cause gaps at a frequency greater than '+str(threshold))
	MSA_degapped = cleanup.gaps(MSA_nonfrag,threshold)
	Bio.AlignIO.write(MSA_degapped,MSA_file.split('.')[0]+"_degapped.fa", "fasta")
	logger.info('Sequences after degapping: '+str(len(MSA_degapped)))

	#group sequences based on sequence identity
	seq_dict = {}
	if cluster == 'None':
		for entry in MSA_degapped:
			if str(entry.seq) in seq_dict.keys():
				seq_dict[str(entry.seq)].append(entry)
			else:
				seq_dict[str(entry.seq)]=[entry]
	
	else:
	#run cd-hit to cluster sequences
		seqs = cleanup.remove_all_gaps(MSA_degapped)
		Bio.SeqIO.write(seqs,MSA_file.split('.')[0]+"_nogaps.fa", "fasta")
		p = subprocess.Popen([cdhit+' -i '+MSA_file.split('.')[0]+'_nogaps.fa'+' -o cdhit_clust -c '+str(cluster)+' -T '+str(parallel)],shell=True,executable='/bin/bash',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		out,err=p.communicate()
		out = out.decode('utf-8').strip()
		err = err.decode('utf-8').strip()
		if not(err == ''):
			logger.info('Quitting due to a problem with CD-HIT.\n CD-HIT an error message: '+err)
			sys.exit()
		if not(os.path.isfile('cdhit_clust.clstr')):
			logger.info('No CD-HIT file produced. CD-HIT command line output: '+out)
			logger.info('Quitting')
			sys.exit()
		MSA_degapped = {x.id.split('|')[0]:x for x in MSA_degapped}
		f = open('cdhit_clust.clstr','r') 
		cluster = f.read()
		f.close()
		seq_dict = {}
		clusters = cluster.split('Cluster')[1:]
		for z in range(0,len(clusters),1):
			cluster = clusters[z]
			entries = cluster.split(',')[1:]
			entries = [x.split('>')[1].split('...')[0] for x in entries]
			entries = [MSA_degapped[x] for x in entries]
			seq_dict[z]=entries

	#generate training/testing/validation datasets
	logger.info('Generating training/testing/validation datasets after clustering sequences')
	MSA_training =[]
	MSA_testing =[]
	MSA_validation =[]
	entries= list(seq_dict.keys())
	logger.info('Number of sequence clusters: '+str(len(entries)))
	random.shuffle(entries)
	for x in entries:
		rand = random.random()
		if rand<=0.7:
			#assign 70% to training set
			MSA_training=MSA_training+seq_dict[x]
		elif rand<=0.8:
			#assign 10% to validation set
			MSA_validation=MSA_validation+seq_dict[x]
		else:
			#assign 20% to testing set
			MSA_testing=MSA_testing+seq_dict[x]

	#write out MSA files
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

