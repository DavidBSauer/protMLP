#take in an MSA and assign traits, keeping only those in the desired range

import logging
logger = logging.getLogger('ranging_sequences')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('Step2.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import sys
logger.info('Python version: '+sys.version)
import platform
logger.info('Platform: '+platform.platform())
del(platform)
import Bio
logger.info('Biopython version: '+Bio.__version__)
del Bio
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import csv
import argparse
import identity

distribution = False
def_range = 'all'
verbose = False

parser = argparse.ArgumentParser(description="Step 2. Take in a species-trait file and MSA files. Assign trait's to all sequences based on assigned species, then remove sequences outside of provided trait range.")
files = parser.add_argument_group('Required files')
files.add_argument("-t",action='store', type=str, help="The species-trait file.",dest='trait',default=None)
files.add_argument("-sq","--seq",action='append', type=str, help="The MSA file in FASTA format.",dest='MSA_file',default=None)
parser.add_argument("-r", "--range",action='store', type=str, help="The range of traits's to keep. Can be 'all', some combination of p/m/t for psychrophiles, mesophile, or thermophiles (probably only relevant when considering Tg's). Or a given range, with 'to' denoting ranges and ',' for multiple ranges. Examples: 'mt' or '-25to35,45to65'. Default is "+str(def_range)+'.',dest='range',default=def_range)
parser.add_argument("-id", "--distribution",help="Calculate pairwise sequence identity distribution for all sequences. Default is "+str(distribution),action="store_true",dest='dist',default=distribution)
parser.add_argument("-v", "--verbose",help="Verbose. Show progress bars while training MLPs. Default is "+str(verbose),action="store_true",dest='verbose',default=verbose)

args = parser.parse_args()
data_range = args.range
distribution = args.dist
verbose = args.verbose

logger.info('Data range: '+str(data_range))
logger.info('Calculate MSA sequence identity distribution: '+str(distribution))
logger.info('Verbose mode: '+str(verbose))

try:
	#read in species-OGT file
	logger.info("Species-trait file: "+args.trait)
	infile = open(args.trait,'r')
	reader = csv.reader(infile,delimiter='\t')
	species_temp = dict((str(rows[0]),float(rows[1])) for rows in reader)
	infile.close()    
	logger.info("found "+str(len(species_temp.keys()))+" species-trait pairs")
except:
	logger.info("Problem reading the species-trait file. Quitting")
	print("Problem reading the species-trait file. Quitting")
	sys.exit()
else:
	pass

files = {}
for file in args.MSA_file:
	try:
		logger.info('MSA file: '+file)
		files[file]= AlignIO.read(file,'fasta')
	except:
		logger.info('Problem reading the MSA file: '+file+'. Quitting')
		print('Problem reading the MSA file: '+file+'. Quitting')
		sys.exit()
	else:
		pass
	
#find out which OGT ranges are desired
all_ogt = ('all' == data_range)
if all_ogt:
	OGT_range = [(-1*float('inf'),float('inf'))]

if not(all_ogt):
	psych = None
	meso = None
	thermo = None
	OGT_range  =[]
	psych = ('p' in data_range)
	if psych:
		OGT_range.append((-1*float('inf'),float(20)))
		psych = True

	meso = ('m' in data_range)
	if meso:
		OGT_range.append((float(20),float(45)))
		meso = True

	thermo = (('t' in data_range) and not('to' in data_range))
	if thermo:
		OGT_range.append((float(45),float('inf')))
		thermo = True

if not(all_ogt or psych or meso or thermo):
	OGT_range = [(float(subrange.split('to')[0]),float(subrange.split('to')[1])) for subrange in data_range.split(',')]

logger.info('Including sequences within the trait ranges of: '+', '.join([' to '.join([str(y) for y in x]) for x in OGT_range]))

#assign temp to each sequence based on species
def spec(txt):
	a = txt.split('|')
	b = a[-1].lower()
	return b

for file in files.keys():
	logger.info('Working on MSA file: '+file)
	MSA_file = files[file]
	logger.info('Number of sequences in the MSA: '+str(len(MSA_file)))
	assigned = []
	unassigned = []
	logger.info("Assigning traits to sequences for MSA")
	for x in MSA_file:
		if spec(x.id) in species_temp.keys():
			assigned.append(SeqRecord(Seq(str(x.seq)), x.id+'|'+str(species_temp[spec(x.id)]),'',''))
		else:
			unassigned.append(SeqRecord(Seq(str(x.seq)), x.id,'',''))
	logger.info('Number of sequences with assigned traits: '+str(len(assigned)))				
	MSA_file= MultipleSeqAlignment(unassigned)
	AlignIO.write(MSA_file,file.split('.')[0]+"_unassigned.fa", "fasta")

	MSA_file= MultipleSeqAlignment(assigned)
	AlignIO.write(MSA_file,file.split('.')[0]+"_assigned.fa", "fasta")


	def temp(txt):
		a = txt.split('|')
		b = a[-1]
		return float(b)

	#retain only those sequences within the desired OGT range
	logger.info("Retaining only those sequences with trait in desired range")
	in_range = []
	for x in MSA_file:
		for ranges in OGT_range:
			if ranges[0]<=temp(x.id)<ranges[1]:
				in_range.append(x)
	logger.info('Number of sequences in range: '+str(len(in_range)))				
	MSA_file= MultipleSeqAlignment(in_range)
	AlignIO.write(MSA_file,file.split('.')[0]+"_ranged.fa", "fasta")

#calculate sequence identity distribution
if distribution:
	logger.info("Calculating pairwise distribution")
	sequences = [x for MSA in files.keys() for x in AlignIO.read(MSA,'fasta')]
	sequences = MultipleSeqAlignment(sequences)
	AlignIO.write(sequences,list(files.keys())[0].split('_')[0]+"_all_ranged.fa", "fasta")
	_ = identity.compare_all(sequences,verbose)

logger.info('Exiting normally')
