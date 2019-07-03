#!/usr/bin/env python3
#take in an MSA and assign OGTs, keeping only those in the desired range

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

parser = argparse.ArgumentParser(description="Step 2. Take in a species-Tg file and MSA files. Assign Tg's to all sequences based on assigned species, then remove sequences outside of provided Tg range.")
files = parser.add_argument_group('Required files')
def_range = 'all'

files = parser.add_argument_group('Required file')
files.add_argument("-t",action='store', type=str, help="The species-Tg file.",dest='ogt',default=None)
files.add_argument("-s","--seq",action='append', type=str, help="The MSA file in FASTA format.",dest='MSA_file',default=None)
parser.add_argument("-r", "--range",action='store', type=str, help="The range of Tg's to keep. Can be 'all', some combination of p/m/t for psychrophiles, mesophile, or thermophiles. Or a given range of temperatures, with '-' denoting ranges and ',' for multiple ranges. Examples: 'mt' or '25-35,45-65'. Default is "+str(def_range)+'.',dest='range',default=def_range)

args = parser.parse_args()
data_range = args.range

try:
	#read in species-OGT file
	logger.info("Species-Tg file: "+args.ogt)
	infile = open(args.ogt,'r')
	reader = csv.reader(infile,delimiter='\t')
	species_temp = dict((str(rows[0]),float(rows[1])) for rows in reader)
	infile.close()    
	logger.info("found "+str(len(species_temp.keys()))+" species-Tg pairs")
except:
	logger.info("Problem reading the species-Tg file. Quitting")
	print("Problem reading the species-Tg file. Quitting")
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

	thermo = ('t' in data_range)
	if thermo:
		OGT_range.append((float(45),float('inf')))
		thermo = True

if not(all_ogt or psych or meso or thermo):
	OGT_range = [(float(subrange.split('-')[0]),float(subrange.split('-')[1])) for subrange in data_range.split(',')]

logger.info('Including sequences within the Tg ranges of: '+', '.join([' to '.join([str(y) for y in x]) for x in OGT_range]))

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
	logger.info("Assigning Tg's to sequences for MSA")
	for x in MSA_file:
		if spec(x.id) in species_temp.keys():
			assigned.append(SeqRecord(Seq(str(x.seq),x.seq.alphabet), x.id+'|'+str(species_temp[spec(x.id)]),'',''))
	logger.info('Number of sequences with assigned Tg: '+str(len(assigned)))				
	MSA_file= MultipleSeqAlignment(assigned)
	AlignIO.write(MSA_file,file.split('.')[0]+"_assigned.fa", "fasta")

	def temp(txt):
		a = txt.split('|')
		b = a[-1]
		return float(b)

	#retain only those sequences within the desired OGT range
	logger.info("Retaining only those Tg's in range")
	in_range = []
	for x in MSA_file:
		for ranges in OGT_range:
			if ranges[0]<=temp(x.id)<ranges[1]:
				in_range.append(x)
	logger.info('Number of sequences in range: '+str(len(in_range)))				
	MSA_file= MultipleSeqAlignment(in_range)
	AlignIO.write(MSA_file,file.split('.')[0]+"_ranged.fa", "fasta")

logger.info('Exiting normally')
