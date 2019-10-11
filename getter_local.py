#!/usr/bin/env python3
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import logging
logger = logging.getLogger('species_assignment')
import sys
from Bio.Align import MultipleSeqAlignment
from tqdm import tqdm 

def uniprot(txt):
	return txt.split('_')[0]

DBs =[]

def setup_xml(files):
	"""Load XML files of Uniprot data"""
	global DBs
	for file in files:
		try:
			DBs.append(SeqIO.index(file,'uniprot-xml'))
		except:
			logger.info('Problem loading the Uniprot XML file: '+file+'. Quitting.')
			sys.exit()
		else:
			logger.info('Loaded the Uniprot XML file: '+file+'')

def setup_dat(files):
	"""Load DAT files of Uniprot data"""
	global DBs
	for file in files:
		try:
			DBs.append(SeqIO.index(file,'swiss'))
		except:
			logger.info('Problem loading the Uniprot DAT file: '+file+'. Quitting.')
			sys.exit()
		else:
			logger.info('Loaded the Uniprot DAT file: '+file+'')


def retriever_spec(x):
	"""If sequences is non-fragment, return the species of origin for a sequence (if available)"""
	to_return = (False,None)
	for db in DBs:
		if x in db:
			new_seq = db[x]
			try:
				new_spec = new_seq.annotations['organism'] 
			except:
				logger.info("No species info for: "+x)
			else:
				species = '_'.join(new_spec.split()[0:2]).lower()
				to_return = (True,species)
	return to_return

def retriever_non_frag(x):
	"""If sequences is non-fragment, return the species of origin for a sequence (if available)"""
	to_return = False
	for db in DBs:
		if x in db:
			new_seq = db[x]
			if not('sequence_fragment' in new_seq.annotations):		
				to_return = True
			else:
				logger.info("Fragment, skipping: "+x)
	return to_return


def spec(MSA):
	"""Label each sequence with the species of origin"""
	#run for list of ids to avoid repeatedly looking up multi-domain proteins
	ids = list(set([uniprot(x.id) for x in MSA]))
	logger.info('Uniprot records to retrieve: '+str(len(ids)))
	id_species ={id:retriever_spec(id) for id in tqdm(ids,unit='ID')}
	id_species ={id:id_species[id][1] for id in id_species.keys() if id_species[id][0]}
	logger.info('Uniprot records retrieved with species: '+str(len(id_species.keys())))
	#add the species if it is available
	MSA_new = [SeqRecord(Seq(str(x.seq),x.seq.alphabet), str(x.id+"|"+id_species[uniprot(x.id)]),'','') for x in MSA if uniprot(x.id) in id_species]
	MSA_new= MultipleSeqAlignment(MSA_new)
	return MSA_new

def non_frag(MSA):
	"""Retain each sequence if non-fragment"""
	#run for list of ids to avoid repeatedly looking up multi-domain proteins
	ids = list(set([uniprot(x.id) for x in MSA]))
	logger.info('Uniprot records to retrieve: '+str(len(ids)))
	id_nonfrag ={id:retriever_non_frag(id) for id in tqdm(ids,unit='ID')}
	id_nonfrag =[id for id in id_nonfrag.keys() if id_nonfrag[id]]
	logger.info('Uniprot records retrieved which are non-fragment: '+str(len(id_nonfrag)))
	#add the species if it is available
	MSA_new = [SeqRecord(Seq(str(x.seq),x.seq.alphabet), str(x.id),'','') for x in MSA if uniprot(x.id) in id_nonfrag]
	MSA_new= MultipleSeqAlignment(MSA_new)
	return MSA_new	
