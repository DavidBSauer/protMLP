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
	global DBs
	for file in files:
		try:
			DBs.append(SeqIO.index(file,'swiss'))
		except:
			logger.info('Problem loading the Uniprot DAT file: '+file+'. Quitting.')
			sys.exit()
		else:
			logger.info('Loaded the Uniprot DAT file: '+file+'')


def retriever(x):
	to_return = (False,None)
	for db in DBs:
		if x in db:
			new_seq = db[x]
			try:
				new_spec = new_seq.annotations['organism'] 
			except:
				logger.info("No species info for: "+x)
			else:
				if not('sequence_fragment' in new_seq.annotations):		
					species = '_'.join(new_spec.split()[0:2])
					to_return = (True,species)
				else:
					logger.info("Fragment, skipping: "+x)
	return to_return

def non_frag_spec(MSA):
	#run for list of ids to avoid repeatedly looking up multi-domain proteins
	ids = list(set([uniprot(x.id) for x in MSA]))
	logger.info('Uniprot records to retrieve: '+str(len(ids)))
	id_species ={id:retriever(id) for id in tqdm(ids,unit='ID')}
	id_species ={id:id_species[id][1] for id in id_species.keys() if id_species[id][0]}
	logger.info('Uniprot records retrieved, non-fragment, with species: '+str(len(id_species.keys())))
	#add the species if it is available
	MSA_new = [SeqRecord(Seq(str(x.seq),x.seq.alphabet), str(x.id+"|"+id_species[uniprot(x.id)]),'','') for x in MSA if uniprot(x.id) in id_species]
	MSA_new= MultipleSeqAlignment(MSA_new)
	return MSA_new
	
