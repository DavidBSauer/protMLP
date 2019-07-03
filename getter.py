#!/usr/bin/env python3
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from urllib.request import urlopen
import logging
logger = logging.getLogger('species_assignment')
import time
import os
import datetime
from Bio.Align import MultipleSeqAlignment
from tqdm import tqdm

logger.info('accessing uniprot on: '+datetime.datetime.utcnow().strftime('%m/%d/%Y'))

min_time_per = 0.34 #max of ~3 per second

def uniprot(txt):
	return txt.split('_')[0]

def retriever(x):
	"""Get species of origin for protein (if available) and return if protein is non-fragment"""
	t0=time.time()
	to_return = (False,None)
	url = 'http://www.uniprot.org/uniprot/'+uniprot(x)+'.xml'	
	try:
		response = urlopen(url,timeout=900)
	except:
		logger.info("trouble downloading: "+uniprot(x))
	else:
		file = open('./'+x+'.xml', "w")
		file.write(response.read().decode('utf-8'))
		file.close()
		file = open('./'+x+'.xml', 'r')
		try:
			new_seq = SeqIO.read(file, "uniprot-xml")
		except:
			logger.info("cannot read xml file for: "+x)
		else:
			try:
				new_spec = new_seq.annotations['organism'] 
			except:
				logger.info("no species info for: "+x)
			else:
				if not('sequence_fragment' in new_seq.annotations):		
					species = '_'.join(new_spec.split()[0:2]).lower()
					to_return=(True,species)
				else:
					logger.info("fragment, skipping: "+x)
		os.remove('./'+x+'.xml')
	time.sleep(max(0,min_time_per-(time.time()-t0))) #avoid abusing the server, query up to 3 per second
	return to_return	

def non_frag_spec(MSA):
	"""Label each sequence with the species of origin if available and the sequence is non-fragment"""
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
	
