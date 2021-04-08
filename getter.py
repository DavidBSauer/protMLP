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
	return txt.split('#')[0]

def retriever_spec(x):
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
				species = '_'.join(new_spec.split()[0:2]).lower()
				to_return=(True,species)
		os.remove('./'+x+'.xml')
	time.sleep(max(0,min_time_per-(time.time()-t0))) #avoid abusing the server, query up to 3 per second
	return to_return	

def retriever_non_frag(x):
	"""Get species of origin for protein (if available) and return if protein is non-fragment"""
	t0=time.time()
	to_return = False
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
			if not('sequence_fragment' in new_seq.annotations):		
				to_return=True
			else:
				logger.info("fragment, skipping: "+x)
		os.remove('./'+x+'.xml')
	time.sleep(max(0,min_time_per-(time.time()-t0))) #avoid abusing the server, query up to 3 per second
	return to_return	


def spec(MSA,verbose):
	"""Label each sequence with the species of origin"""
	#run for list of ids to avoid repeatedly looking up multi-domain proteins
	ids = list(set([uniprot(x.id) for x in MSA]))
	logger.info('Uniprot records to retrieve: '+str(len(ids)))
	if verbose:
		print('Finding species records')
		id_species ={id:retriever_spec(id) for id in tqdm(ids,unit='ID')}
	else:
		id_species ={id:retriever_spec(id) for id in ids}
	id_species ={id:id_species[id][1] for id in id_species.keys() if id_species[id][0]}
	logger.info('Uniprot records retrieved with species: '+str(len(id_species.keys())))
	#add the species if it is available
	MSA_new = [SeqRecord(Seq(str(x.seq)), str(x.id+"|"+id_species[uniprot(x.id)]),'','') for x in MSA if uniprot(x.id) in id_species]
	MSA_new= MultipleSeqAlignment(MSA_new)
	return MSA_new

def non_frag(MSA,verbose):
	"""Retain each sequence if non-fragment"""
	#run for list of ids to avoid repeatedly looking up multi-domain proteins
	ids = list(set([uniprot(x.id) for x in MSA]))
	logger.info('Uniprot records to retrieve: '+str(len(ids)))
	if verbose:
		print('Finding fragment sequences')
		id_nonfrag ={id:retriever_non_frag(id) for id in tqdm(ids,unit='ID')}
	else:
		id_nonfrag ={id:retriever_non_frag(id) for id in ids}
	id_nonfrag =[id for id in id_nonfrag.keys() if id_nonfrag[id]]
	logger.info('Uniprot records retrieved which are non-fragment: '+str(len(id_nonfrag)))
	#add the species if it is available
	MSA_new = [SeqRecord(Seq(str(x.seq)), str(x.id),'','') for x in MSA if uniprot(x.id) in id_nonfrag]
	MSA_new= MultipleSeqAlignment(MSA_new)
	return MSA_new
	
