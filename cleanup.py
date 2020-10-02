from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import Counter

def gaps(MSA_ref,threshold):
	"""Remove gap causing sequences based on provided threshold"""
	#the threshold is the maximum gap frequency allowed at any position within the alignment
	seq_num =0
	length = MSA_ref.get_alignment_length()
	seq_count = [0]*(length)
	#calculate gap frequency
	for x in MSA_ref:
		seq_num = seq_num+1
		for y in range(0,length):
			if x.seq[y] != "-":
				seq_count[y] = seq_count[y] +1.0
	
	gap_freq = [(1-(x / seq_num)) for x in seq_count]
	#remove gap causing sequences
	seqs_ref =[]
	for x in MSA_ref:
		to_write = True
		for y in range(0,length):
			if gap_freq[y] > threshold:
				if x.seq[y] != "-":
					to_write = False
		if to_write:
			seqs_ref.append(x)
	MSA2= MultipleSeqAlignment(seqs_ref)

	return MSA2
	
def removeXs(MSA,nucleotide):
	"""Remove sequences with non-canonical AAs or nucleotides"""
	seqs = []
	AA_set = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-','.'}
	if nucleotide:
		AA_set = {'A','C','G','T','-','.'}
	for x in MSA:
		if set(list(str(x.seq).upper())).issubset(AA_set):
			seqs.append(SeqRecord(Seq(str(x.seq).upper().replace('.','-'),x.seq.alphabet),x.id,'',''))
	MSA = MultipleSeqAlignment(seqs)
	return MSA

def rendundant_names(MSA):
	"""Give unique names to sequences"""
	names = []
	for x in MSA:
		names.append(x.id.split('/')[0].split('.')[0].split('_')[0])
	names = Counter(names)
	seqs = []
	for x in MSA:
		seqs.append(SeqRecord(Seq(str(x.seq),x.seq.alphabet),str(x.id.split('/')[0].split('.')[0].split('_')[0])+'#'+str(names[x.id.split('/')[0].split('.')[0]]),'',''))
		names[x.id.split('/')[0].split('.')[0]] = names[x.id.split('/')[0].split('.')[0]] -1
	MSA = MultipleSeqAlignment(seqs)
	return MSA

def remove_all_gaps(MSA):
	return [SeqRecord(Seq(str(x.seq).replace('-','').replace('.',''),x.seq.alphabet),x.id.split('|')[0],'','') for x in MSA]
