# protMLP
Predict from a protein sequence the growth temperature of the species of origin.

See: Sauer & Wang. Using machine learning to predict organismal growth temperatures from protein primary sequences (2019) https://doi.org/10.1101/677328

## Installation and Requirements
This has been developed and tested on Ubuntu 18.04 LTS. The scripts should work on any system supporting Python 3, so long as the external programs are installed properly.

1. Download these scripts. This is easiest using git to clone the repository.
```
git clone https://github.com/DavidBSauer/protMLP
```

2. Install the requirements.
These scripts depend upon Python3 with the following python packages also installed: numpy, scipy, matplotlib, biopython, pandas, tqdm, pydot, graphviz, and tensorflow.

To install everything in Ubuntu (or other system that use the apt package manager), go into the downloaded directory and use the pre-made bash script. 
```
cd protMLP
./Ubuntu_setup.bash
```

## Step 1 - Retrieve the species of origin for each protein seqeunce.
Take in an MSA file and assign the species from Uniprot (locally or via web). Will remove fragments and gap incuding sequences and generate train, test, and validation MSAs

Run locally as:
```
python3 step1.py -f fasta_sequence_file.fa -s stockholm_sequence_file.stk -ld local_copy_of_Uniprot_DAT_file
```

Run over web as:
```
python3 step1.py -f fasta_sequence_file.fa -s stockholm_sequence_file.stk -w
```


## Step 2 - Remove protein sequences outside of a provided growth temperature range.
Take in a species-Tg file and MSA files. Assign Tg's to all sequences based on species of origin, then remove sequences outside of provided Tg range.

```
python3 step2.py -s MSA_file.fa -r 25-35,45-65
```

## Step 3 - One-hot encode the protein sequences and train MLPs
One-hot encode the protein sequences, the calculate a linear regression and MLPs. Optionally remove amino acids which are not correlated with Tg and/or balance the training data. 

```
python3 step3.py -tr training_file.fa -te testing_file.fa -v validation_file.fa -o 1 -ld 5 -p -i
```

## Predicting Tg from sequences
Predict the Tg of a provided set of sequences in FASTA format. Note: to get meaningful results the sequences must be aligned to the training MSA.

```
python3 predictor.py -s sequences.fa -t NN_AA_template.txt -m model.h5 -p
```

## Predict Tg of point mutants to a provided sequence
Given a provided protein sequence, predict the Tg of all possible amino acids observed at each position of the training MSA. Note, can predict compound (double, triple, etc) mutants also. However, mutational space increase exponentially with the number of mutations, therefore requiring exponentially more CPU-time and memory to calculate. If the program crashes, try decreasing the batch size.

```
python3 point_mutant_screening.py -s sequences.fa -t NN_AA_template.txt -m model.h5 -p -n 1
```