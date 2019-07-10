# protMLP
Predict from a protein sequence the growth temperature of the species of origin.

See: Sauer & Wang. Using machine learning to predict organismal growth temperatures from protein primary sequences (2019) https://doi.org/10.1101/677328

## Installation and Requirements
This has been developed and tested on Ubuntu 18.04 LTS. The scripts should work on any system supporting Python 3, so long as the external programs are installed properly.

1. Download these scripts. This is easiest using git to clone the repository.
```
git clone https://github.com/DavidBSauer/protMLLP
```

2. Install the requirements.
These scripts depend upon Python3 with the following python packages also installed: numpy, scipy, matplotlib, biopython, pandas, tqdm, pydot, graphviz, and tensorflow.

To install everything in Ubuntu (or other system that use the apt package manager), go into the downloaded directory and use the pre-made bash script. 
```
cd OGT_prediction
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
Take in a species-Tg file and MSA files. Assign Tg's to all sequences based on assigned species, then remove sequences outside of provided Tg range.

```
python3 step2.py -s MSA_file.fa -r 25-35,45-65
```

## Step 3 - One-hot encode the protein sequences and train MLPs
One-hot encode the protein sequences. Optionally remove less significant columns and/or rebalance the data. Calculate a linear regression and MLPs.

```
python3 step3.py -tr training_file.fa -te testing_file.fa -v validation_file.fa -o 1 -ld 5 -p -i
```
