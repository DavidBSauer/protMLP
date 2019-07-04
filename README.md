# protMLP
Predict from a protein sequence the growth temperature of the species of origin.

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
