# protMLP
Use homologus protein sequences to predict the growth temperature of the species of origin.

## Step 1 - Retrieve the species of origin for each protein seqeunce.
Take in an MSA file and assign the species from Uniprot (locally or via web). Will remove fragments and gap incuding sequences and generate train, test, and validation MSAs

Required file:
  -f INPUT_FASTA, --fasta INPUT_FASTA
                        The MSA file in FASTA format. Can be invoked multiple times for multiple files.
  -s INPUT_STOCK, --stock INPUT_STOCK
                        The MSA file in Stockholm format. Can be invoked multiple times for multiple files.

optional arguments:
  -h, --help            show help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        Removing sequences which cause gaps at a frequency greater than provided frequency. Default is 0.995.
  -g, --group           
			Place idential sequences into the same train/test/valid dataset.
  -w, --web             
			Access the uniprot website for Uniprot information on each sequence. This is the default behavior.
  -lx GETTERS_LOCAL_XML, --local_xml GETTERS_LOCAL_XML
                        Provide a local XML Uniprot database file. Will use this file for retrieving information on each sequence. Can be invoked repeatedly for multiple files.
  -ld GETTERS_LOCAL_DAT, --local_dat GETTERS_LOCAL_DAT
                        Provide a local DAT Uniprot database file. Will use this file for retrieving information on each sequence. Can be invoked repeatedly for multiple files.

## Step 2 - Remove protein sequences outside of a provided growth temperature range.
Take in a species-Tg file and MSA files. Assign Tg's to all sequences based on assigned species, then remove sequences outside of provided Tg range.

Required file:
  -t OGT                
			The species-Tg file.
  -s MSA_FILE, --seq MSA_FILE
                        The MSA file in FASTA format.

optional arguments:
  -h, --help            
			show help message and exit
  -r RANGE, --range RANGE
                        The range of Tg's to keep. Can be 'all', some combination of p/m/t for psychrophiles, mesophile, or thermophiles. Or a given range of temperatures, with '-' denoting ranges and ',' for multiple ranges. Examples: 'mt' or '25-35,45-65'. Default is all.

## Step 3 - One-hot encode the protein sequences and train MLPs
One-hot encode the protein sequences. Optionally remove less significant columns and/or rebalance the data. Calculate a linear regression and MLPs.

optional arguments:
  -h, --help            
			show this help message and exit
  -tr TRAIN_FILE, --train TRAIN_FILE
                        The MSA training file in FASTA format.
  -te TEST_FILE, --test TEST_FILE
                        The MSA testing file in FASTA format.
  -v VAL_FILE, --validation VAL_FILE
                        The MSA validation file in FASTA format.
  -o OVERDETERMINED, --overdetermined OVERDETERMINED
                        The overdetermined level required of the MLPs. Default is 1.
  -ld MAX_DEPTH, --max_depth MAX_DEPTH
                        The maximum number of hidden layers (depth) in the MLPs. Default is 5.
  -lw MAX_LAYER, --max_width MAX_LAYER
                        The maximum number of nodes (width) in a layer of the MLPs. Default is inf.
  -n MAX_NUM, --max_MLP MAX_NUM
                        The maximum number of MLPs to train per generation. Default is 500.
  -b, --balance         
			The balance the training MSA. Default is False.
  -th TH_THRESHOLD, --th_threshold TH_THRESHOLD
                        Threshold for the tophat fit correlation coefficients. Used to exclude columns in the MSA. Default is None.
  -pb BS_THRESHOLD, --pb_threshold BS_THRESHOLD
                        Threshold for the point-biserial correlation coefficients. Used to exclude columns in the MSA. Default is None.
  -p, --parallel        
			Run parallel where-ever possible. Avoid using if there are errors (sometimes seen in the labels of plots). Default is False
  -g G, --generations G
                        Number of generations to run the genetic algorithm. Default is 10
  -k TO_KEEP, --keep TO_KEEP
                        Fraction of the MLPs to keep every generation. Default is 0.2
  -i, --identity        
			Train regressions with an identity activation function for comparison. Default is False

