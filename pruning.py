#prune a MLP per given inputs
import logging
logger = logging.getLogger('pruning')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('Pruning.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import converter
import argparse
import sys
import os
from tensorflow import keras
from Bio import AlignIO
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from tqdm import tqdm
import tensorflow.keras.backend as K
import random

parallel =1
prune = None
unit = 'Tg'
scan = None
verbose = False

parser = argparse.ArgumentParser(description='Given a trained model and input template, predict the optimal growth temperature of a FASTA file of protein sequences.')
parser.add_argument("-m","--model",action='store', type=str, help="The regression model (model.h5) file to use.",dest='model',default=None)
parser.add_argument("-t","--template",action='store', type=str, help="The template file of MLP inputs. (From step3 this will be the file NN_AA_template.txt.)",dest='template',default=None)
parser.add_argument("-s", "--sparcity",help="Sparcity target in the final MLP after pruning. Fraction of the weights to set to zero. Set 'auto' to prune until the MLP is overdetermined.",action="store",dest='prune',default=prune,type=str)
parser.add_argument("-tr","--train",action='store', type=str, help="The MSA training file in FASTA format.",dest='train_file',default=None)
parser.add_argument("-te","--test",action='store', type=str, help="The MSA testing file in FASTA format.",dest='test_file',default=None)
parser.add_argument("-vd", "--validation", type=str,help="The MSA validation file in FASTA format.",action="store",dest='val_file',default=None)
parser.add_argument("-p", "--parallel",help="Number of threads to run in parallel. Avoid using if there are errors (sometimes seen in the labels of plots). Default is "+str(parallel),action="store",dest='parallel',default=parallel)
parser.add_argument("-u", "--unit",help="Description/unit for the regression target. Default is "+str(unit),action="store",dest='unit',default=unit)
parser.add_argument("-sc", "--scan",type=float,help="Scan sparcity levels in 0.01 (1 percent) increments.",action="store",dest='scan',default=scan)
parser.add_argument("-v", "--verbose",help="Verbose. Show progress bars while training MLPs. Default is "+str(verbose),action="store_true",dest='verbose',default=verbose)

folder = './pruned'
if not(os.path.isdir(folder)):
	os.mkdir(folder)

args = parser.parse_args()
prune = args.prune
model = args.model
training_file = args.train_file
testing_file = args.test_file
val_file = args.val_file
template = args.template
parallel = int(args.parallel)
unit = args.unit
scan = args.scan
verbose = args.verbose

logger.info('Template file: '+str(template))
logger.info('Model file: '+str(model))
logger.info('Number of threads to run in parallel: '+str(parallel))
logger.info('Sparcity target: '+str(prune))
logger.info('Scan sparcity increment: '+str(scan))
logger.info('Train file: '+str(training_file))
logger.info('Test file: '+str(testing_file))
logger.info('Validation file: '+str(val_file))
logger.info('Description/unit of the regression target: '+str(unit))

logger.info('Verbose: '+str(verbose))

old_model = keras.models.load_model(model)

#read in the template
if not(os.path.isfile(template)):
	logger.info('Problem reading the template file. Quitting.')
	print('Problem reading the template file. Quitting.')
	sys.exit()
else:
	f = open(template,'r')
	template = [line.strip() for line in f.readlines()]
	f.close()

#load and convert the input MSAs
try:
	training_file = AlignIO.read(training_file,'fasta')
except:
	logger.info('Unreadable training MSA file, provided as '+training_file)
	print('Unreadable training MSA file, provided as '+training_file)
	sys.exit()
else:
	pass

try:
	testing_file = AlignIO.read(testing_file,'fasta')
except:
	logger.info('Unreadable testing MSA file, provided as '+testing_file)
	print('Unreadable testing MSA file, provided as '+testing_file)
	sys.exit()
else:
	pass	

try:
	val_file = AlignIO.read(val_file,'fasta')
except:
	logger.info('Unreadable validation MSA file, provided as '+val_file)
	print('Unreadable validationn MSA file, provided as '+val_file)
	sys.exit()
else:
	pass

training_file = converter.convert_on_template(training_file,template,parallel)
testing_file = converter.convert_on_template(testing_file,template,parallel)
val_file = converter.convert_on_template(val_file,template,parallel)

train_data = training_file.copy(deep=True)	
train_ids = train_data.pop('id')
train_target = train_data.pop('target')
test_data = testing_file.copy(deep=True)
test_ids = test_data.pop('id')
test_target = test_data.pop('target')
valid_data = val_file.copy(deep=True)
valid_ids = valid_data.pop('id')
valid_target = valid_data.pop('target')
training_count = train_data.shape[0]

#set the sparcity
trainable_count = np.sum([K.count_params(w) for w in old_model.trainable_weights])

batch_size = 500
epochs = 10**6
end_step = np.ceil(training_count / batch_size).astype(np.int32) * epochs

if not(prune == None):
	logger.info('Pruning to a specific value.')
	if prune == 'auto':
		logger.info('No sparcity target given. Calculating a target such that the model is overdetermined.')
		logger.info('Number of trainable parameters in the initial model: '+str(trainable_count))
		logger.info('Number of training examples: '+str(training_count))
		prune = max([1.0 - float(training_count)/trainable_count,0])
		logger.info('Sparcity target for an overdetermined network: '+str(prune))

	else:
		logger.info('Pruning to provided value.')
		try:
			prune = float(prune)
		except:
			logger.info('Problem converting provided prune value to float.')
			sys.exit()
		else:
			pass

	data_param = training_count / ((1.0 - prune)*trainable_count)
	logger.info('Data to parameter ratio: '+str(data_param))
	# Compute end step to finish pruning after 10 epochs.

	prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
	pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=np.float32(prune),begin_step=0,end_step=end_step,frequency=1)}
	model_for_pruning = prune_low_magnitude(old_model, **pruning_params)
	model_for_pruning.compile(optimizer='adam',loss='mse',metrics=['mse'])
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
	#checkpoint = keras.callbacks.ModelCheckpoint(folder+'/pruned_model.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=0,save_freq='epoch')
	#model_for_pruning.fit(train_data, train_target,batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_target),callbacks=[early_stop,checkpoint,tfmot.sparsity.keras.UpdatePruningStep()],verbose=1)
	model_for_pruning.fit(train_data, train_target,batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_target),callbacks=[early_stop,tfmot.sparsity.keras.UpdatePruningStep()],verbose=0)


	_, baseline_model_accuracy = old_model.evaluate(test_data, test_target, verbose=0)
	logger.info('Initial model MSE: '+str(baseline_model_accuracy))
	_, prune_model_accuracy = model_for_pruning.evaluate(test_data, test_target, verbose=0)
	logger.info('Pruned model MSE: '+str(prune_model_accuracy))

	model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
	tf.keras.models.save_model(model_for_export, folder+'/pruned_model.h5', include_optimizer=True)

	#reset graph
	del(old_model)
	del(model_for_pruning)
	keras.backend.clear_session()
	new_model = keras.models.load_model(folder+'/pruned_model.h5',compile=False)
	keras.utils.plot_model(new_model,show_shapes=True,show_layer_names=True,to_file=folder+'/pruned_model.png')

	#for i, w in enumerate(model.get_weights()):
	#    print("{} -- Total:{}, Zeros: {:.2f}%".format(model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100))

	#predict the final results
	test_pred = new_model.predict(test_data).flatten()
	valid_pred = new_model.predict(valid_data).flatten()
	train_pred = new_model.predict(train_data).flatten()

	#calculate residuals
	train_residuals = train_pred-train_target.values
	valid_residuals = valid_pred-valid_target.values
	test_residuals = test_pred-test_target.values

	test_resid_dict = {}

	#record the results		
	f = open(folder+'/train_results.tsv','w')
	f.write('sequence\ttarget_value\tpredicted_value\tresidual difference\n')
	for x in train_ids.index.values:
		f.write(train_ids.loc[x]+'\t'+str(train_target.loc[x])+'\t'+str(train_pred[x])+'\t'+str(train_residuals[x])+'\n')
	f.close()
	f = open(folder+'/test_results.tsv','w')
	f.write('sequence\ttarget_value\tpredicted_value\tresidual difference\n')
	for x in test_ids.index.values:
		f.write(test_ids.loc[x]+'\t'+str(test_target.loc[x])+'\t'+str(test_pred[x])+'\t'+str(test_residuals[x])+'\n')
		test_resid_dict[test_ids.loc[x]] = test_residuals[x]
	f.close()
	f = open(folder+'/valid_results.tsv','w')
	f.write('sequence\ttarget_value\tpredicted_value\tresidual difference\n')
	for x in valid_ids.index.values:
		f.write(valid_ids.loc[x]+'\t'+str(valid_target.loc[x])+'\t'+str(valid_pred[x])+'\t'+str(valid_residuals[x])+'\n')
	f.close()

	MSE = np.mean((test_pred - test_target.values) ** 2)
	logger.info('Test MSE: '+str(MSE))
	RMSE = math.sqrt(MSE)
	logger.info('Test RMSE: '+str(RMSE))
	r = pearsonr(test_target.values,test_pred)[0]
	logger.info('Test Pearson r: '+str(r))

	#plot the data
	fig, ax = plt.subplots(nrows=1,ncols=1)
	## find figure bounds
	all_targets = list(train_target.values)+list(test_target.values)+list(valid_target.values)
	all_preds = np.array(list(train_pred)+list(test_pred)+list(valid_pred))
	all_preds = list(all_preds[np.logical_not(np.isnan(all_preds))])
	all_values = all_targets+all_preds
	minimum = min(all_values) 
	maximum = max(all_values) 
	ax.set_ylim(minimum-5,maximum+5)
	ax.set_xlim(minimum-5,maximum+5)
	ax.plot(range(int(minimum*0.9)-1,int(maximum*1.1)+1,1),range(int(minimum*0.9)-1,int(maximum*1.1)+1,1),'--',linewidth=3,color='#566573',alpha=1)
	ax.set_xlabel('Reported '+unit)
	ax.set_ylabel('Predicted '+unit)
	#plot the training data
	ax.plot(train_target.values,train_pred,'.',label='Training Set',rasterized=True,markersize=14,color='#222299',alpha=0.6)
	#plot the valid data
	ax.plot(valid_target.values,valid_pred,'.',fillstyle='none',label='Validation Set',rasterized=True,markersize=14,color='#222299',alpha=0.6)
	#plot the testing data
	ax.plot(test_target.values,test_pred,'g.',label='Test Set',rasterized=True,markersize=14,alpha=0.6)
	ax.set_title('Pruned MLP Regression\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f')+' data/param= '+format(data_param,'.1f'))

	##save the figure
	ax.legend(loc='upper left')
	plt.savefig(folder+'/pruned_regression.png')
	plt.cla()
	plt.clf()
	plt.close()

	#plot the residuals
	fig, ax = plt.subplots(nrows=1,ncols=1)
	## find figure bounds
	xminimum = min(list(train_target.values)+list(valid_target.values)+list(test_target.values))
	xmaximum = max(list(train_target.values)+list(valid_target.values)+list(test_target.values))
	ax.set_xlim(xminimum-5,xmaximum+5)
	residuals = [abs(x) for x in list(train_residuals)+list(valid_residuals)+list(test_residuals) if not(math.isnan(float(x)))]
	if len(residuals)>0:
		yrange = max(residuals)
		ax.set_ylim(-1.1*yrange,1.1*yrange)
		ax.plot(range(int(minimum*0.9)-1,int(maximum*1.1)+1,1),[0]*len(range(int(minimum*0.9)-1,int(maximum*1.1)+1,1)),'--',linewidth=3,color='#566573',alpha=1)
		ax.set_xlabel('Reported '+unit)
		ax.set_ylabel('Prediction '+unit)
		#plot the training data
		ax.plot(train_target.values,train_residuals,'.',label='Training Set',rasterized=True,markersize=14,color='#222299',alpha=0.6)
		#plot the valid data
		ax.plot(valid_target.values,valid_residuals,'.',fillstyle='none',label='Validation Set',rasterized=True,markersize=14,color='#222299',alpha=0.6)
		#plot the testing data
		ax.plot(test_target.values,test_residuals,'g.',label='Test Set',rasterized=True,markersize=14,alpha=0.6)
		##save the figure
		ax.legend(loc='lower left')
		ax.set_title('Pruned MLP Regression\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f')+' data/param= '+format(data_param,'.1f'))
		plt.savefig(folder+'/pruned_regression_residuals.png')
		plt.cla()
		plt.clf()
		plt.close()

def pruner(prune_level,model):
	try:
		old_model = keras.models.load_model(model)
		data_param = training_count / ((1.0 - prune)*trainable_count)
		prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
		pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=np.float32(prune),begin_step=0,end_step=end_step,frequency=1)}
		model_for_pruning = prune_low_magnitude(old_model, **pruning_params)
		model_for_pruning.compile(optimizer='adam',loss='mse',metrics=['mse'])
		early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
		#checkpoint = keras.callbacks.ModelCheckpoint(folder+'/pruned_model.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=0,save_freq='epoch')
		#model_for_pruning.fit(train_data, train_target,batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_target),callbacks=[early_stop,checkpoint,tfmot.sparsity.keras.UpdatePruningStep()],verbose=1)
		model_for_pruning.fit(train_data, train_target,batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_target),callbacks=[early_stop,tfmot.sparsity.keras.UpdatePruningStep()],verbose=0)
	
		_, prune_model_accuracy = model_for_pruning.evaluate(test_data, test_target, verbose=0)
		model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
		tf.keras.models.save_model(model_for_export, folder+'/scan_model.h5', include_optimizer=True)
	
		#reset graph
		del(old_model)
		del(model_for_pruning)
		keras.backend.clear_session()
		new_model = keras.models.load_model(folder+'/scan_model.h5',compile=False)
	
		#predict the final results
		test_pred = new_model.predict(test_data).flatten()
		valid_pred = new_model.predict(valid_data).flatten()
		train_pred = new_model.predict(train_data).flatten()
	
		MSE = np.mean((test_pred - test_target.values) ** 2)
		RMSE = math.sqrt(MSE)
		r = pearsonr(test_target.values,test_pred)[0]
	except:
		return (np.nan,np.nan,np.nan,np.nan)
	else:
		return (data_param,MSE,RMSE,r)

if not(scan == None):
	logger.info('Scanning sparcity')
	MSEs = []
	RMSEs = []
	rs = []
	data_params = []
	g = open(folder+'/sparcity_scan.tsv','w')
	g.write('sparcity\tdata/param\tMSE\tRMSE\tr\n')
	scan_range = list(np.arange(start=1-scan, stop=0, step=-1*scan)) + [0]
	random.shuffle(scan_range) #shuffle to get better ETA estimate
	if verbose:
		print('Screening varying levels of sparcity.')
		for prune in tqdm(scan_range):
			(data_param,MSE,RMSE,r) = pruner(prune,model)
			MSEs.append(MSE)
			RMSEs.append(RMSE)
			rs.append(r)
			data_params.append(data_param)
			g.write(str(prune)+'\t'+str(data_param)+'\t'+str(MSE)+'\t'+str(RMSE)+'\t'+str(r)+'\n')
	else:
		for prune in scan_range:
			(data_param,MSE,RMSE,r) = pruner(prune,model)
			MSEs.append(MSE)
			RMSEs.append(RMSE)
			rs.append(r)
			data_params.append(data_param)
			g.write(str(prune)+'\t'+str(data_param)+'\t'+str(MSE)+'\t'+str(RMSE)+'\t'+str(r)+'\n')
	g.close()
	os.remove(folder+'/scan_model.h5')

	plt.plot(scan_range,MSEs,'.',label='MSEs',markersize=14,alpha=1.0)
	plt.xlim(-0.05,1.05)
	plt.grid()
	plt.xlabel('Sparcity')
	plt.ylabel('MSE of Test Data')
	plt.legend()
	plt.savefig(folder+'/MLP_sparcity_vs_MSE.png')
	plt.cla()
	plt.clf()
	plt.close()	

	plt.plot(scan_range,RMSEs,'.',label='RMSEs',markersize=14,alpha=1.0)
	plt.xlim(-0.05,1.05)
	plt.grid()
	plt.xlabel('Sparcity')
	plt.ylabel('RMSE of Test Data')
	plt.legend()
	plt.savefig(folder+'/MLP_sparcity_vs_RMSE.png')
	plt.cla()
	plt.clf()
	plt.close()	

	plt.plot(scan_range,rs,'.',label="Pearson r's",markersize=14,alpha=1.0)
	plt.xlim(-0.05,1.05)
	plt.grid()
	plt.xlabel('Sparcity')
	plt.ylabel("r of Test Data")
	plt.legend()
	plt.savefig(folder+'/MLP_sparcity_vs_Pearson_r.png')
	plt.cla()
	plt.clf()
	plt.close()	

	plt.plot(scan_range,data_params,'.',label="Data/Parameter ratio",markersize=14,alpha=1.0)
	plt.xlim(-0.05,1.05)
	plt.grid()
	plt.xlabel('Sparcity')
	plt.ylabel("Data/Parameter ratio")
	plt.legend()
	plt.savefig(folder+'/MLP_sparcity_vs_data_parameter.png')
	plt.cla()
	plt.clf()
	plt.close()	
