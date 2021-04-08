#make a modifiable MLP NN using TF

import logging
logger = logging.getLogger('MLP_training')
try:
	import tensorflow as tf
except:
	logger.info('Tensorflow import failed, must be using another backend')
	import keras
else:
	logger.info('TensorFlow version: '+tf.__version__)
	del(tf)
	from tensorflow.python.client import device_lib
	devices = device_lib.list_local_devices()
	logger.info('Tensorflow devices:\n'+'\n'.join(['device type: '+a.device_type+'. name: '+a.name+'. memory_limit: '+str(a.memory_limit) for a in devices]))
	del(device_lib)
	from tensorflow import keras
import scipy
logger.info('SciPy version: '+scipy.__version__)
del(scipy)
import numpy as np
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import random as python_random

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

max_iter = 10**6
logger.info('Maximum number of training epochs: '+str(max_iter))
learning_rate = 0.01
logger.info('Learning rate: '+str(learning_rate))
beta_1 = 0.9
logger.info('Beta 1: '+str(beta_1))
beta_2 = 0.999
logger.info('Beta 2: '+str(beta_2))
epsilon = 1e-8
logger.info('Epsilon: '+str(epsilon))
batch_size = 500
logger.info('Batch size: '+str(batch_size))
alpha = 0.05
logger.info('Alpha: '+str(alpha))
n_train_seq = None
n_input = None
seq_data = None
unit = None
multiprocess = False
process = 1
max_q = 10
	
def setup_data(all_data,my_unit):
	"""Set constant variables"""
	global seq_data
	seq_data = all_data
	global n_train_seq
	global n_input	
	(n_train_seq,n_input) = all_data['train'].shape
	n_input = n_input-2
	global max_q
	max_q = math.ceil(n_train_seq/batch_size)
	#logger.info('Keras fit queue size: '+str(max_q))
	global unit
	unit = my_unit

def setup_params(parallel,seed):
	global multiprocess
	global process
	if parallel > 1:
		multiprocess = True
		process = parallel
	try:
		import tensorflow as tf
	except:
		logger.info('Tensorflow import failed, must be using another backend. Cannot set the seed.')
	else:
		tf.random.set_seed(seed)
	finally:
		np.random.seed(seed)
		python_random.seed(seed)	
	
def connections(model):
	"""Calculate the number of parameters of a topology"""
	biases = sum(model)+1
	model = [n_input]+list(model)+[1]
	weights = sum([model[x]*model[x+1] for x in range(0,len(model)-1,1)])
	conns = weights+biases
	return conns

def trainer(input):
	"""Train a MLP or linear regression"""
	(NN,activ)=input
	if activ == 'Linear':
		folder = './results/Linear'
		os.mkdir(folder)
	else:
		folder = './results/MLP_'+'-'.join([str(x) for x in NN])
		if not(os.path.isdir(folder)):
			os.mkdir(folder)
		folder = './results/MLP_'+'-'.join([str(x) for x in NN])+'/'+activ
		os.mkdir(folder)

	#calculate data/param ratio
	if activ == 'Linear':
		data_param = n_train_seq/(n_input+1)
	else:
		data_param = n_train_seq/connections(NN)
	
	train_data = seq_data['train'].copy(deep=True)	
	train_ids = train_data.pop('id')
	train_target = train_data.pop('target')
	valid_data = seq_data['valid'].copy(deep=True)
	valid_ids = valid_data.pop('id')
	valid_target = valid_data.pop('target')

	#reset graph
	keras.backend.clear_session()
	
	def build_model_relu():
		model = keras.Sequential()
		model.add(keras.layers.Dense(NN[0],activation='linear', input_shape=[n_input],use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal',name='Dense_0'))
		model.add(keras.layers.LeakyReLU(alpha=alpha,name='LReLu_0'))
		for y in range(1,len(NN),1):
			x = NN[y]
			model.add(keras.layers.Dense(x,activation='linear',use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal',name='Dense_'+str(y)))
			model.add(keras.layers.LeakyReLU(alpha=alpha,name='LReLu_'+str(y)))
		model.add(keras.layers.Dense(1,activation='linear',use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal',name='Output'))
		optimizer = keras.optimizers.Adam(lr=learning_rate,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
		model.compile(loss='mse',optimizer=optimizer,metrics=['mse'])
		return model

	def build_model_ident():
		model = keras.Sequential()
		model.add(keras.layers.Dense(NN[0],activation='linear', input_shape=[n_input],use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal',name='Dense_0'))
		for y in range(1,len(NN),1):
			x=NN[y]
			model.add(keras.layers.Dense(x,activation='linear',use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal',name='Dense_'+str(y)))
		model.add(keras.layers.Dense(1,activation='linear',use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal',name='Output'))
		optimizer = keras.optimizers.Adam(lr=learning_rate,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
		model.compile(loss='mse',optimizer=optimizer,metrics=['mse'])
		return model

	def build_model_linear():
		model = keras.Sequential([keras.layers.Dense(1,activation='linear',input_shape=[n_input],use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal',name='Dense_0')])
		optimizer = keras.optimizers.Adam(lr=learning_rate,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
		model.compile(loss='mse',optimizer=optimizer,metrics=['mse'])
		return model

	#run regression with early stopping and checkpointing
	if activ == 'ReLu':
		model = build_model_relu()
	elif activ =='Identity':
		model = build_model_ident()
	else:
		model = build_model_linear()

	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
	checkpoint = keras.callbacks.ModelCheckpoint(folder+'/model.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=0,save_freq='epoch')
	history = model.fit(train_data, train_target, epochs=max_iter,batch_size=batch_size, shuffle=True,validation_data = (valid_data,valid_target), verbose=0, use_multiprocessing=multiprocess, workers = process,max_queue_size=max_q, callbacks=[early_stop,checkpoint])	

	#predict the final results
	del(model)
	keras.backend.clear_session()
	model = keras.models.load_model(folder+'/model.h5')
	valid_pred = model.predict(valid_data).flatten()

	MSE = np.mean((valid_pred - valid_target.values) ** 2)
	MAE = np.mean(np.absolute(valid_pred - valid_target.values))
	RMSE = math.sqrt(MSE)
	r = pearsonr(valid_target.values,valid_pred)[0]
	spearman = spearmanr(valid_target.values,valid_pred)[0]
	del(model)
	
	return {'NN':NN,'MSE':MSE,'MAE':MAE,'RMSE':RMSE,'r':r,'spearman':spearman,'data_param':data_param,'folder':folder}

def final_eval(input):
	"""Evaluate a MLP against the test dataset"""
	(NN,activ)=input
	if activ == 'Linear':
		folder = './results/Linear'
	else:
		folder = './results/MLP_best'

	train_data = seq_data['train'].copy(deep=True)	
	train_ids = train_data.pop('id')
	train_target = train_data.pop('target')
	test_data = seq_data['test'].copy(deep=True)
	test_ids = test_data.pop('id')
	test_target = test_data.pop('target')
	valid_data = seq_data['valid'].copy(deep=True)
	valid_ids = valid_data.pop('id')
	valid_target = valid_data.pop('target')

	#calculate data/param ratio
	if activ == 'Linear':
		data_param = n_train_seq/(n_input+1)
	else:
		data_param = n_train_seq/connections(NN)

	#reset graph
	keras.backend.clear_session()

	model = keras.models.load_model(folder+'/model.h5')
	keras.utils.plot_model(model,show_shapes=True,show_layer_names=True,to_file=folder+'/model.png')
	
	#predict the final results
	test_pred = model.predict(test_data).flatten()
	valid_pred = model.predict(valid_data).flatten()
	train_pred = model.predict(train_data).flatten()
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
	MAE = np.mean(np.absolute(test_pred - test_target.values))
	RMSE = math.sqrt(MSE)
	r = pearsonr(test_target.values,test_pred)[0]
	spearman = spearmanr(test_target.values,test_pred)[0]
	
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
	if activ in ['ReLu','Identity']:
		ax.set_title('MLP Regression - '+activ+' Activation. MLP = '+'-'.join([str(x) for x in NN])+'\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f')+' data/param= '+format(data_param,'.1f'))
	else:
		ax.set_title('Linear Regression.\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f')+' data/param= '+format(data_param,'.1f'))		
	##save the figure
	ax.legend(loc='upper left')
	plt.savefig(folder+'/regression.png')
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
		if activ in ['ReLu','Identity']:
			ax.set_title('MLP Regression - '+activ+' Activation. MLP = '+'-'.join([str(x) for x in NN])+'\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f')+' data/param= '+format(data_param,'.1f'))
		else:
			ax.set_title('Linear Regression.\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f')+' data/param= '+format(data_param,'.1f'))		
		plt.savefig(folder+'/regression_residuals.png')
		plt.cla()
		plt.clf()
		plt.close()

	return {'NN':NN,'MSE':MSE,'MAE':MAE,'RMSE':RMSE,'r':r,'spearman':spearman,'data_param':data_param,'residuals':test_resid_dict,'folder':folder}

def knn(inputs):
	"""Train a k-Nearest-Neighbors regression"""
	(threads,max_k,verbose) = inputs
	folder = './results/kNN'
	os.mkdir(folder)
	
	train_data = seq_data['train'].copy(deep=True)	
	train_ids = train_data.pop('id')
	train_target = train_data.pop('target')
	valid_data = seq_data['valid'].copy(deep=True)
	valid_ids = valid_data.pop('id')
	valid_target = valid_data.pop('target')
	test_data = seq_data['test'].copy(deep=True)	
	test_ids = test_data.pop('id')
	test_target = test_data.pop('target')

	def knn_neighbors(inputs):
		"""return the kNN MSE of the validation set given a n number of neighbors"""
		(n,tx,ty,vx,vy)=inputs
		model = KNeighborsRegressor(n_neighbors=n,weights='uniform',metric='hamming',n_jobs=threads)
		model.fit(tx,ty)
		mse=np.mean((model.predict(vx).flatten()-vy) ** 2)
		rmse = math.sqrt(mse)
		r = pearsonr(model.predict(vx).flatten(),vy)[0]
		mae = np.mean(np.absolute(model.predict(vx).flatten()-vy))
		spearman = spearmanr(model.predict(vx).flatten(),vy)[0]
		return {'mse':mse,'rmse':rmse,'k':n,'r':r,'mae':mae,'spearman':spearman}
	
	n_range = [(n,train_data.values,train_target.values,valid_data.values,valid_target.values) for n in range(1,max_k+1,1)]
	python_random.shuffle(n_range)
	if verbose:
		print('Calculating kNN regression')
		results = map(knn_neighbors,tqdm(n_range,unit='neighbor'))
	else:
		results = map(knn_neighbors,n_range)
	results = list(results)
	n = min([x['k'] for x in results if x['mse'] == min([y['mse'] for y in results])]) #select the fewest number of neighbors with the minimum validation MSE
	logger.info('Best trained kNN validation MSE: '+str(min([y['mse'] for y in results])))
	logger.info('Best number of neighbors for kNN: '+str(n))
	model = KNeighborsRegressor(n_neighbors=n,weights='uniform',metric='hamming',n_jobs=threads)
	model.fit(train_data.values,train_target.values)
	
	#predict the final results
	test_pred = model.predict(test_data.values).flatten()
	valid_pred = model.predict(valid_data.values).flatten()
	train_pred = model.predict(train_data.values).flatten()
	#calculate residuals
	train_residuals = train_pred-train_target.values
	valid_residuals = valid_pred-valid_target.values
	test_residuals = test_pred-test_target.values
	
	#record the results	
	test_resid_dict = {}	
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
	MAE = np.mean(np.absolute(test_pred - test_target.values))
	RMSE = math.sqrt(MSE)
	r = pearsonr(test_target.values,test_pred)[0]
	spearman = spearmanr(test_target.values,test_pred)[0]
	
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
	ax.set_title('kNN Regression.\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f'))		
	##save the figure
	ax.legend(loc='upper left')
	plt.savefig(folder+'/regression.png')
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
		ax.set_title('kNN Regression.\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f'))		
		plt.savefig(folder+'/regression_residuals.png')
		plt.cla()
		plt.clf()
		plt.close()
	
	return {'MSE':MSE,'MAE':MAE,'RMSE':RMSE,'r':r,'spearman':spearman,'residuals':test_resid_dict,'folder':folder,'results':results}	


def RFR(inputs):
	"""Train a Random Forest regression"""
	(threads,overdetermined,verbose) = inputs
	folder = './results/RFR'
	os.mkdir(folder)
	
	train_data = seq_data['train'].copy(deep=True)	
	train_ids = train_data.pop('id')
	train_target = train_data.pop('target')
	valid_data = seq_data['valid'].copy(deep=True)
	valid_ids = valid_data.pop('id')
	valid_target = valid_data.pop('target')
	test_data = seq_data['test'].copy(deep=True)	
	test_ids = test_data.pop('id')
	test_target = test_data.pop('target')
	
	model = RandomForestRegressor(n_jobs=threads)
	model.fit(train_data.values,train_target.values)

	def rf_trees(inputs):
		"""return the RFR MSE of the validation set given a n number of trees"""
		(n,overdet,tx,ty,vx,vy)=inputs
		model = RandomForestRegressor(n_estimators=n,min_samples_split=2*n*overdet,n_jobs=threads)
		model.fit(tx,ty)
		mse=np.mean((model.predict(vx).flatten()-vy) ** 2)
		rmse = math.sqrt(mse)
		r = pearsonr(model.predict(vx).flatten(),vy)[0]
		mae = np.mean(np.absolute(model.predict(vx).flatten()-vy))
		spearman = spearmanr(model.predict(vx).flatten(),vy)[0]
		return {'mse':mse,'rmse':rmse,'k':n,'r':r,'mae':mae,'spearman':spearman}

	n_range = [(n,overdetermined,train_data.values,train_target.values,valid_data.values,valid_target.values) for n in [10,50,100,500,1000]] #use the default of 100 trees
	python_random.shuffle(n_range)
	if verbose:
		print('Calculating RF regression')
		results = map(rf_trees,tqdm(n_range,unit='Forest'))
	else:
		results = map(rf_trees,n_range)
	results = list(results)
	n = min([x['k'] for x in results if x['mse'] == min([y['mse'] for y in results])]) #select the fewest number of trees with the minimum validation MSE
	logger.info('Best trained RFR validation MSE: '+str(min([y['mse'] for y in results])))
	logger.info('Best number of neighbors for RFR: '+str(n))
	model = RandomForestRegressor(n_estimators=n,n_jobs=threads)
	model.fit(train_data.values,train_target.values)
	
	#predict the final results
	test_pred = model.predict(test_data.values).flatten()
	valid_pred = model.predict(valid_data.values).flatten()
	train_pred = model.predict(train_data.values).flatten()
	#calculate residuals
	train_residuals = train_pred-train_target.values
	valid_residuals = valid_pred-valid_target.values
	test_residuals = test_pred-test_target.values
	
	#record the results	
	test_resid_dict = {}	
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
	MAE = np.mean(np.absolute(test_pred - test_target.values))
	RMSE = math.sqrt(MSE)
	r = pearsonr(test_target.values,test_pred)[0]
	spearman = spearmanr(test_target.values,test_pred)[0]
	
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
	ax.set_title('RFR Regression.\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f'))		
	##save the figure
	ax.legend(loc='upper left')
	plt.savefig(folder+'/regression.png')
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
		ax.set_title('RFR Regression.\nMSE= '+format(MSE,'.3f')+', r= '+format(r,'.3f')+', RMSE= '+format(RMSE,'.3f'))		
		plt.savefig(folder+'/regression_residuals.png')
		plt.cla()
		plt.clf()
		plt.close()	
	return {'MSE':MSE,'MAE':MAE,'RMSE':RMSE,'r':r,'spearman':spearman,'residuals':test_resid_dict,'folder':folder,'results':results}
	
def infer_MLP(model,data):
	"""Load a model and infer the Tg of provided sequences"""	
	working_data = data.copy(deep=True)	
	ids = working_data.pop('id')
	model = keras.models.load_model(model)
	pred = model.predict(working_data).flatten()
	data=data.assign(prediction = [x for x in pred])
	return data
	
def infer_knn(k,seq_data,data,threads):
	"""Use kNN to infer the Tg of provided sequences"""	

	train_data = seq_data.copy(deep=True)	
	train_ids = train_data.pop('id')
	train_target = train_data.pop('target')
	test_data = data.copy(deep=True)	
	test_ids = test_data.pop('id')

	model = KNeighborsRegressor(n_neighbors=k,weights='uniform',metric='hamming',n_jobs=threads)
	model.fit(train_data.values,train_target.values)
	
	#predict the final results
	pred = model.predict(test_data.values).flatten()
	data=data.assign(prediction = [x for x in pred])
	return data
	
def infer_RFR(n,seq_data,data,threads):
	"""Use Random Forest Regressor to infer the Tg of provided sequences"""	

	train_data = seq_data.copy(deep=True)	
	train_ids = train_data.pop('id')
	train_target = train_data.pop('target')
	test_data = data.copy(deep=True)	
	test_ids = test_data.pop('id')

	model = RandomForestRegressor(n_estimators=n,n_jobs=threads)
	model.fit(train_data.values,train_target.values)

	#predict the final results
	pred = model.predict(test_data.values).flatten()
	data=data.assign(prediction = [x for x in pred])
	return data

