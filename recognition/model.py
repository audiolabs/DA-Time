import tensorflow as tf
from keras import backend as K, layers
from keras.models import Model, Input, load_model
from keras.layers.merge import add
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_contrib.layers import CRF
from recognition.crf_losses import crf_loss as crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.optimizers import Adam
import nltk
import recognition.util as util

import numpy as np
import pandas as pd 
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import os.path
import recognition.eval as eval
import os
from sklearn.metrics import confusion_matrix
import recognition.plot as plot 
import pickle

from flair.embeddings import BertEmbeddings
from flair.data import Sentence, Token

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#=========================================================================
#
#  This file defines the core TE recognition neural model         
#
#=========================================================================

class BertEmbedder:
	"""Embed Bert Embeddings"""	
	def __init__(self, len, emb='en'):
		"""
		Args:
			len (int): max length for the model input
			lang (str, optional): embedding language. Defaults to 'en'.
		"""		
		if emb=='en':	self.embedder = BertEmbeddings("distilbert-base-uncased")
		self.MAX_LEN = len
		
	def embed_sentence(self, sentence):
		"""This function embed each sentence with BERT embedder

		Args:
			sentence (str): raw sentence

		Returns:
			np.array: embedded matrix
		"""		
		flair_sentence = Sentence(sentence)
		while len(flair_sentence) < self.MAX_LEN:	flair_sentence.add_token(Token("__PAD__"))
		self.embedder.embed(flair_sentence)
		return np.stack([t.embedding.cpu().numpy() for t in flair_sentence])


class TErecognizer:
	"""TE Recognizer model"""

	def __init__(self, config, ip='', mode='train'):
		"""[summary]

		Args:
			config (dictionary): Config data holding hyperparameter information
			ip (str, optional): Data from the input pipeline. Defaults to ''.
			mode (str, optional): Model operation mode (train/test/fine-tune). Defaults to 'train'.
		"""		

		self.config = config
		self.tag_index = self.config['directories']['tag_index']
		self.unit_index = self.config['directories']['unit_index']
		self.MAX_LEN = self.config['train']['max_len']
		if ip:
			self.x_tr = ip.x_tr
			self.x_te = ip.x_te
			self.x_te_orig = ip.x_te
			self.y_tr = ip.y_tr
			self.y_te = ip.y_te
			self.y_tr_unit = ip.y_tr_unit
			self.y_te_unit = ip.y_te_unit
			self.y_tr_rel = ip.y_tr_rel
			self.y_te_rel = ip.y_te_rel
			self.tag2idx = ip.tag2idx
			self.unit2idx = ip.unit2idx
			self.n_tags = ip.n_tag
			self.n_units = ip.n_unit
			self.value = ip.testing_value
		
		try:
			if self.tag2idx in locals():	print("Index exists")
		except:
			if os.path.exists(self.tag_index):
				pickle_out = open(self.tag_index,'rb')
				self.tag2idx = pickle.load(pickle_out)

		try:
			if self.unit2idx in locals():	print("Index exists")
		except:
			if os.path.exists(self.unit_index):
				pickle_out = open(self.unit_index,'rb')
				self.unit2idx = pickle.load(pickle_out)
		
		self.mode = mode
		self.lang = self.config['lang']

		#HyperParam
		self.embedding = self.config['train']['embedding']
		self.BATCH_SIZE = self.config['train']['batch_size']
		self.n_epochs = self.config['train']['epochs']
		self.optimizer = self.config['train']['optimizer']
		self.activation = self.config['train']['activation']
		self.patience = self.config['train']['patience']
		self.dropout = self.config['train']['dropout']
		self.lstm_units = self.config['train']['lstm_units']
		self.dense_units = self.config['train']['dense_unit']
		self.bert_dim = self.config['train']['BERT_dim']
		self.recurrent_dropout = self.config['train']['recurrent_dropout']
		self.metrics = ''
		self.run = str(self.config['train']['run'])
		self.amount = str(self.config['train']['amount'])

		#File directories
		self.saved_model = self.config['directories']['source_model']+"_"+self.run+'.h5'
		#self.saved_model = self.config['directories']['fine_tuned_model']+"_"+self.run+'_'+self.amount+'.h5'
		self.fine_tuned_model = self.config['directories']['fine_tuned_model']+"_"+self.run+'_'+self.amount+'.h5'
		self.saved_score_file = self.config['directories']['saved_score_en']
		self.idx2tag = {i: w for w, i in self.tag2idx.items()}
		self.idx2unit = {i: w for w, i in self.unit2idx.items()}
		self.validation_split = self.config['train']['validation_split']

		self.es = EarlyStopping(monitor="val_loss", mode='min', patience=self.patience, verbose=1)
		self.mc = ModelCheckpoint(self.saved_model, monitor="val_loss", save_best_only=True, mode='min', verbose=1)

	def recognizer_model(self, test_empty=[]):
		"""Core function for the model: Build and train TE Recognizer

		Args:
			test_empty (list, optional): [description]. Defaults to [].
		"""		

		self.test_empty = test_empty
		print("creating embedding...")
		bert_embedder = BertEmbedder(self.MAX_LEN, emb=self.embedding)
		if self.mode!='test':	self.x_tr = np.array([bert_embedder.embed_sentence(i) for i in self.x_tr])
		self.x_te = np.array([bert_embedder.embed_sentence(i) for i in self.x_te])
		print("building model...")
		embedding = Input(shape=(self.MAX_LEN,self.bert_dim), dtype="float32")
		x = Bidirectional(LSTM(units=self.lstm_units, return_sequences=True, recurrent_dropout=self.recurrent_dropout, dropout=self.dropout))(embedding)
		x_rnn = Bidirectional(LSTM(units=self.lstm_units, return_sequences=True, recurrent_dropout=self.recurrent_dropout, dropout=self.dropout))(x)
		x_final = add([x, x_rnn])
		td_model = TimeDistributed(Dense(self.dense_units, activation=self.activation))(x_final)
		crf = CRF(self.n_tags, name='tag_output')
		crf2 = CRF(self.n_units, name='unit_output', learn_mode='marginal') 
		out = crf(td_model)
		out2 = crf2(td_model)
		self.model = Model(embedding, [out, out2])
		self.model.summary()

		sess = tf.Session()
		K.set_session(sess)
		K.tensorflow_backend._get_available_gpus()
		if os.path.exists(self.saved_model):
			self.model = load_model(self.saved_model, custom_objects=get_custom_object())	
			if self.mode=='ft':
				self.method = self.config['fine_tune']['method']
				self.x_tr = self.x_tr[:int(len(self.x_tr)*(int(self.amount)/10))]
				self.y_tr = self.y_tr[:int(len(self.y_tr)*(int(self.amount)/10))]
				self.y_tr_unit = self.y_tr_unit[:int(len(self.y_tr_unit)*(int(self.amount)/10))]
				self.transfer_learning()
		else:
			self.compile()
			print("training model...")
			history = self.train()
			self.model = load_model(self.saved_model, custom_objects=get_custom_object())
			self.compile()

		print("evaluating model...")
		test_pred = self.model.predict([self.x_te], verbose=1, batch_size=len(self.x_te))
		self.evaluate_model(test_pred)

	def transfer_learning(self):	
		"""
		Fine-tune the model based on the hyperparameter
		"""
		self.optimizer = Adam(lr=self.config['fine_tune']['lr'])
		self.mc = ModelCheckpoint(self.fine_tuned_model, monitor="val_loss", save_best_only=True, mode='min', verbose=1)
		print("fine-tuning model")
		if self.method=='chain':
			for i in range(8):
				self.n_epochs = self.config['fine_tune']['epochs']
				if i==2 or i==5:	continue
				for layer in self.model.layers[1:]:	layer.trainable = True
				if i==0:
					for layer in self.model.layers[:-1]:	layer.trainable = False
					self.optimizer = Adam(lr=0.001)
				if i==1:
					for layer in self.model.layers[:-2]:	layer.trainable = False
					self.model.layers[-1].trainable = False
					self.optimizer = Adam(lr=0.001)

				if i>2 and i<7:
					self.optimizer = Adam(lr=self.config['fine_tune']['lr'])
					for idx, layer in enumerate(self.model.layers):
						if idx==i-2:	layer.trainable = True
						else:	layer.trainable = False
				for layer in self.model.layers:
					print(layer, layer.trainable)

				self.compile()
				history = self.train()
				self.model = load_model(self.fine_tuned_model, custom_objects=get_custom_object())
		if self.method=='full':
			self.compile()		
			history = self.train()
			self.model = load_model(self.fine_tuned_model, custom_objects=get_custom_object())
		if self.method=='last':
			for layer in self.model.layers[:-1]:
				layer.trainable = False
			self.compile()		
			history = self.train()
			self.model = load_model(self.fine_tuned_model, custom_objects=get_custom_object())

	def compile(self):
		"""Compile the neural model """		
		losses = {"tag_output": crf_loss, "unit_output":"categorical_crossentropy"}
		self.model.compile(optimizer=self.optimizer, loss=losses, metrics=self.metrics)

	def train(self):
		""""Train the neural model with different hyperparameter"""
		return self.model.fit( [self.x_tr], {"tag_output": self.y_tr, "unit_output":self.y_tr_unit},
			validation_split=self.validation_split, batch_size=self.BATCH_SIZE, epochs=self.n_epochs, verbose=1, callbacks=[self.es, self.mc])

	def evaluate_model(self, test_pred):
		""" Evaluate the predicted output. Calculate and plot results. Write outputs to file.

		Args:
			test_pred (list): Model prediction 
		"""		

		pred_labels = pred2label(test_pred[0], self.idx2tag)
		pred_labels_unit = pred2label(test_pred[1], self.idx2unit)
		pred_labels_unit_second = pred2label_second(test_pred[1], self.idx2unit)
		
		print(self.idx2tag)
		print(self.idx2unit)
		test_labels = pred2label(self.y_te, self.idx2tag)
		test_labels_unit = pred2label(self.y_te_unit, self.idx2unit)
		test_labels, pred_labels, test_labels_unit, pred_labels_unit, pred_labels_unit_second = self.postprocess_data(test_labels, pred_labels, test_labels_unit, pred_labels_unit, pred_labels_unit_second) #, test_rel, pred_rel
		pred_labels = self.add_empty_rule(pred_labels)
		print(len(pred_labels), len(pred_labels_unit))
		
		f = open(self.config['directories']['saved_output']+"_"+self.run+"_"+self.amount+".txt", "a+")
		for i, j, k, a, b, x in zip(self.x_te_orig, test_labels, pred_labels, test_labels_unit, pred_labels_unit, pred_labels_unit_second):
			f.write(i.replace('__PAD__','')+"\n")
			print(len(i), len(j), len(k), len(a), len(b), len(x))
			for idx, (m, n, o, c, d, y) in enumerate(zip(i.split(" "), j, k, a, b, x)):
				if m!='__PAD__':
					if n!='O' or o!='O' or c!='O' or d!='O':
						f.write("%s %s %s %s %s %s \n" % (m, n, o, c, d, y)) 
						print(m, n, o, c, d, y)
			print("---------------")
			f.write("---------------\n")
		f.close()

		print("Strict F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
		unit_score = f1_score(test_labels_unit, pred_labels_unit)
		print("Strict F1-score Unit: {:.1%}".format(unit_score))
		f1_type_str = f1_score(test_labels, pred_labels)*100
		print(classification_report(test_labels, pred_labels))
		#classes = sorted(list(set(get_cleaned_label(util.flatten(test_labels)))), key=len, reverse=True)
		#cm = confusion_matrix( get_cleaned_label(util.flatten(test_labels)),  get_cleaned_label(util.flatten(pred_labels)), classes)
		#plot.plot_confusion_matrix(cm, classes, run=self.run, amount=str(self.amount), normalize=True)
		#plot.plot_classification_report(classification_report(test_labels, pred_labels), run=self.run, amount=str(self.amount))
		
		#print(classification_report(test_labels_unit, pred_labels_unit))
		#classes_unit = sorted(list(set(get_cleaned_label(util.flatten(test_labels_unit)))), key=len, reverse=True)
		#cm = confusion_matrix( get_cleaned_label(util.flatten(test_labels_unit)),  get_cleaned_label(util.flatten(pred_labels_unit)), classes_unit)
		#plot.plot_confusion_matrix(cm, classes_unit, run=self.run, amount=str(self.amount)+"_unit", normalize=True)
		#plot.plot_classification_report(classification_report(test_labels_unit, pred_labels_unit), run=self.run, amount=str(self.amount)+"_unit")

		str_t_fscore, str_r_fscore, f1_type, f1_type_str, f1_unit, f1_str_unit, f1_val, gold_len, system_len, acc_val = eval.tempeval_annotate(self.x_te_orig, test_labels, pred_labels, self.value, pred_labels_unit, pred_labels_unit_second, test_labels_unit, self.lang)
		with open(self.saved_score_file,'a') as fd:
			row = str(str_t_fscore)+","+str(str_r_fscore)+","+ str(f1_type_str)+","+ str(f1_type)+","+ str(f1_str_unit)+","+  str(f1_unit)+","+ str(f1_val)+","+ str(gold_len)+","+ str(system_len)+","+ str(acc_val)+"\n"
			fd.write("%s" % row)
		fd.close()

	def postprocess_data(self, test_labels, pred_labels, test_labels_unit, pred_labels_unit, pred_labels_unit_second):  #, test_rel, pred_rel
		"""Post-process data for empty tags

		Args:
			test_labels (list): gold label 
			pred_labels (list): predicted label
			test_labels_unit (list): gold unit
			pred_labels_unit (list): predicted unit
			pred_labels_unit_second (list): second max predicted unit

		Returns:
			test_labels: added extra labels "__EMPTY__" on the gold labels if empty
			pred_labels: added extra empty labels on the predicted labels
			test_labels_unit: gold unit
			pred_labels_unit: predicted unit
			pred_labels_unit_second: second max predicted unit 
		"""		
		for id, (x, y, z, m, n, nn) in enumerate(zip(self.test_empty, test_labels, pred_labels, test_labels_unit, pred_labels_unit, pred_labels_unit_second)): #, test_rel, pred_rel , q, r
			if x:
				for xx in x:
					y.append('B-'+xx[0])
					z.append('O')
					m.append('O')
					n.append('O')
					nn.append('O')
					self.x_te_orig[id] += ' __EMPTY__'
		return test_labels, pred_labels, test_labels_unit, pred_labels_unit, pred_labels_unit_second

	def postprocess_data_predict(self, te, pred_labels, pred_labels_unit, pred_labels_unit_second):
		"""Data post-processing while predicting, checking for empty tags

		Args:
			te (list): list of sentence
			pred_labels (list): predicted labels add empty if pattern match
			pred_labels_unit (list): predicted unit
			pred_labels_unit_second (list): Second max predicted unit

		Returns:
			te (list): list of sentence
			pred_labels (list): predicted labels add empty if pattern match
			pred_labels_unit (list): predicted unit
			pred_labels_unit_second (list): Second max predicted unit
		"""		

		#Remove PAD
		modified_pred_labels = []
		modified_pred_unit = []
		modified_pred_unit_second = []
		for t, p, p_u, p_u_second in zip(te, pred_labels, pred_labels_unit, pred_labels_unit_second):
			p_in = []
			p_u_in = []
			p_u_in_second = []
			for tt, pp, p_uu, p_uu_second in zip(nltk.word_tokenize(t), p, p_u, p_u_second):
				p_in.append(pp)
				p_u_in.append(p_uu)
				p_u_in_second.append(p_uu_second)
			modified_pred_labels.append(p_in)
			modified_pred_unit.append(p_u_in)
			modified_pred_unit_second.append(p_u_in_second)
		pred_labels = modified_pred_labels
		pred_labels_unit = modified_pred_unit
		pred_labels_unit_second = modified_pred_unit_second
		
		#ADD _EMPTY_
		start_point, end_point = get_tokens(self.lang)
		for idx, (t, p) in enumerate(zip(te, pred_labels)):
			begin_token = False
			end_token = False
			for s in start_point:
				if s in nltk.word_tokenize(t):
					begin_token = True
					break
			for e in end_point:
				if e in nltk.word_tokenize(t):
					end_token = True
					break

			if begin_token and end_token:
				te[idx] += ' __EMPTY__'
				p.append('O')

		return te, pred_labels, pred_labels_unit, pred_labels_unit_second
		
	def add_empty_rule(self, pred_labels):
		"""Add empty rule based on the being and end token

		Args:
			pred_labels (list): predicted labels

		Returns:
			predicted_labels: added empty tags if matched on the predicted labels
		"""		
		"""
		"""
		
		start_point, end_point = get_tokens(self.lang)
		first_tag = ''
		second_tag = ''

		for x, z in zip(self.x_te_orig, pred_labels):
			x_split = x.split(" ")
			print(x_split)
			for idx, xx in enumerate(x_split):
				if xx in start_point and z[idx+1].startswith('B-'):
					first_tag = z[idx+1].split("-")[1]
					for idx2, zz in enumerate(z[idx+1:]):
						if zz=='O' and x_split[idx+idx2+1] in end_point:
							if z[idx+idx2+2].startswith('B-'):	second_tag = z[idx+idx2+2].split("-")[1]
							else:	break
						elif zz=='O' and x_split[idx+idx2+1] not in end_point:	break
					if first_tag and second_tag:
						if first_tag==second_tag:	z[-1] = 'B-DURATION'
					first_tag = ''
					second_tag = ''
				try:
					if z[idx+1].startswith('B-'):
						first_tag = z[idx+1].split("-")[1]
						for idx2, zz in enumerate(z[idx+1:]):
							if zz=='O' and x_split[idx+idx2+1] in 'to':
								if z[idx+idx2+2].startswith('B-'):	second_tag = z[idx+idx2+2].split("-")[1]
								else:	break
							elif zz=='O' and x_split[idx+idx2+1] not in end_point:	break
						if first_tag and second_tag:
							if first_tag==second_tag:	z[-1] = 'B-DURATION'
						first_tag = ''
						second_tag = ''
				except:	pass
		return pred_labels

	def predict(self, te):
		"""Model predict pipeline

		Args:
			te ([list]): list of sentences with TEs

		Returns:
			te: list of sentence with TEs
			timex_list: list of timex3 object after prediction
		"""		

		self.x_te_orig = te
		final_model_de = self.config['predict']['final_model_de']
		final_model_en = self.config['predict']['final_model_en']
		if self.lang=='en':	self.model = load_model(final_model_en, custom_objects=get_custom_object())
		self.compile()
		bert_embedder = BertEmbedder(self.MAX_LEN, emb=self.embedding)
		self.x_te = np.array([bert_embedder.embed_sentence(i) for i in self.x_te_orig])
		test_pred = self.model.predict([self.x_te], verbose=1, batch_size=len(self.x_te))
		pred_labels = pred2label(test_pred[0], self.idx2tag)
		pred_labels_unit = pred2label(test_pred[1], self.idx2unit)
		pred_labels_unit_second = pred2label_second(test_pred[1], self.idx2unit)
		self.x_te_orig, pred_labels, pred_labels_unit, pred_labels_unit_second = self.postprocess_data_predict(self.x_te_orig, pred_labels, pred_labels_unit, pred_labels_unit_second)
		pred_labels = self.add_empty_rule(pred_labels)
		timex_list = eval.pred_wrapper(self.x_te_orig, pred_labels, pred_labels_unit, pred_labels_unit_second, self.lang)
		return te, timex_list

def pred2label(pred, idx2tag):
	"""Convert one-hot encoded predicted score to label

	Args:
		pred ([list): list of predicted score
		idx2tag (dictionary): index to tag dictionary

	Returns:
		out: predicted labeled output
	"""	

	out = []
	for pred_i in pred:
		out_i = []
		for p in pred_i:
			p_i = np.argmax(p)
			out_i.append(idx2tag[p_i])
		out.append(out_i)
	return out

def pred2label_second(pred, idx2tag):
	"""Convert one-hot encoded predicted score to second highest label

	Args:
		pred ([list): list of predicted score
		idx2tag (dictionary): index to tag dictionary

	Returns:
		out: second highest predicted labeled output
	"""	
	out = []
	for pred_i in pred:
		out_i = []
		for p in pred_i:
			i=1
			while not idx2tag[np.argsort(-p)[i]].startswith('B'):
				i+=1
			p_i = np.argsort(-p)[i]
			out_i.append(idx2tag[p_i])
		out.append(out_i)
	return out

def get_custom_object():
	"""Returns additional parameters for model saving

	Returns:
		[dictionary]: dictionary containing Keras specific model information
	"""	

	return {
	"crf_loss": crf_loss, 
	'crf_viterbi_accuracy':crf_viterbi_accuracy, 
	#'hub':hub, 
	#'tf':tf, 
	'CRF':CRF
	}

def get_cleaned_label(l):
	
	"""
	Given a list of label with 'B-','I-' prefix, returns a cleaned label without prefix
	"""
	ll=[]
	for label in l:
		if "-" in label:	ll.append(label.split("-")[1])
		else:	ll.append(label)
	return ll

def get_tokens(lang):
	""" Given the language return start and end point token"""
	
	if lang=='en':
		start_point = ['for', 'from', 'on', 'between']
		end_point = ['through', 'to', 'and', 'until']
		
	return start_point, end_point