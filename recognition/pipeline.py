from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle 
import json
import recognition.util as util
import re
import copy
import csv
from recognition.reader import read_input_tml, read_input_json_pate, read_input_json_snips, read_input_simplified_text, Corpus
import nltk

#=========================================================================
#
#  This file sets up the input pipeline for the temporal tagging task         
#
#=========================================================================


class InputPipeline:	
	"""
	This class is a pipeline setup for reading raw input and processing to standard BIO-labeling format
	from different source of files
	"""	
	def __init__(self, config, training_dataset, testing_dataset):
		"""[summary]

		Args:
			config (dictionary): Config file to hold all the hyperparameters
			training_dataset (list): List of training data folder name
			testing_dataset (list): List of testing data folder name
		"""		
		self.config = config
		self.MAX_LEN = self.config['train']['max_len']
		self.tag_index = self.config['directories']['tag_index']
		self.unit_index = self.config['directories']['unit_index']
		self.simplify = self.config['train']['simplify']
		self.testing_dataset = []
		(self.training_ext, self.training_dataset), = training_dataset.items()
		if testing_dataset:	(self.testing_ext, self.testing_dataset), = testing_dataset.items()
		
		self.unique_tag = set()
		self.unique_unit = set()
		self.tag2idx = None
		self.unit2idx = None
		self.n_tag = 0
		self.n_unit = 0

		#Fetch the raw data from different raw source input
		self.get_data()

		#Prepare only data with TIMEX3 information
		self.x, self.y, self.unit, self.training_length, self.rel_pos = self.prepare_input(self.c_obj_training)
		if self.testing_dataset:
			self.x, self.y, self.unit, self.rel_pos = util.io_random_shuffle( list(zip(self.x, self.y, self.unit, self.rel_pos))) #Shuffle input/output
			self.x2, self.y2, self.unit2, self.testing_length, self.rel_pos2 = self.prepare_input(self.c_obj_testing)

		#Create index for type and label
		self.tag2idx, self.n_tag = util.create_index(self.unique_tag, self.tag_index)
		self.unit2idx, self.n_unit = util.create_index(self.unique_unit, self.unit_index)

		#Process training/testing set for feeding into the model
		if not self.testing_dataset:
			self.x_tr, self.x_te, self.y_tr, self.y_te, self.y_tr_unit, self.y_te_unit, self.y_tr_rel, self.y_te_rel, self.training_value, self.testing_value = self.prcoess_input(self.x, self.y, self.unit, self.rel_pos, same_data=True)
		else:
			self.x_tr, self.y_tr, self.y_tr_unit, self.y_tr_rel = self.prcoess_input(self.x, self.y, self.unit, self.rel_pos, same_data=False)
			self.x_te, self.y_te, self.y_te_unit, self.y_te_rel =  self.prcoess_input(self.x2, self.y2, self.unit2, self.rel_pos2, same_data=False)

	def get_data(self):
		"""
		Get training and/or testing data from scratch 
		"""
		self.c_obj_training, self.training_value, self.train_timex, self.train_empty = self.read_input(self.training_dataset, 'train', self.training_ext)
		self.test_empty = []
		if self.testing_dataset:
			self.c_obj_testing, self.testing_value, self.test_timex, self.test_empty = self.read_input(self.testing_dataset, 'test', self.testing_ext)

	def read_input(self, dataset, name, extension='.tml'):
		"""Read different input and structure them into the Corpus class object

		Args:
			dataset (list): containing different folder from different dataset
			name (str): A flag whether it is training or testing data
			extension (str, optional): An extension separates different formal of inputs. Defaults to '.tml'.

		Returns:
			c_obj_list: a list of corpus object
			input_value: a list of timex value information
			timex_l: a list of timex information
		"""		
		print(extension)
		print("In input read...")
		#c_obj_list = []
		#timex_l = []
		empty_list = []
		input_value = []
		unit_l = []
		if extension=='.tml':
			c_obj_list, timex_l, unit_l, input_value = read_input_tml(dataset, self.MAX_LEN)
			if self.simplify:
				input_modified_dir, timex_dir = self.config['directories']['input_modified_for_simplify'], self.config['directories']['timex_for_simplify'] 
				util.preprocess_simplify(c_obj_list, input_modified_dir, timex_dir) #Simplify input text
		elif extension=='.json_snips':	
			c_obj_list, timex_l, unit_l = read_input_json_snips(dataset, self.MAX_LEN)
		elif extension=='.pkl':
			input_simplified_dir = self.config['directories']['input_simplified']
			timex_for_simplify_dir = self.config['directories']['timex_for_simplify']
			c_obj_list, timex_l, unit_l, input_value = read_input_simplified_text(dataset, input_simplified_dir, timex_for_simplify_dir, self.MAX_LEN)
		elif extension=='.json_pate':
			c_obj_list, timex_l, unit_l, input_value, empty_list = read_input_json_pate(dataset, self.MAX_LEN)
		#util.write_to_csv(timex_l) #Write timexes into a csv
		util.count_label(timex_l)
		print(unit_l)
		util.sentence_stats(c_obj_list)
		util.count_stats(c_obj_list)
		print("Length of the TIMEX", len(timex_l))
		print("Length of the input value", len(input_value))
		print("Input read...")
		return c_obj_list, input_value, timex_l, empty_list

	def prepare_input(self, corpus_ob):
		"""This function takes list of corpus objects and separates input, output and features into different lists

		Args:
			corpus_ob (Corpus object): list of object containing sentence and timex

		Returns:
			input_X: list of input sentence
			output_label: list of timex label
			output_unit: list of timex unit
			length: list of sentence length
			relation_position: list of anchor head position
		"""		
		print("In prepare input ")
		input_X = []
		output_label = []
		output_unit = []
		relation_position = []
		length = []
		for c_obj in corpus_ob:
			for sentences in c_obj.tagged_text:
				w_in_sentence = []
				l_in_sentence = []
				u_in_sentence = []
				r_in_sentence = []		
				if not all(s[1]=='O' for s in sentences):
					for word, type, unit, rel_pos in sentences:
						w_in_sentence.append(word)
						l_in_sentence.append(type)
						u_in_sentence.append(unit)
						r_in_sentence.append(rel_pos)
						self.unique_tag.add(type)
						self.unique_unit.add(unit)
					input_X.append(w_in_sentence)
					output_label.append(l_in_sentence)
					output_unit.append(u_in_sentence)
					relation_position.append(r_in_sentence)
					length.append(len(w_in_sentence))
		print("Input prepared...")
		return input_X, output_label, output_unit, length, relation_position

	def prcoess_input(self, input_X, output_Y, output_unit, rel_pos, same_data=False):
		"""This function pads input with max length, split train and test set if from the same dataset
		and categorizes labels to one-hot encoding

		Args:
			input_X (list): list of input sentence
			output_Y (list): list of timex3 type
			output_unit (unit): list of input unit
			rel_pos (list): list of relations heading
			same_data (bool, optional): If training/testing data is from same sample. Defaults to False.

		Returns:
		when same_data is True
			X_tr: list of padded training input
			X_te: list of padded testing input
			y_tr: list of one-hot encoded training labels
			y_te: list of one-hot encoded testing labels
			y_unit_tr: list of one-hot encoded training unit
			y_unit_te: list of one-hot encoded testing unit
			y_rel_tr: list of one-hot encoded training relation
			y_rel_te: list of one-hot encoded testing relation
			val_tr: list of training values
			val_te: list of testing values

		when same_data is False
			newInputX: list of padded input
			output_Y: list of one-hot encoded labels
			output_unit: list of one-hot encoded units
			rel: list of one-hot encoded relation
		"""		

		print("In process....")
		new_X = []
		new_Y = []
		new_Z = []
		new_W = []
		for seq, out_seq, out_unit, out_rel in zip(input_X, output_Y, output_unit, rel_pos):
			new_seq = []
			new_out_seq = []
			new_out_unit = []
			new_out_rel  = []
			for i in range(self.MAX_LEN):
				try:
					new_seq.append(seq[i])
					new_out_seq.append(self.tag2idx[out_seq[i]])
					new_out_unit.append(self.unit2idx[out_unit[i]])
					new_out_rel.append(out_rel[i])
				except IndexError:
					new_seq.append("__PAD__")
					new_out_seq.append(self.tag2idx["O"])
					new_out_unit.append(self.unit2idx["O"])
					new_out_rel.append(self.MAX_LEN)
			new_X.append(new_seq)
			new_Y.append(new_out_seq)
			new_Z.append(new_out_unit)
			new_W.append(new_out_rel)

		newInputX = []
		for x in new_X:
			str = ' '.join(x)
			newInputX.append(str)

		print("Input is processing...")
		if same_data:
			new_X, new_Y, new_Z, new_W, self.training_value = util.io_random_shuffle( list(zip(newInputX, new_Y, new_Z, new_W, self.training_value))) #Shuffle input/output
			X_tr, X_te, y_tr, y_te, y_unit_tr, y_unit_te, y_rel_tr, y_rel_te, val_tr, val_te = train_test_split(new_X, new_Y, new_Z, new_W, self.training_value, test_size=0.1, random_state=2018)
			
			y_tr = util.categorical(y_tr, self.n_tag)
			y_te = util.categorical(y_te, self.n_tag)

			y_unit_tr = util.categorical(y_unit_tr, self.n_unit)
			y_unit_te = util.categorical(y_unit_te, self.n_unit)

			y_rel_tr = util.categorical(y_rel_tr, self.MAX_LEN+1)
			y_rel_te = util.categorical(y_rel_te, self.MAX_LEN+1)

			return X_tr, X_te, y_tr, y_te, y_unit_tr, y_unit_te, y_rel_tr, y_rel_te, val_tr, val_te
		else:
			output_Y = util.categorical(new_Y, self.n_tag)
			output_unit = util.categorical(new_Z, self.n_unit)
			rel = util.categorical(new_W, self.MAX_LEN+1)

			return newInputX, output_Y, output_unit, rel
