import nltk
from numpy import cumsum
import numpy as np
import time
import pandas as pd
import pprint
import pickle
import csv
import random
import re
import yaml
import os
from keras.utils import to_categorical

#=========================================================================
#
#  This file defines utility functions for our recognizer           
#
#=========================================================================


def parser(data, tags):
	"""
	Parse specific XML element and return their tags (e.g., type, value)
	"""
	for node in data:
		if node.tag in tags:
			yield node.text, node.get('type'), node.get('value'), node.get('tid'), node.get('anchorTimeID')

def count_label(data):
	"""
	Count the total label/class in the data
	"""
	label = {}
	timex_freq = {}
	total = 0
	for d in data:
		for timex in d:
			total += 1
			timex_processed = (" ".join(timex[0].split())).split(" ")
			len_freq = len(timex_processed)
			if timex[1] not in label:	label[timex[1]] = 1
			else:	label[timex[1]] += 1
			if len_freq not in timex_freq:	timex_freq[len_freq] = 1
			else:	timex_freq[len_freq] += 1

	print("TIMEX_len", timex_freq)
	print("Label", label)
	print("Total label", total)

def count_stats(tagged_obj):
	"""
	Count the vocab and token statistics
	"""
	unique_word = set()
	token = []
	for obj in tagged_obj:
		for sentence in obj.tagged_text:
			for word, type, unit, rel in sentence:
				unique_word.add(word.lower())
				token.append(word)
	print("Vocab & Token ",len(unique_word), len(token))

def sentence_stats(tagged_obj):
	"""
	Get sentence distribution stats
	"""
	max_sent_length = -5
	max_sentence = ''
	total = 0
	tagged_sent = 0
	for obj in tagged_obj:
		for sentences in obj.tagged_text:
			total += 1
			if not all(s[1]=='O' for s in sentences):
				if len(sentences) > max_sent_length:
					max_sent_length = len(sentences)
					max_sentence = sentences
				tagged_sent+=1

	print("Largest sentence", " ".join([x[0] for x in max_sentence])) #TODO: check if this works
	print("Total sentence", total)
	print("Total tagged sentence", tagged_sent)
	print("Biggest sentence length", max_sent_length)
	
def find_sub_list(sl,l):
	"""
	Given a text and keyword, find the matching substring
	"""
	results=[]
	sll=len(sl)
	for ind in (i for i,e in enumerate(l) if e==sl[0]):
		if l[ind:ind+sll]==sl:
			results.append((ind,ind+sll-1))
			#TODO: fix duplicacy issue?
	return results
	
def cumulative_calculate(timex_freq):
	"""
	Returns cumulative percentage given a dictionary of items and their occurence
	"""
	keys = []
	values = []
	for k, v in timex_freq.items():
		keys.append(k)
		values.append(v)
	total = sum(values)

	# calculate cumsum and zip with keys into new dict
	d_cump = dict(zip(keys, (100*subtotal/total for subtotal in cumsum(values))))
	return d_cump

def get_percentage(dict):
	"""
	Returns percentages of dictionary values
	"""
	return [(x/sum(dict.values()))*100 for x in list(dict.values())]

def preprocess_simplify(c_obj, input_modified_dir, timex_dir):
	"""
	Renames TEs as XXXXXNUM format for easy simplification
	e.g., Let's meet on Thursday at 7 -> Let's meet on XXXXX1 at XXXXX2
	"""
	input_modified = []
	timexes = []
	tagged_value = []
	for obj in c_obj:
		tagged_value.append(obj.tagged_value)
		for sentences in obj.tagged_text:
			sent = []
			labeled_word_with_tags = []
			if not all(s[2]=='O' for s in sentences):
				labeled_word = []
				tags = []
				num = 0
				for word, type, unit, rel in sentences:
					if type=='O':
						sent.append(word)
						if labeled_word:
							labeled_word = " ".join(labeled_word)
							labeled_word_with_tags.append((labeled_word, tags))
							labeled_word = []
							tags = []
					elif type.startswith('B-'):
						if labeled_word:
							labeled_word = " ".join(labeled_word)
							labeled_word_with_tags.append((labeled_word, tags))
							labeled_word = []
							tags = []
						sent.append('XXXXX'+str(num))
						num += 1
						labeled_word.append(word)
						tags.append(type)
						tags.append(unit)
						tags.append(rel)
					else:
						labeled_word.append(word)
				if labeled_word:
					labeled_word = " ".join(labeled_word)
					labeled_word_with_tags.append((labeled_word, tags))
				sent = " ".join(sent)	
				input_modified.append(sent)
				timexes.append(labeled_word_with_tags)
	
	pickle_out = open(input_modified_dir,'wb')
	pickle.dump(input_modified, pickle_out)
	pickle_out = open(timex_dir,'wb')
	pickle.dump(timexes, pickle_out)

def snips_preprocess(text):
	"""
	Preprocess for snips data
	"""
	return text.strip().replace(',','').split(" ")

def flatten(l):
	"""
	Given a nested list, return a flattend list
	"""
	return [item for sublist in l for item in sublist]

def write_to_csv(timex):
	"""
	Write timex list to a csv for generating samples
	"""
	timex = flatten(timex)
	print("In csv writer")
	with open('large.csv','w+') as f1:
	    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
	    for l in timex:
	        row = [l[0], l[1], l[2], l[3], l[4], l[5]] #TODO: make dynamic
	        writer.writerow(row)

def io_random_shuffle(combined):
	"""
	Given n lists, combines and shuffle containing same index in n lists.
	"""
	random.shuffle(combined)
	return tuple([list(elem) for elem in zip(*combined)])

def average_results():
	"""
	Average the results for multiple runs of the experiments
	"""
	path = ['output/FT3_Simplified_Pate_Chain/', 'output/FT3_Original_Pate_Chain/', 'output/FT2_Simplified_Snips_Chain/', 'output/FT3_Snips_+_PATE/']
	#path = ['output/FT3_test/']
	columns = ['Strict Extent', 'Relaxed Extent', 'Strict Type', 'Relaxed Type', 'Strict Unit', 'Relaxed Unit', 'F1 Val', 'Gold', 'Sys', 'Val Acc']
	
	for p in path:
		df = pd.read_csv(p+"results.csv", sep=",", header=None)

		df2 = df.groupby(np.arange(len(df))//3).mean()
		df2.columns = columns
		df = df2
		df['index'] = df.index*10
		del df['Gold']
		del df['Sys']
		del df['Strict Type']
		del df['Strict Unit']
		del df['Val Acc']
		df.to_csv(p+'results_averaged.csv', index=False)

def create_index(unique_list, path):
	"""
	Given the tag list, return the tag index and length of the index
	"""
	if os.path.exists(path):
		pickle_out = open(path,'rb')
		index = pickle.load(pickle_out)
	else:	
		pickle_out = open(path, 'wb')
		index = {t: i for i, t in enumerate(unique_list)}
		pickle.dump(index, pickle_out)
	index_length = len(index)
	return index, index_length

def load_config(config_name):
	"""
	Load and return the config yaml file
	"""
	with open(os.path.join(config_name)) as file:
		config = yaml.safe_load(file)
	return config

def categorical(inputs, class_len):
	"""
	Given inputs and class length, return to categorical
	"""
	return np.array([to_categorical(i, num_classes=class_len) for i in inputs])

def get_normalized_unit(v):
	"""
	Given normalized value, return the unit
	"""
	if re.search(r'[0-9X]{4}-W[0-9X]+[-WE]*$|^P[0-9X]+W$', v):
		return 'WEEK'
	elif re.search(r'[0-9X]{4}-[0-9X]{2}$|^P[0-9X]+M$', v):
		return 'MONTH'
	elif re.search(r'[0-9X]{4}-[0-9X]{2}-[0-9X]{2}$|^P[0-9X]+D$', v):
		return 'DAY'
	elif re.search(r'[0-9X]{4}$|^P[0-9X]+Y$', v):
		return 'YEAR'
	elif re.search(r'[0-9X]{4}-[0-9X]{2}-[0-9X]{2}T|^P[0-9XT]+H$|[0-9X]{4}-W[0-9X]+-[0-9X]+T', v):
		return 'HOUR'
	elif re.search(r'[0-9X]{4}-[Q|H][0-9X]+$|^P[0-9X]+Q$', v):
		return 'QUARTER'
	elif v.endswith('_REF'):
		return 'REF'
	elif re.search(r'[0-9X]{4}-[A-Z]{2}$', v):
		return 'SEASON'
	elif re.search(r'^PT[0-9XH]+M$', v):
		return 'MINUTE'
	elif re.search(r'^P[0-9XTHM]+S$', v):
		return 'SECOND'
	else:
		return 'UNK' #or Other