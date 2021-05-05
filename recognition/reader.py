import recognition.util as util
import nltk
from text_to_num import alpha2digit
import json
from lxml import etree
import os
import pickle

#=======================================================================================
#
#  This file reads data from different sources and store them in a Corpus data structure           
#
#=======================================================================================


class Corpus:
	"""
	A class holding the input data structure
	"""
	def __init__(self, doc_id, dct, tagged_text, tagged_value):
		"""
		Args:
			doc_id (str): .tml files document Id
			dct (str): Document creation time
			tagged_text ([list): Tagged text with types and units
			tagged_value (list): Input values
		"""		
		self.doc_id = doc_id
		self.dct = dct
		self.tagged_text = tagged_text
		self.tagged_value = tagged_value

def read_input_tml(dataset, MAX_LEN):
	"""	Read .tml file data and returns corpus object, timex info

	Args:
		c_obj_list : List to hold Corpus object
		timex_l : Timex List
		dataset : Dataset directories
		MAX_LEN : Maximum length for the model

	Returns:
		c_obj_list : List to hold Corpus object
		timex_l : Timex List
		norma_unit_l: Unit list
		input_value: Value list
	"""	
	c_obj_list = []
	timex_l = []
	file_count=0
	input_value = []
	norma_unit_l = {}
	for d in dataset:
		path = 'data/'+d+'/'
		for file in os.listdir(path):
			file_count+=1 
			print("File count",file_count)
			tree = etree.parse(path+file)
			timex=[]
			root = tree.getroot()
			for child in root:
				if child.tag=='DOCID':	doc_id = child.text
				if child.tag=='DCT':	dct = child[0].get('value')
				if child.tag=='TEXT':
					results = util.parser(child, {'TIMEX3'})
					for text, type, value, tid, anchor in results:
						if text!=None:	timex.append((text, type, value, tid, anchor))
					text = ' '.join(child.itertext())

			sentence_tagged = []
			fulltext = []
			for sentence in nltk.sent_tokenize(text):
				for sent_new_line in sentence.split("\n"):
					sent = []
					full_sent = ''
					for token in nltk.word_tokenize(sent_new_line):
						sent.append((token, 'O', 'O', MAX_LEN))
						full_sent += token + " "
					fulltext.append(full_sent)
					sentence_tagged.append(sent)

			timex_l.append(timex)
			tagged_value = [[] for x in range(len(sentence_tagged))]
			textIdx = 0
			ind_tuple_check = set()
			next_tuple = False
			temp_timex = timex
			while textIdx < len(fulltext):
				if len(timex)==0:	break
				chunk, type, value, tid, anchor = timex[0]
				normalized_unit = util.get_normalized_unit(value)

				ind_tuple = util.find_sub_list(nltk.word_tokenize(chunk), fulltext[textIdx].split(" "))
				if ind_tuple:
					if len(ind_tuple)==1:	ind_tuple = ind_tuple[0]
					else:
						if not next_tuple and not ((ind_tuple[0][0],textIdx,) in [(ii[0], ii[2],) for ii in ind_tuple_check ] ):
							ind_tuple = ind_tuple[0]
							next_tuple = True
						else:	ind_tuple = ind_tuple[1]
					
				counter = 0
				if ind_tuple and (ind_tuple+(textIdx,) not in ind_tuple_check) and not ((ind_tuple[0],textIdx,) in [(ii[0], ii[2],) for ii in ind_tuple_check]):
					ind_tuple_check.add(ind_tuple+(textIdx,))
					tagged_value[textIdx].append(value)
					if normalized_unit in norma_unit_l:	norma_unit_l[normalized_unit] += 1
					else: norma_unit_l[normalized_unit] = 1
					for i in range(ind_tuple[0], ind_tuple[1]+1):
						if counter == 0:
							sentence_tagged[textIdx][i] = list(sentence_tagged[textIdx][i])
							sentence_tagged[textIdx][i][1] = 'B-'+type
							sentence_tagged[textIdx][i][2] = 'B-'+normalized_unit
							try:
								if anchor!= None:
									for t in temp_timex:
										if t[4]==anchor:
											position = util.find_sub_list(nltk.word_tokenize(t[0]), fulltext[textIdx].split(" "))[0][0]
											sentence_tagged[textIdx][i][3] = MAX_LEN if position >  MAX_LEN else position
							except:	print("Error in fetching anchorTimeID")
							sentence_tagged[textIdx][i] = tuple(sentence_tagged[textIdx][i])
						else:
							sentence_tagged[textIdx][i] = list(sentence_tagged[textIdx][i])
							sentence_tagged[textIdx][i][1] = 'I-'+type
							sentence_tagged[textIdx][i][2] = 'I-'+normalized_unit
							sentence_tagged[textIdx][i] = tuple(sentence_tagged[textIdx][i])
						counter+=1
					timex = timex[1:]
					textIdx-=1
				textIdx+=1
			doc_id = ''

			tagged_value = [x for x in tagged_value if x] #comment out when news
			c_obj = Corpus(doc_id, dct, sentence_tagged, tagged_value)
			c_obj_list.append(c_obj)
			for value in c_obj.tagged_value:
				input_value.append((c_obj.dct, value))

	return c_obj_list, timex_l, norma_unit_l, input_value

def read_input_json_pate(dataset, MAX_LEN):
	"""Read json format timex data and returns corpus object and timex info

	Args:
		c_obj_list : List to hold Corpus object
		timex_l : Timex List
		dataset : Dataset directories
		MAX_LEN : Maximum length for the model

	Returns:
		c_obj_list : List to hold Corpus object
		timex_l : Timex List
		norma_unit_l: Unit list
		input_value: Value list
		empty_list: empty tag list
	"""	
	c_obj_list = []
	timex_l = []
	input_value = []
	ref_date_dict = '2019-11-18' #TODO: add in config
	timex_expression = []
	timex_empty = []
	empty_list = []
	empty = False
	entity_mark = True
	norma_unit_l = {}
	for d in dataset:
		path = 'data/'+d+'/'
		for file in os.listdir(path):
			with open(path+file, 'r') as f:
				tree = json.load(f)
				sentence_tagged = []
				for item in tree:
					tagged_value = []
					empty_values = []
					timex = []
					text = item["text"]
					temp_sent_tagged = []
					fulltext = []

					for sentence in nltk.sent_tokenize(text):
						for sent_new_line in sentence.split("\n"):
							sent = []
							full_sent = ''
							for token in nltk.word_tokenize(sent_new_line):
								sent.append((token, 'O', 'O', MAX_LEN))
								full_sent += token + " "
							fulltext.append(full_sent)
							temp_sent_tagged.append(sent)

					#HANDLING EMPTY TAGS
					for entity in item["entities"]:
						if entity['entity']=='datetime':
							if "TIMEX3" in entity:
								for tx3 in entity["TIMEX3"]:
									if tx3['type'] != '':
										timex_expression.append(tx3["expression"])
										timex.append((tx3["expression"], tx3["type"], tx3['value'], util.get_normalized_unit(tx3["value"]), tx3['tid'], tx3['anchorTimeID']))
										if tx3['expression']=='':	empty_values.append(tx3['value'])
										else:	tagged_value.append(tx3['value'])
										if tx3["expression"]=='':
											empty = True
											timex_empty.append((tx3["type"], tx3['value'], tx3['tid']))
									else:	entity_mark = False
							else:	entity_mark = False
						else:	entity_mark = False
					if entity_mark:
						if empty:
							empty_list.append(timex_empty)
							timex_empty = []
							empty = False
						else:	empty_list.append([])
					entity_mark = True #MAYBE check this line
					if timex:	timex_l.append(timex)

					textIdx = 0
					ind_tuple_check = set()
					next_tuple = False
					temp_timex = timex
					while textIdx < len(fulltext):
						if len(timex)==0:	break
						chunk, type, value, unit, tid, anchor = timex[0]
						if chunk=='':
							timex = timex[1:]
							continue
						#TODO: ADD a while for second TE with same token
						ind_tuple = util.find_sub_list(nltk.word_tokenize(chunk), fulltext[textIdx].split(" "))
						if unit in norma_unit_l:	norma_unit_l[unit] += 1
						else: norma_unit_l[unit] = 1
						if ind_tuple:
							if len(ind_tuple)==1:	ind_tuple = ind_tuple[0]
							else:
								if not next_tuple and not ((ind_tuple[0][0],textIdx,) in [(ii[0], ii[2],) for ii in ind_tuple_check ] ):
									ind_tuple = ind_tuple[0]
									next_tuple = True
								else:	ind_tuple = ind_tuple[1]
							
						counter = 0
						if ind_tuple and (ind_tuple+(textIdx,) not in ind_tuple_check) and not ((ind_tuple[0],textIdx,) in [(ii[0], ii[2],) for ii in ind_tuple_check ] ):
							ind_tuple_check.add(ind_tuple+(textIdx,))
							for i in range(ind_tuple[0], ind_tuple[1]+1):
								if counter == 0:
									temp_sent_tagged[textIdx][i] = list(temp_sent_tagged[textIdx][i])
									temp_sent_tagged[textIdx][i][1] = 'B-'+type
									temp_sent_tagged[textIdx][i][2] = 'B-'+unit
									if anchor!= None:
										for t in temp_timex:
											if t[4]==anchor:	pass #TODO: ADD anchor for PATE
									temp_sent_tagged[textIdx][i] = tuple(temp_sent_tagged[textIdx][i])
								else:
									temp_sent_tagged[textIdx][i] = list(temp_sent_tagged[textIdx][i])
									temp_sent_tagged[textIdx][i][1] = 'I-'+type
									temp_sent_tagged[textIdx][i][2] = 'I-'+unit
									temp_sent_tagged[textIdx][i] = tuple(temp_sent_tagged[textIdx][i])
								counter+=1
							timex = timex[1:]
							textIdx-=1
						textIdx+=1
					if empty_values:	tagged_value.extend(empty_values)
					if tagged_value:	input_value.extend([(ref_date_dict, tagged_value)])
						
					sentence_tagged.extend(temp_sent_tagged)
				input_value = [x for x in input_value if x]
				c_obj = Corpus('', '', sentence_tagged, input_value)
				c_obj_list.append(c_obj)

	return c_obj_list, timex_l, norma_unit_l, input_value, empty_list

def read_input_json_snips(dataset, MAX_LEN):
	"""Read Snips json format timex data and returns corpus object and timex info

	Args:
		c_obj_list : List to hold Corpus object
		timex_l : Timex List
		dataset : Dataset directories
		MAX_LEN : Maximum length for the model

	Returns:
		c_obj_list : List to hold Corpus object
		timex_l : Timex List
		norma_unit_l: Unit list
	"""	
	#TODO: mapping relation to anchor
	c_obj_list = []
	timex_l = []
	timex_expression = []
	norma_unit_l = {}
	for d in dataset:
		path = 'data/'+d+'/'
		for file in os.listdir(path):
			with open(path+file, 'r') as f:
				tree = json.load(f)
				for item in tree:
					sent = []
					item = item['data']
					for subitem in item:
						if 'TIMEX3' in subitem:
							text = alpha2digit(subitem['text'], 'en')
							text = util.snips_preprocess(text)
							for titem in subitem['TIMEX3']:
								titem['expression'] = alpha2digit(titem['expression'], 'en')
							timex = [(t_item['expression'], t_item['type'], t_item['value'],  util.get_normalized_unit(t_item["value"]), t_item['tid'], t_item['anchorTimeID'] ) for t_item in subitem['TIMEX3'] if ( t_item['type'])] #t_item['expression'] and
							timex_plot = [(t_item['expression'], t_item['type'], t_item['value'], t_item['tid']) for t_item in subitem['TIMEX3']]

							if not timex:	continue
							timex_l.append(timex_plot)
							timex_expression.append(timex_plot[0][0])
							chunk, type, value, unit, tid, anchor = timex[0]
							if unit in norma_unit_l:	norma_unit_l[unit] += 1
							else: norma_unit_l[unit] = 1
							splitted_chunk = util.snips_preprocess(chunk)
							prefix = 'B-'
							for t in text:
								if splitted_chunk:
									if t==splitted_chunk[0]:	
										sent.append((t, prefix+type, prefix+unit, MAX_LEN))
										if splitted_chunk:
											splitted_chunk = splitted_chunk[1:]
											prefix = 'I-'
								else:
									sent.append((t, 'O', 'O', MAX_LEN))
									timex, splitted_chunk, prefix, type, unit, anchor = snips_chunk(timex)
						else:
							sent.extend([(token, 'O', 'O', MAX_LEN) for token in nltk.word_tokenize(subitem['text']) ])
					c_obj = Corpus('', '', [sent], [])
					c_obj_list.append(c_obj)

	return c_obj_list, timex_l, norma_unit_l

def read_input_simplified_text(dataset, input_simplified, timex_for_simplify, MAX_LEN='30'):
	"""Read json format timex data and returns corpus object and timex info

	Args:
		c_obj_list : List to hold Corpus object
		timex_l : Timex List
		dataset : Dataset directories
		MAX_LEN : Maximum length for the model

	Returns:
		c_obj_list : List to hold Corpus object
		timex_l : Timex List
		norma_unit_l: Unit list
		input_value: Value list
	"""	
	c_obj_list = []
	timex_l = []
	pickle_out = open(input_simplified, 'rb')
	simplified = pickle.load(pickle_out)

	pickle_out = open(timex_for_simplify, 'rb')
	timexes = pickle.load(pickle_out)
	for t in timexes:	
		timex_l.append([(t[0][0], t[0][1][0])])
	for idx, sentences  in enumerate(simplified):
		tagged = []
		for sentence in sentences:
			tuples = []
			word = sentence.split(" ")
			for w in word:
				if not w.startswith('XXXXX'):	tuples.append((w,'O', 'O', MAX_LEN))
				else:
					w = timexes[idx][int(w[5])]
					splitted_t=w[0].split(" ")
					for i in range(len(splitted_t)):
						if i==0:	tuples.append((splitted_t[i], w[1][0], w[1][1], w[1][2] if w[1][2] <MAX_LEN+1 else MAX_LEN))
						else:	tuples.append((splitted_t[i],'I-'+w[1][0].split("-")[1], 'I-'+w[1][1].split("-")[1], MAX_LEN))
			tagged.append(tuples)
		c_obj = Corpus('', '', tagged, [])
		c_obj_list.append(c_obj)
	return c_obj_list, timex_l, {}, []

def snips_chunk(timex):
	"""Split chunk and return respective timex chunk and type

	Args:
		timex (list): list of tuples containing timex information

	Returns:
		timex: list of tuple Remaining timex omitting the first item
		spliited_chunk: list of splitted string
		prefix: labels of beginning identifier
		type: type of the label
		unit: unit of the label
		anchor: anchorid of the label
	"""
	timex = timex[1:]
	chunk, type, value, unit, tid, anchor = timex[0]
	chunk = chunk.replace(',','')
	splitted_chunk = chunk.split(" ")
	return timex, splitted_chunk, 'B-', type, unit, anchor
