import re
#from normalization.normalizer import norma
#from normalization.normalizer import norma_predict
import nltk

#==================================================================================
#
#  This file contains data post-processing and evaluation from TempEval-3 Challenge
#
#==================================================================================

class Timex():
	"""A TIMEX3 class that holds different attribute for a TE
	"""	
	def __init__(self, tid, text, type, offset, unit="", unit_second="", value="", beginpoint="", endpoint=""):
		"""
		Args:
			tid (str): Id of the TE
			text (str):  A time expression
			type (str): TE type
			offset (tuple): TE's beginning and end position in the sentence
			unit (str, optional): Unit of the TE. Defaults to "".
			unit_second (str, optional): Second probable unit of the TE. Defaults to "".
			value (str, optional): TE's normalized value. Defaults to "".
			beginpoint (str, optional): If beginpoint exist, the tid. Defaults to "".
			endpoint (str, optional): If endpoint exist, the tid. Defaults to "".
		"""		
		self.tid = tid
		self.text = text
		self.type = type
		self.offset = offset
		self.unit = unit
		self.unit_second = unit_second
		self.value = value
		self.beginPoint = beginpoint
		self.endPoint = endpoint


def tempeval_annotate(sentence, test, pred, value, unit, unit_second, test_unit, lang):
	"""Annotate the model prediction to TIMEX3

	Args:
		sentence (str): Original input sentence
		test (list): gold type
		pred (list): predicted type
		value (list): timex value
		unit (list): predicted unit
		unit_second (list): predicted second unit
		test_unit (list): gold unit
		lang (str): language of the model

	Returns:

		str_t_fscore: Strict extent f1
		str_r_fscore: Relaxed extent f1
		f1_type: Relaxed type f1
		f1_str_type: Strict type f1
		f1_unit: Relaxed unit f1
		f1_str_unit: Strict unit f1
		f1_val: Value f1
		gold_len: length of the gold entity
		system_len: length of the predicted entity
		acc_val: Value accuracy
	"""	

	test_list = []
	pred_list = []
	for (s, t, p, u, uu, t_u) in  zip(sentence, test, pred, unit, unit_second, test_unit):
		start_index_gold = -1
		end_index_gold = -1
		start_index_system = -1
		end_index_system = -1
		count_gold = 1
		count_system = 1
		gold_entity = {}
		system_entity = {}
		s = s.split(" ") #TODO: change to nltk?

		for i, (s_ind, t_ind, p_ind, u_ind, uu_ind, tu_ind) in enumerate(zip(s, t, p, u, uu, t_u)):
			gold_id = 't'+str(count_gold)
			system_id = 't'+str(count_system)

			if t_ind.startswith('B-'):
				if start_index_gold!=-1:
					count_gold+= 1
					gold_id = 't'+str(count_gold)
				start_index_gold = i
				end_index_gold = i
				gold_type = t_ind.split("-")[1]
				try:	gold_unit = tu_ind.split("-")[1]
				except: gold_unit = 'O'
				gold_entity[gold_id] = Timex(gold_id, " ".join(s[start_index_gold:end_index_gold+1]) , gold_type, (start_index_gold, end_index_gold), gold_unit)
				system_entity[gold_id] = Timex(gold_id, "" , "", "", "")
				if not p_ind.startswith('I-'):
					count_system = count_gold
					system_id = 't'+str(count_system)

			if t_ind.startswith('I-'):
				end_index_gold = i
				gold_entity[gold_id] = Timex(gold_id, " ".join(s[start_index_gold:end_index_gold+1]) , gold_type, (start_index_gold, end_index_gold), gold_unit)
			
			if p_ind.startswith('B-'):
				start_index_system = i
				end_index_system = i
				system_type = p_ind.split("-")[1]
				try:
					system_unit = u_ind.split("-")[1]
					system_unit_second = uu_ind.split("-")[1]
				except:
					#setting most probable unit in each type
					system_unit = 'O'
					if len(uu_ind.split("-")) > 1:	system_unit_second = uu_ind.split("-")[1]
					else: system_unit_second = 'O'
				system_entity[system_id] = Timex(system_id, " ".join(s[start_index_system:end_index_system+1]) , system_type, (start_index_system, end_index_system), system_unit, system_unit_second)
			
			if p_ind.startswith('I'):
				end_index_system = i
				system_entity[system_id] = Timex(system_id, " ".join(s[start_index_system:end_index_system+1]) , system_type, (start_index_system, end_index_system), system_unit, system_unit_second)
			
			if (t_ind=='O' and start_index_gold!=-1 and end_index_gold!=-1):
				gold_entity[gold_id] = Timex(gold_id, " ".join(s[start_index_gold:end_index_gold+1]) , gold_type, (start_index_gold, end_index_gold), gold_unit)
				start_index_gold=-1
				end_index_gold=-1
				count_gold+= 1

			if (p_ind=='O' and start_index_system!=-1 and end_index_system!=-1):
				system_entity[system_id] = Timex(system_id, " ".join(s[start_index_system:end_index_system+1]) , system_type, (start_index_system, end_index_system), system_unit, system_unit_second)
				start_index_system=-1
				end_index_system=-1
				count_system+= 1

		for k, v in system_entity.copy().items():
			if v.text=='':	del system_entity[k]
		test_list.append(gold_entity)
		pred_list.append(system_entity)

	gold_len = 0
	system_len = 0

	str_t_fscore, str_r_fscore, f1_type, f1_str_type, f1_unit, f1_str_unit, f1_val, acc_val = tempeval_calculate(test_list, pred_list, value, lang)
	
	for gold_entity, system_entity in zip(test_list, pred_list):
		for k, v in gold_entity.items():
			gold_len+=1
		for k, v in system_entity.items():
			system_len+=1
	print("Total gold: ",gold_len)
	print("Total system: ", system_len)
	
	return str_t_fscore, str_r_fscore, f1_type, f1_str_type, f1_unit, f1_str_unit, f1_val, gold_len, system_len, acc_val

def get_fscore(p, r): 
	"""Given precision and recall calculate f1 score"""
	if p+r == 0: 
		return 0 
	return 2.0*p*r/(p+r) 

'''
Eval Toolkit from TempEval-3 challenge: https://github.com/naushadzaman/tempeval3_toolkit
'''

def tempeval_calculate(test, pred, value, lang):
	"""
	Tools from the Tempeval-3 challenge to calculate evaluation
	"""
	global_system_timex = 0
	global_gold_timex = 0
	global_timex_strict_match4precision = 0
	global_timex_relaxed_match4precision = 0
	global_timex_strict_match4recall = 0
	global_timex_relaxed_match4recall = 0
	global_type_match = 0
	global_strict_type_match = 0
	global_value_match = 0
	global_strict_unit_match = 0
	global_unit_match = 0
	
	value_lower_level = []
	value_to_classify = []
	for gold_timex, system_timex, value_timex in zip(test, pred, value):
		for tid, val in zip(gold_timex, value_timex[1]):
			if gold_timex[tid].text.strip()=='':
				continue
			if tid in system_timex and system_timex[tid].text.strip() != '':
				g = gold_timex[tid]
				s = system_timex[tid]
				value_lower_level.append((s.text, val, value_timex[0], s.type, s.unit, s.unit_second))
		value_to_classify.append(value_lower_level)
		value_lower_level = []
	
	print("THE LEN IS," , len(value_to_classify))
	global_value_match = 0 #norma(lang, value_to_classify)

	for gold_timex, system_timex, value_timex in zip(test, pred, value):
		total_timex_relaxed_match4recall = 0
		total_timex_strict_match4recall = 0
		total_type_match = 0
		total_value_match = 0
		total_strict_type_match = 0
		total_strict_unit_match = 0
		total_unit_match = 0

		for tid, val in zip(gold_timex, value_timex[1]):
			if gold_timex[tid].text.strip()=='':
				continue
			if tid in system_timex and system_timex[tid].text.strip() != '':
				total_timex_relaxed_match4recall += 1
				#relaxed match
				g = gold_timex[tid]
				s = system_timex[tid]
				if g.text == s.text:
					total_timex_strict_match4recall += 1
					#strict match
					if g.type == s.type:	total_strict_type_match += 1
					if g.unit == s.unit:	total_strict_unit_match += 1
				if g.type == s.type:	total_type_match += 1
				if g.unit == s.unit:	
					total_unit_match += 1

		total_gold_timex = len(gold_timex)
		
		if total_gold_timex != 0:
			strict_timex_recall = 1.0*total_timex_strict_match4recall/total_gold_timex
			relaxed_timex_recall = 1.0*total_timex_relaxed_match4recall/total_gold_timex
		else:
			strict_timex_recall = 0 
			relaxed_timex_recall = 0		

		total_timex_strict_match4precision = 0
		total_timex_relaxed_match4precision = 0

		for tid in system_timex:
			if system_timex[tid].text.strip()=='':
				continue
			if tid in gold_timex and gold_timex[tid].text.strip() != '':
				total_timex_relaxed_match4precision += 1
				# relaxed match
				g = gold_timex[tid]
				s = system_timex[tid]
				if g.text == s.text:
					total_timex_strict_match4precision += 1
					# strict match
		total_system_timex = len(system_timex)

		if total_system_timex != 0: 
			strict_timex_precision = 1.0*total_timex_strict_match4precision/total_system_timex 
			relaxed_timex_precision = 1.0*total_timex_relaxed_match4precision/total_system_timex 
		else: 
			strict_timex_precision = 0 
			relaxed_timex_precision = 0

		global_system_timex += total_system_timex 
		global_gold_timex += total_gold_timex 
		global_timex_strict_match4precision += total_timex_strict_match4precision
		global_timex_relaxed_match4precision += total_timex_relaxed_match4precision
		global_timex_strict_match4recall += total_timex_strict_match4recall
		global_timex_relaxed_match4recall += total_timex_relaxed_match4recall
		global_type_match += total_type_match
		global_strict_type_match += total_strict_type_match
		global_value_match += total_value_match
		global_strict_unit_match += total_strict_unit_match
		global_unit_match += total_unit_match


	if global_gold_timex != 0 : 
		global_strict_timex_recall = 1.0*global_timex_strict_match4recall/global_gold_timex
		global_relaxed_timex_recall = 1.0*global_timex_relaxed_match4recall/global_gold_timex 
	else: 
		global_strict_timex_recall = 0 
		global_relaxed_timex_recall = 0 

	if global_system_timex != 0: 
		global_strict_timex_precision = 1.0*global_timex_strict_match4precision/global_system_timex 
		global_relaxed_timex_precision = 1.0*global_timex_relaxed_match4precision/global_system_timex 
	else: 
		global_strict_timex_precision = 0 
		global_relaxed_timex_precision = 0 

	if global_system_timex == 0 and global_gold_timex == 0: 
		strict_timex_fscore = 1 
		relaxed_timex_fscore = 1
		performance_type = 1 
		performance_value = 1
	else: 
		strict_timex_fscore = get_fscore(global_strict_timex_precision, global_strict_timex_recall)
		relaxed_timex_fscore = get_fscore(global_relaxed_timex_precision, global_relaxed_timex_recall)

		if global_timex_relaxed_match4recall != 0:
			performance_type = global_type_match*1.0/global_timex_relaxed_match4recall*relaxed_timex_fscore
			performance_strict_type = global_strict_type_match*1.0/global_timex_strict_match4recall*strict_timex_fscore
			performance_value = global_value_match*1.0/global_timex_relaxed_match4recall*relaxed_timex_fscore
			accuracy_type = global_type_match*1.0/global_timex_relaxed_match4recall
			accuracy_value = global_value_match*1.0/global_timex_relaxed_match4recall
			performance_unit = global_unit_match*1.0/global_timex_relaxed_match4recall*relaxed_timex_fscore
			performance_strict_unit = global_strict_unit_match*1.0/global_timex_strict_match4recall*strict_timex_fscore
		else: 
			performance_type = 0
			performance_strict_type = 0
			performance_value = 0 
			accuracy_type = 0 
			accuracy_value = 0
			performance_unit = 0
			performance_strict_unit = 0
			#timex_performance = strict_timex_fscore*w_strict_timex_fscore + relaxed_timex_fscore*w_relaxed_timex_fscore + performance_type*w_type + performance_value*w_value
	timex_extraction_performance = strict_timex_fscore*0.5 + relaxed_timex_fscore*0.5
	print("Strict timex fscore: ", str( round(strict_timex_fscore*100, 2)))
	print("Relaxed timex fscore: ", str( round(relaxed_timex_fscore*100, 2)))
	print("Attribute F1 strict Type: ", str( round(performance_strict_type*100, 2)))
	print("Attribute F1 Type: ", str( round(performance_type*100, 2)))
	print("Attribute F1 strict Unit: ", str( round(performance_strict_unit*100, 2)))
	print("Attribute F1 Unit: ", str( round(performance_unit*100, 2)))
	print("Attribute F1 Value: ", str( round(performance_value*100, 2)))
	print("Accuracy Value: ",  str( round(accuracy_value*100, 2)))

	return round(strict_timex_fscore*100, 2), round(relaxed_timex_fscore*100, 2), round(performance_type*100, 2), round(performance_strict_type*100, 2), round(performance_unit*100, 2), round(performance_strict_unit*100, 2), round(performance_value*100, 2), round(accuracy_value*100, 2)


def pred_wrapper(te, label, unit, unit_second, lang):
	"""TIMEX Annotation for predict model 

	Args:
		te (list): Sentence with TE
		label (list): timex type
		unit (list): timex unit
		unit_second (list): timex second unit
		lang (str): language of the model

	Returns:
		timex_list: list of timex object for the TEs
	"""	
	predicted_list = []

	for t, l, u, uu in zip(te, label, unit, unit_second):
		start_index_system = -1
		end_index_system = -1
		count_system = 1
		system_entity = {}
		t = nltk.word_tokenize(t) 
		pred_list_inner = []

		for idx, (t_ind, l_ind) in enumerate(zip(t, l)):
			print(t)
			system_id = 't'+str(count_system)


			if l_ind.startswith('B-') and start_index_system!=-1 and end_index_system!=-1:
				count_system += 1
				system_id = 't'+str(count_system)
				start_index_system = idx
				end_index_system = idx
				type = l_ind.split("-")[1]
				try:	system_unit = u[idx].split("-")[1]
				except:	system_unit = 'O'
				try:	system_unit_second = uu[idx].split("-")[1]
				except:	system_unit_second = 'O'
				try:	system_entity[system_id] = Timex(system_id, " ".join(t[start_index_system:end_index_system+1]), type, (start_index_system, end_index_system), system_unit, system_unit_second)
				except Exception as e:	print(e)
				
			if l_ind.startswith('B-') and start_index_system==-1 and end_index_system==-1:
				start_index_system = idx
				end_index_system = idx
				type = l_ind.split("-")[1]
				try:	system_unit = u[idx].split("-")[1]
				except:	system_unit = 'O'
				try:	system_unit_second = uu[idx].split("-")[1]
				except:	system_unit_second = 'O'
				try:	system_entity[system_id] = Timex(system_id, " ".join(t[start_index_system:end_index_system+1]), type, (start_index_system, end_index_system), system_unit, system_unit_second)
				except Exception as e:	print(e)
			
			if l_ind.startswith('I-'):
				end_index_system = idx
				try:	system_entity[system_id] = Timex(system_id, " ".join(t[start_index_system:end_index_system+1]), type, (start_index_system, end_index_system), system_unit, system_unit_second)
				except Exception as e:	print(e)
			
			if l_ind=='O' and start_index_system!=-1 and end_index_system!=-1:
				try:
					#print("HERE??", type)
					system_entity[system_id] = Timex(system_id, " ".join(t[start_index_system:end_index_system+1]), type, (start_index_system, end_index_system), system_unit, system_unit_second)
					start_index_system = -1
					end_index_system = -1
					count_system+= 1
				except Exception as e:	print(e)

		pred_list_inner.append(system_entity)
		predicted_list.append(pred_list_inner)
		timex_list = []#norma_predict(lang, predicted_list)
			
	return timex_list