import json
from recognition.util import load_config

#=========================================================================
#
#  This file writes the predicted output to a file           
#
#=========================================================================


config = load_config("config.yaml")

def write_json(te_list, timex_list):
	"""	Given the list of TEs and their TIMEXes, write to a json file

	Args:
		te_list (list(string)): List containing TEs
		timex_list (list(timex object)): List containing TIMEX3 for TEs
	"""	
	
	write_to_json = []
	print(te_list, timex_list)
	for te, timex in zip(te_list, timex_list):
		data = {}
		data['text'] = te
		t_list = []
		for t in timex:
			for k, v in t.items():
				t_data = {}
				t_data['expression'] = v.text if v.text != '__EMPTY__' else ""
				t_data['tid'] = v.tid
				t_data['type'] = v.type
				t_data['unit'] = v.unit if v.unit != 'O' else ""
				t_data['value'] = v.value
				t_data['beginPoint'] = v.beginPoint if v.beginPoint else ""
				t_data['endPoint'] = v.endPoint if v.endPoint else ""
				t_list.append(t_data)
		data['TIMEX3'] = t_list
		write_to_json.append(data)
		
	with open(config['directories']['timex_predict_output'],'w') as f:
		json.dump(write_to_json, f, indent=2)