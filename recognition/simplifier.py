from pycorenlp import StanfordCoreNLP
import pprint
import requests
import parser
from nltk import Tree
import pickle
import recognition.util as util


#=========================================================================
#
#  This file simplifies text with an external parser           
#
#=========================================================================


config = util.load_config("config.yaml")
pickle_out = open(config['directories']['input_modified_for_simplify'],'rb')
sentences = pickle.load(pickle_out)

texts=[]
for lines in sentences:
	lines = lines.strip()

	splitted = lines.split(" ")
	splitted = [s for s in splitted if s=='XXXXX']
	nlp = StanfordCoreNLP(config['external']['corenlp_ap'])
	output = nlp.annotate(lines, properties={
		    'annotators': 'parse',
		    'outputFormat': 'json',
		    'timeout': 10000,
	})
	tree = output['sentences'][0]["parse"]
	t = Tree.fromstring(tree)
	print(t)
	print(t.pretty_print())
	subtexts = []
	for subtree in t.subtrees():
		if subtree.label()=="S" or subtree.label()=="SBAR" or subtree.label()=='ROOT':
			subtexts.append(' '.join(subtree.leaves()))
	print(subtexts)
	if subtexts:
		main_sentence = subtexts[0]
		for i in reversed(range(len(subtexts)-1)):
			try:
				subtexts[i] = subtexts[i][0:subtexts[i].index(subtexts[i+1])]
			except IndexError as e:	print(e)
		subtexts = [l for l in subtexts if not len(l.split())<2 ]
		for t in subtexts:
			main_sentence = main_sentence.replace(t, '')
		subtexts.append(main_sentence)
		subtexts = [l for l in subtexts if not len(l.split())<2 ]
	else:	subtexts = [lines]

	l = [i for i in subtexts if 'XXXXX' in i]
	if not l:	l = [lines]

	cspl = 0
	for c in l:
		cs = c.split(" ")
		cs = [s for s in cs if s=='XXXXX']
		cspl += len(cs)
	texts.append(l)
pickle_out = open(config['directories']['input_simplified'],'wb')
pickle.dump(texts, pickle_out)
