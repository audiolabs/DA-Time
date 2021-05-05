"""
A Domain-Adapted Hybrid Temporal Tagger
Author: Touhidul Alam, Alessandra Zarcone, Sebastian Pado
Fraunhofer IIS, Uni Stuttgart

Usage:
	main.py exec --mode=<mode> [--run=<run>] [--amount==<amount>] [--emb=<emb>]
	main.py plot
	main.py predict <sent>...
	main.py --help

"""

from recognition.model import TErecognizer
from recognition.writer import write_json
from recognition.pipeline import InputPipeline
from recognition.dataset_mapping import *
from docopt import docopt
import recognition.util as util
import recognition.plot as plot


#=========================================================================
#
#  This file executes different model operation and plotting data            
#
#=========================================================================


if __name__=='__main__':
	args = docopt(__doc__)
	print(args)
	config = util.load_config("config.yaml")

	if args['exec']:
		max_len = config['train']['max_len']
		mode = args['--mode'] #train, test, ft
		train = config['train']['data']
		if args['--run']:	config['train']['run'] = args['--run']
		if args['--amount']:	config['train']['amount'] = args['--amount']
		if args['--emb']:	config['train']['embedding'] = args['--emb']

		if train=='tbaq': t_data = [tbaq_data, tempeval_plat_data]
		if train=='te3':	t_data = [tempeval_train_data, tempeval_plat_data]
		if train=='simpli_te3':	t_data = [simplified_tempeval_data, tempeval_plat_data]
		if train=='pate':	t_data = [pate_train_data, pate_test_data]

		print(config)
		ip = InputPipeline(config, t_data[0], t_data[1])
		TErecognizer(config, ip, mode).recognizer_model(ip.test_empty)

	if args['plot']:
		plot.plot_unit()
		plot.plot_label_dist()
		plot.plot_timex()		
		
		ip = InputPipeline(config, tempeval_train_data, snips_data)
		ip2 = InputPipeline(config, simplified_tempeval_data, pate_data)
		plot.plot_distribution([ip.training_length, ip.testing_length, ip2.testing_length, ip2.training_length],
			['TE-3','Snips','PATE', 'TE-3 Simplified'], ['red','blue','green', 'black'])
		
	if args['predict']:
		sents = args['<sent>']
		print(sents)
		#DUMMY ex: 'can you schedule from Saturday to Thursday?' 'Let us meet tomorrow'
		te_list, timex_list = TErecognizer(config=config).predict(sents)
		write_json(te_list, timex_list)
