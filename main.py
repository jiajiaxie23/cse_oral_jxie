import numpy as np
from exper import experiment
import os
import json
import argparse


def main(xi, output_path,relax, volume, st, states):
	path = '{}vol_{}_xi_{}_relax_{}_{}.json'.format(output_path, volume, xi, relax, states)
	count = 0
	while count < volume:
		rej, _dict = experiment(st, xi, relax = relax)
		if rej == True:
			pass
		else:
			count+=1
			if os.path.exists(path) == False:
				with open(path,'w') as f:
					f.write(json.dumps(_dict))
					f.write('\n')

			else:
				with open(path, 'a') as f:
					f.write(json.dumps(_dict))
					f.write('\n')



	return None






if __name__ =='__main__':


	parser = argparse.ArgumentParser(description='Run experiment of mixture begative binomial models with parameters')
	parser.add_argument("-xi", "--xi", default = 2, type=int, help="The Number of Topics")
	parser.add_argument("-v", "--volume", default = 500, type=int, help="The Number of Experiments")
	parser.add_argument("-re", "--relax", type = float, default = 0.2, help="Relaxation ratio")
	parser.add_argument('-p', '--path', type = str,  help = 'Path to output results')
	parser.add_argument('-st', '--states', type = str, default = 'all', help = 'Path to output results')
	args = parser.parse_args()
	xi = args.xi 
	relax = args.relax
	volume = args.volume
	path = args.path
	states = args.states

	if states == 'all':
		st = []
	else:
		st = states.split(',')

	main(xi, path, relax, volume, st, states)