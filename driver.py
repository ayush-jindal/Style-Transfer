#!/home/jindal/Py37/bin/python

import sys
import getopt
import os
from itertools import zip_longest

from main import train

def main(argv):

	content_path = ''
	style_path = ''
	input_file = ''
	result_path = ''
	result_dir = 'reconstructions_losses/'
	beta = 1e4
	epochs = 7.5e3
	optim_fn = 'adam'
	verbose = False
	preserve_color = False
	multiple_styles = 1
	max_size = []
	combinations = {}
	results = []
	
	try:
		opts, args = getopt.getopt(argv, 'hvpb:c:s:o:e:', ['preserve-color','beta=', 'content-file=', 'style-file=', 'output-file=', 'epochs=', 'output-dir=', 'max-size=', 'optimizer=', 'input-file='])
	except getopt.GetoptError:
		print('error')
		sys.exit(1)
	print(opts)
	for opt, arg in opts:
		if opt == '-h':
			print('driver.py -c <contentfile> -s <stylefile> -b [beta] -e [epochs] -o [outputfile]')
			print('-h\n-v\n--beta\n--content-file\n--style-file\n--output-file\n--epochs\n--output-dir\n--max-size\n--optimizer\n--input-file')
			sys.exit(0)
		elif opt == '-v':
			verbose = True
		elif opt in ('-b', '--beta'):
			beta = arg
		elif opt in ('-c', '--content-file'):
			content_path = arg
		elif opt in ('-s', '--style-file'):
			style_path = arg
		elif opt in ('-o', '--output-file'):
			result_path = arg
		elif opt in ('-e', '--epochs'):
			epochs = arg
		elif opt in ('--output-dir'):
			result_dir = arg
		elif opt in ('--max-size'):
			max_size = [(int(arg),)*2]
		elif opt in ('--optimizer'):
			optim_fn = arg.lower()
		elif opt in ('--input-file'):
			input_file = arg
		elif opt in('-p', '--preserve-color'):
			preserve_color = True

	if input_file != '':
		
		with open('file.txt', 'r') as imgs:
			lines = imgs.readlines()
		i = 0
		while i < len(lines):	
			num_styles = int(lines[i].strip('\n'))
			i+=1
			content = lines[i].strip('\n')
			result_path = content.split('/')[-1][:-4]
			i+=1
			combinations[content] = []
			for j in range(num_styles):
				style = lines[i].strip('\n')
				result_path += '_' + style.split('/')[-1][:-4]
				combinations[content].append(style)
				i+=1
			results.append(result_path+str(epochs)+optim_fn+' '+str(beta)+'.jpg')
		del lines, content, i
	elif '' in [content_path, style_path]:
		print('driver.py -b [beta] -c <contentfile> -s <stylefile> -o [outputfile]')
		sys.exit(1)
	else:
		combinations[content_path] = [style_path]
		if result_path == '':
			results.append(style_path.split('/')[-1][:-4]+'_'+content_path.split('/')[-1][:-4]+str(epochs)+optim_fn+' '+str(beta)+'.jpg')
		else:
			results.append(result_path)
			
	del content_path, style_path, result_path
	
	os.makedirs(result_dir, exist_ok=True)
	
	print('Starting the program...')
	
	print(combinations)
	print(results)
	print(verbose)
	
	train(combinations, results, result_dir, beta, epochs, max_size, optim_fn, verbose, preserve_color)
		
if __name__ == '__main__':
	main(sys.argv[1:])
