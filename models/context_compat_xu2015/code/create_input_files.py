from utils import create_input_files
import argparse
import sys

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('label', type=str, choices=['description', 'caption'],
        help='whether the model is optimized to generate descriptions or captions')
	parser.add_argument('context', type=str,
		choices=['caption', 'context', 'description'],
        help='the type of text that the model receives as additional context to inform generation')
	parser.add_argument('--randomized', type=str, choices=['none', 'context', 'img'],
        help='what should be randomized for a baseline condition')
	parser.add_argument('--root_dir', type=str, 
		required=True, help='where the raw dataset is stored')
	parser.add_argument('--image_subdir', type=str,
		default="images",
		help='the name of the image subdir within `root_dir` that houses a decompressed version of `resized.zip`')
	args = parser.parse_args()

	if args.randomized == "context":
		json_name = 'wiki_split_randomcontext'
	elif args.randomized == "img":
		json_name = 'wiki_split_randomfilename'
	elif args.randomized == "none":
		json_name = 'wiki_split'
	else:
		sys.exit("invalid parser argument -- try img, context, or none.")

	# Create input files (along with word map)
	create_input_files(root_dir=args.root_dir,
		json_path=json_name+'.json',
		label=args.label,
		context=args.context,
		image_folder=args.image_subdir,
    	labels_per_image=1,
		# arxiv paper version:
    	# min_word_freq=1,
		min_word_freq=1,
		output_folder='concadia_xu2015_data/' + args.label + '_' + args.context + '_random' + args.randomized + '/',
    	max_len=50)
