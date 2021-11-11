import dgl
from dgl.data.utils import save_graphs
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from block_dataloader import generate_dataloader
import dgl.nn.pytorch as dglnn
import time
import argparse

import deepspeed
import random
from graphsage_model import SAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data
from load_graph import load_ogbn_mag
from memory_usage import see_memory_usage

from cpu_mem_usage import get_memory

# from utils import draw_graph_global


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.gpu >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)



#### Entry point
def run(args, device, data):
		# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in_feats--------------------------------------')
	print(in_feats)
	# dataloader_device = torch.device('cpu')

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])

	# full_batch_size = len(train_nid)
	given_output=args.given_output
	batch_dataloader = dgl.dataloading.NodeDataLoader(
		g,
		train_nid,
		sampler,
		batch_size=given_output,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	
	avg_src=[]
	import numpy
	while True:
		for ee in range(4):
			for step, (input_nodes, output_seeds, blocks) in enumerate(batch_dataloader):
			    				# print('full_batch_blocks')
				print(blocks)
				avg_src.append(len(input_nodes))
				src_len=len(input_nodes)
				print(str(args.given_input) +' V.S. '+str(src_len))
				if abs(src_len - args.given_input) < 20: 
					for cur_block in blocks:
						block_to_graph=dgl.block_to_graph(cur_block)
						save_graphs('./DATA/ideal_mini_subgraph/'+args.dataset+'_'+str(len(input_nodes))+'_subgraph.bin',[block_to_graph])
						print('--------------------done--------------------', args.given_input)
						return
		
		if numpy.mean(avg_src) < args.given_input:
			given_output = given_output + 10
		else:
			given_output = given_output - 10
		avg_src=[]
		batch_dataloader = dgl.dataloading.NodeDataLoader(
			g,
			train_nid,
			sampler,
			batch_size=given_output,
			shuffle=True,
			drop_last=False,
			num_workers=args.num_workers)
    		
        			
		
			


def main(args):
    
	device = "cpu"
	
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--gpu', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)

	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--selection-method', type=str, default='range')
	argparser.add_argument('--num-runs', type=int, default=2)
	argparser.add_argument('--num-epochs', type=int, default=6)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='20')
	argparser.add_argument('--fan-out', type=str, default='10')
	#----------------------------------------------------------------------------
	# argparser.add_argument('--given-input', type=int, default=108631)
	# argparser.add_argument('--given-output', type=int, default=18900)
	#----------------------------------------------------------------------------
	#----------------------------------------------------------------------------
	# argparser.add_argument('--given-input', type=int, default=54315)
	# argparser.add_argument('--given-output', type=int, default=6800)
	#----------------------------------------------------------------------------
	#----------------------------------------------------------------------------
	# argparser.add_argument('--given-input', type=int, default=27157)
	# argparser.add_argument('--given-output', type=int, default=2950)
	#----------------------------------------------------------------------------
	#----------------------------------------------------------------------------
	# argparser.add_argument('--given-input', type=int, default=13578)
	# argparser.add_argument('--given-output', type=int, default=1350)
	#----------------------------------------------------------------------------
	#----------------------------------------------------------------------------
	# argparser.add_argument('--given-input', type=int, default=6789)
	# argparser.add_argument('--given-output', type=int, default=650)
	#----------------------------------------------------------------------------
	# argparser.add_argument('--given-input', type=int, default=3394)
	# argparser.add_argument('--given-output', type=int, default=324)
	#----------------------------------------------------------------------------
	argparser.add_argument('--given-input', type=int, default=1697)
	argparser.add_argument('--given-output', type=int, default=159)


	argparser.add_argument('--batch-size', type=int, default=157393)
	# argparser.add_argument('--batch-size', type=int, default=78697)
	# argparser.add_argument('--batch-size', type=int, default=39349)
	# argparser.add_argument('--batch-size', type=int, default=19675)
	# argparser.add_argument('--batch-size', type=int, default=9838)
	# argparser.add_argument('--batch-size', type=int, default=4919)

	# argparser.add_argument('--batch-size', type=int, default=15341)
	# argparser.add_argument('--batch-size', type=int, default=78697)


	# argparser.add_argument('--batch-size', type=int, default=196571)
	# argparser.add_argument('--batch-size', type=int, default=98308)
	# argparser.add_argument('--batch-size', type=int, default=49154)
	# argparser.add_argument('--batch-size', type=int, default=24577)
	# argparser.add_argument('--batch-size', type=int, default=12289)
	# argparser.add_argument('--batch-size', type=int, default=6145)
	# argparser.add_argument('--batch-size', type=int, default=3000)
	# argparser.add_argument('--batch-size', type=int, default=1500)
	# argparser.add_argument('--batch-size', type=int, default=8)

	argparser.add_argument("--eval-batch-size", type=int, default=100000,
                        help="evaluation batch size")
	argparser.add_argument("--R", type=int, default=5,
                        help="number of hops")

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument('--inductive', action='store_true',
		help="Inductive learning setting") #The store_true option automatically creates a default value of False
	argparser.add_argument('--data-cpu', action='store_true',
		help="By default the script puts all node features and labels "
		     "on GPU when using it to save time for data copy. This may "
		     "be undesired if they cannot fit in GPU memory at once. "
		     "This flag disables that.")
	args = argparser.parse_args()

	# set_seed(args)
	
	main(args)
