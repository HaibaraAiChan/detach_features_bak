# '''
# reddit data mini batch train and full batch subgraph generation, save to /DATA/ folder 
# number of graphs == number of epoches
# '''
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
# from block_dataloader import generate_dataloader
from block_dataloader_graph import generate_dataloader
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import deepspeed
import random
from graphsage_model import SAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data
from load_graph import load_ogbn_mag
from memory_usage import see_memory_usage, nvidia_smi_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean
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

def CPU_DELTA_TIME(tic, str1):
	toc = time.time()
	print(str1 + ' spend:  {:.6f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	model.eval()
	with torch.no_grad():
		pred = model.inference(g, nfeat, device, args)
	model.train()
	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
	# Unpack data
	g, feats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(feats[0])
	print('in_feats ', in_feats)
	nvidia_smi_list=[]

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])

	full_batch_size = len(train_nid)
	train_dataloader = dgl.dataloading.NodeDataLoader(
		g,
		train_nid,
		sampler,
		batch_size=args.batch_size,
		# batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	
	
	model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	nvidia_smi_list.append(nvidia_smi_usage()) # GPU

	print(args.batch_size)
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	iter_tput = []
	avg_step_data_trans_time_list = []
	avg_step_GPU_train_time_list = []
	avg_step_time_list = []
	avg_CPU_without_train_dataloader=[]
	
	avg = 0
	
	for epoch in range(args.num_epochs):
		tic = time.time()
		print('Epoch ' + str(epoch))		
		tic_step = time.time()
		nvidia_smi_list.append(nvidia_smi_usage()) # GPU
		
		CPU_without_train_dataloader=[]
		step_time_list = []
		step_data_trans_time_list = []
		step_GPU_train_time_list = []
		for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
			if args.batch_size-len(seeds)>50: # drop the tail less than ideal mini batch 
				continue
			print('\t step ', step)
			print('length of inputs: ',len(input_nodes))
			print('length of outputs: ',len(seeds))
			

			#----------------------------------- Load the input features as well as output labels
			step_s=time.time()
			torch.cuda.synchronize()
			start.record()
			
			batch_inputs, batch_labels = load_subtensor(feats, labels, seeds, input_nodes, device)
			nvidia_smi_list.append(nvidia_smi_usage()) # GPU
			blocks = [block.int().to(device) for block in blocks]
			nvidia_smi_list.append(nvidia_smi_usage()) # GPU
			
			end.record()
			torch.cuda.synchronize()  # wait for move to complete
			step_data_trans_time_list.append(start.elapsed_time(end))
			
			start.record()
			batch_pred = model(blocks, batch_inputs)
			loss = loss_fcn(batch_pred, batch_labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			end.record()
			torch.cuda.synchronize()  # wait for all training steps to complete
			step_GPU_train_time_list.append(start.elapsed_time(end))
			nvidia_smi_list.append(nvidia_smi_usage()) # GPU
			
			step_time = time.time() - tic_step
			step_time_list.append(step_time)
			CPU_without_train_dataloader.append(time.time()-step_s)
			
			iter_tput.append(len(seeds) / (time.time() - tic_step))
			# if step % args.log_every == 0:
			# 	acc = compute_acc(batch_pred, batch_labels)
			# 	gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 /1024 if torch.cuda.is_available() else 0
			# 	print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.4f} GB'.format(
			# 		epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
			tic_step = time.time()

		toc = time.time()
		
		print('Epoch Time(s): {:.4f}'.format(toc - tic))
		if epoch >= 2:
			avg += toc - tic
			
			
		transer_time = mean(step_data_trans_time_list)
		print('\t\t avg batch data from CPU -> GPU time\t:%.8f s' % (transer_time/1000))
		avg_step_data_trans_time_list.append(transer_time)
	
		gpu_time = mean(step_GPU_train_time_list)
		print('\t\t avg batch  GPU training time\t:%.8f s' % (gpu_time/1000))
		avg_step_GPU_train_time_list.append(gpu_time)
	
		CPU_time_with_dataloader = mean(step_time_list)
		print('\t\t avg batch train with train_dataloader total CPU time \t:%.8f s' % (CPU_time_with_dataloader ))
		avg_step_time_list.append(CPU_time_with_dataloader)
		
		CPU_without_dataloader= mean(CPU_without_train_dataloader)
		print('\t\t avg batch  train  without datloader total CPU time \t:%.8f s' % (CPU_without_dataloader ))
		avg_CPU_without_train_dataloader.append(CPU_without_dataloader)
		
	print('Avg epoch time: {}'.format(avg / (epoch - 1)))
	print('='*100)
	print('CPU memory usage ') 
	print(get_memory('')) # CPU
	print('\n max nvidia-smi memory usage, '+ str(max(nvidia_smi_list))+' GB' )
	see_memory_usage('') # GPU
	
	print()
	print('-*'*50)
	out_indent = 2 # skip the first 2 epochs, initial epoch time is not stable.
	print('avg_step_time_list-------------')
	print(len(avg_step_time_list))
	avg_batch_CPU_time = sum(avg_step_time_list[out_indent:]) / len(avg_step_time_list[out_indent:])
	print('\t avg batch train CPU time with train dataloader \t:%.8f s' % (avg_batch_CPU_time ))
	
	avg_batch_cpu_train_time_wo = sum(avg_CPU_without_train_dataloader[out_indent:]) / len(avg_CPU_without_train_dataloader[out_indent:])
	print('\t avg batch train CPU w/o train dataloader\t:%.8f s' % (avg_batch_cpu_train_time_wo ))
	
	avg_batch_data_trans = sum(avg_step_data_trans_time_list[out_indent:]) / len(avg_step_data_trans_time_list[out_indent:])
	print('\t avg batch data from cpu -> GPU time\t\t:%.8f s' % (avg_batch_data_trans/1000))
	avg_batch_gpu = sum(avg_step_GPU_train_time_list[out_indent:]) / len(avg_step_GPU_train_time_list[out_indent:])
	print('\t avg batch GPU training time \t\t\t:%.8f s' % (avg_batch_gpu/1000))
	print()

	print('='*100)
	# train_acc = evaluate(model, g, feats, labels, train_nid, device)
	# print('train Acc: {:.4f}'.format(train_acc))
	# test_acc = evaluate(model, g, feats, labels, test_nid, device)
	# print('Test Acc: {:.4f}'.format(test_acc))
	


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
	# argparser.add_argument('--selection-method', type=str, default='range')
	argparser.add_argument('--num-runs', type=int, default=2)
	argparser.add_argument('--num-epochs', type=int, default=6)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='20')
	argparser.add_argument('--fan-out', type=str, default='10')


	# argparser.add_argument('--batch-size', type=int, default=157393)
	# argparser.add_argument('--batch-size', type=int, default=78697)
	# argparser.add_argument('--batch-size', type=int, default=39349)
	# argparser.add_argument('--batch-size', type=int, default=19675)
	# argparser.add_argument('--batch-size', type=int, default=9838)
	# argparser.add_argument('--batch-size', type=int, default=4919)

	argparser.add_argument('--batch-size', type=int, default=153431)
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

	# argparser.add_argument("--eval-batch-size", type=int, default=100000,
                        # help="evaluation batch size")
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

	set_seed(args)
	
	main(args)

