import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import pandas as pd
import model
import config
import evaluate
import data_utils

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", 
		type=float, 
		default=0.01, 
		help="learning rate")
	parser.add_argument("--lamda", 
		type=float, 
		default=0.001, 
		help="model regularization rate")
	parser.add_argument("--batch_size", 
		type=int, 
		default=4096, 
		help="batch size for training")
	parser.add_argument("--epochs", 
		type=int,
		default=20,  
		help="training epoches")
	parser.add_argument("--top_k", 
		type=int, 
		default=[1, 3, 5, 10, 20, 50], 
		help="compute metrics@top_k")
	parser.add_argument("--factor_num", 
		type=int,
		default=32, 
		help="predictive factors numbers in the model")
	parser.add_argument("--num_ng", 
		type=int,
		default=4, 
		help="sample negative items for training")
	parser.add_argument("--test_num_ng", 
		type=int,
		default=99, 
		help="sample part of negative items for testing")
	parser.add_argument("--out", 
		default=True,
		help="save model or not")
	parser.add_argument("--gpu", 
		type=str,
		default="0",  
		help="gpu card ID")
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	cudnn.benchmark = True


	############################## PREPARE DATASET ##########################
	train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
	# construct the train and test datasets
	train_dataset = data_utils.BPRData(
			train_data, item_num, train_mat, args.num_ng, True)
	test_dataset = data_utils.BPRData(
			test_data, item_num, train_mat, 0, False)
	train_loader = data.DataLoader(train_dataset,
			batch_size=args.batch_size, shuffle=True, num_workers=4)
	test_loader = data.DataLoader(test_dataset,
			batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

	########################### CREATE MODEL #################################
	model = model.BPR(user_num, item_num, args.factor_num)
	#model.cuda()

	optimizer = optim.SGD(
				model.parameters(), lr=args.lr, weight_decay=args.lamda)
	# writer = SummaryWriter() # for visualization

	########################### TRAINING #####################################
	count, best_hr = 0, [0]
	HR_total, NDCG_total = [],[]
	for epoch in range(args.epochs):
		model.train() 
		start_time = time.time()
		train_loader.dataset.ng_sample()

		for user, item_i, item_j in train_loader:
			

			model.zero_grad()
			prediction_i, prediction_j = model(user, item_i, item_j)
			loss = - (prediction_i - prediction_j).sigmoid().log().sum()
			loss.backward()
			optimizer.step()
			# writer.add_scalar('data/loss', loss.item(), count)
			count += 1

		model.eval()
		HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
		
		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print('Top_Ks:',args.top_k,'\tHR:',HR,'\tNDCG',NDCG)
		HR_total.append(HR)
		NDCG_total.append(NDCG)
		if HR[0] > best_hr[0]:
			best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
			if args.out:
				if not os.path.exists(config.model_path):
					os.mkdir(config.model_path)
				torch.save(model, '{}BPR.pt'.format(config.model_path))
	HR_df = pd.DataFrame(HR_total, columns = ['Top-1','Top-3','Top-5','Top-10','Top-20','Top-50'])
	NDCG_df = pd.DataFrame(NDCG_total, columns = ['Top-1','Top-3','Top-5','Top-10','Top-20','Top-50'])
	HR_df.to_csv('HR_bpr_1_30_10m.csv')
	NDCG_df.to_csv('NDCG_bpr_1_30_10m.csv')

	print("End. Best epoch {:03d}: HR = {}, \
		NDCG = {}".format(best_epoch, best_hr, best_ndcg))
