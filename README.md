# baselines
 BPR_NeuMF
 
run:

python main.py

if you want to use other dataset, change the line 38 in data_utils.py:

with open(config.dataset_100k, 'rb') as f:

  to
  
with open(config.dataset_1m, 'rb') as f:
