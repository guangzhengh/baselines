# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# paths
main_path = '/Users/guangzhengyisha/Desktop/pairwise/BPR-pytorch/Data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)
dataset_100k = main_path + 'dataset_100k.pickle'
dataset_1m = main_path + 'dataset_10m.pickle'
model_path = './models/'
BPR_model_path = model_path + 'BPR.pth'
