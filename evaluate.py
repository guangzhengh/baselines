import numpy as np
import torch
from tqdm import tqdm

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

'''def hit(gt_item, pred_items):
    rate = 0

    for g in gt_item:
        if g in pred_items:
            rate += 1

    return float(rate)/float(len(gt_item))'''

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0
'''def ndcg(gt_item, pred_items):
    rate = 0
    for g in gt_item:
        if g in pred_items:
            index = pred_items.index(g)
            rate += np.reciprocal(np.log2(index+2))
    return float(rate)/float(len(gt_item))'''

def metrics(model, test_loader, top_k):
    HR, NDCG = [list() for i in range(len(top_k))], [list() for i in range(len(top_k))]
    for user, item_i, item_j in tqdm(test_loader):
        #user = user.cuda()
        #item_i = item_i.cuda()
        #item_j = item_j.cuda() # not useful when testing
        prediction_i, prediction_j = model(user, item_i, item_j)

        gt_item = item_i[0].item()
        for i in range(len(top_k)):
            _, indices = torch.topk(prediction_i, top_k[i])
            recommends = torch.take(
                    item_i, indices).cpu().numpy().tolist()
            HR[i].append(hit(gt_item, recommends))
            NDCG[i].append(ndcg(gt_item, recommends))

    HR_new = []
    NDCG_new = []
    for i in HR:

        HR_new.append(np.mean(i, dtype=np.float32))
    for i in NDCG:
        NDCG_new.append(np.mean(i, dtype=np.float32))

    return HR_new, NDCG_new
