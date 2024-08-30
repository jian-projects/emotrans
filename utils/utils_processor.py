import torch, os, copy
import numpy as np
from typing import Any
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import Adam, AdamW, SGD
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

def set_rng_seed(rng_seed: int = None, random: bool = True, numpy: bool = True,
                pytorch: bool = True, deterministic: bool = True):
    """
    设置模块的随机数种子。由于pytorch还存在cudnn导致的非deterministic的运行，所以一些情况下可能即使seed一样，结果也不一致
        需要在fitlog.commit()或fitlog.set_log_dir()之后运行才会记录该rng_seed到log中
        
    :param int rng_seed: 将这些模块的随机数设置到多少，默认为随机生成一个。
    :param bool, random: 是否将python自带的random模块的seed设置为rng_seed.
    :param bool, numpy: 是否将numpy的seed设置为rng_seed.
    :param bool, pytorch: 是否将pytorch的seed设置为rng_seed(设置torch.manual_seed和torch.cuda.manual_seed_all).
    :param bool, deterministic: 是否将pytorch的torch.backends.cudnn.deterministic设置为True
    """
    if rng_seed is None:
        import time
        import math
        rng_seed = int(math.modf(time.time())[0] * 1000000)
    if random:
        import random
        random.seed(rng_seed)
    if numpy:
        try:
            import numpy
            numpy.random.seed(rng_seed)
        except:
            pass
    if pytorch:
        try:
            import torch
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed(rng_seed)
            torch.cuda.manual_seed_all(rng_seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
        except:
            pass
    os.environ['PYTHONHASHSEED'] = str(rng_seed)  # 为了禁止hash随机化，使得实验可复现。
    return rng_seed

def get_scheduler(args, optimizer, iter_total, method=None):
    scheduler = None
    if method is None: method = args.model['optim_sched'][-1]
    if method is None: return None

    warmup_ratio = args.train['warmup_ratio']
    if 'linear' in method:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=iter_total*warmup_ratio, 
            num_training_steps=iter_total
        )
    if 'cosine' in method:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_ratio*iter_total, 
            num_training_steps=iter_total
        )

    return scheduler

def get_optimizer(model, methods=None):
    args = model.args
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if methods is None: methods = args.model['optim_sched']

    lr, lr_pre = args.train['learning_rate'], args.train['learning_rate_pre']
    weight_decay, adam_epsilon, l2reg = args.train['weight_decay'], args.train['adam_epsilon'], args.train['l2reg']

    no_decay = ['bias', 'LayerNorm.weight']
    if 'AdamW_' in methods:
        plm_params = list(map(id, model.plm_model.parameters()))
        model_params, warmup_params = [], []
        for name, model_param in model.named_parameters():
            weight_decay_ = 0 if any(nd in name for nd in no_decay) else weight_decay 
            lr_ = lr_pre if id(model_param) in plm_params else lr

            model_params.append({'params': model_param, 'lr': lr_, 'weight_decay': weight_decay_})
            warmup_params.append({'params': model_param, 'lr': lr_/4 if id(model_param) in plm_params else lr_, 'weight_decay': weight_decay_})
        
        model_params = sorted(model_params, key=lambda x: x['lr'])
        optimizer = AdamW(model_params)

    if 'AdamW' in methods:
        model_params = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
        ]
        optimizer = AdamW(model_params, lr=lr_pre, eps=adam_epsilon)
    
    if 'Adam' in methods: 
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(model_params, lr=lr, weight_decay=l2reg)

    if 'SGD' in methods:
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = SGD(model_params, lr=lr, weight_decay=l2reg)

    return optimizer


class Metrics(object):
    def __init__(self, args, dataset) -> Any:
        self.args = args
        self.dataset = dataset
        if dataset.name[0] in ['absa', 'img', 'asqp']:
            criterion = {'f1': 0, 'acc': 0, 'loss': 0, 'epoch': 0}

        if 'tkg' in dataset.name: self.static = {'mrr': -1, 'hits': {}}
        if 'msa' in dataset.name: self.static = {'mae': 1e8, 'corr': 0, 'loss': 1e5, 'epoch': 0}
        if 'med' in dataset.name: self.static = {'f1': 0, 'acc': 0, 'epoch': 0}

        self.train = copy.deepcopy(criterion)
        self.valid = copy.deepcopy(criterion)
        self.test  = copy.deepcopy(criterion)

        dataset.metric = next(iter(criterion)) # 更新比较对象

    def accuracy_score(self, preds=None, labels=None):
        return round(accuracy_score(labels, preds), 4)

    def f1_score(self, preds=None, labels=None, average='macro'):
        return round(f1_score(labels, preds, average=average), 4)

    def get_metric(self, results, task, stage='train'):
        # if task == 'erc': return self.dataset._score(results)

        if task in ['absa', 'img']:
            labels, preds, total_loss = [], [], 0
            for rec in results:
                labels.extend(rec['labels'].cpu().numpy())
                preds.extend(rec['preds'].cpu().numpy())
                total_loss += rec['loss'].item()*len(rec['labels'])
            
            average = 'macro' if task == 'absa' else 'weighted'
            return {
                'f1'  : round(f1_score(labels, preds, average=average), 4),
                'acc' : round(accuracy_score(labels, preds), 4),
                'loss': round(total_loss/len(labels), 3)
            }
        
        if task == 'asqp':
            sent_pred, sent_gold = [], []
            for rec in results:
                sent_pred.extend(rec['output'])
                sent_gold.extend(rec['target'])
            
            preds = [self.dataset.extract_span(s, 'pred') for s in sent_pred] 
            labels = [self.dataset.extract_span(s, 'gold') for s in sent_gold] 

            return self.dataset._score(preds, labels)
        
        # if task == 'seg':
        #     total, total_dice, total_iou, total_loss = 0,0,0,0
        #     for rec in results:
        #         bz = rec['labels'].shape[0]
        #         total_dice += rec['dice_fn'](rec['logits'], rec['labels'].float()).item() * bz
        #         total_iou += rec['iou_fn'](rec['labels'], rec['logits']) * bz
        #         total_loss += rec['loss'].item() * bz
        #         total += bz
            
        #     return {
        #         'loss': round(total_loss/total, 2),
        #         'dice': round(total_dice/total, 2),
        #         'iou':  round(total_iou/total, 2),
        #     }
    
    def get_metric_tkg(self, results, hits=[1, 3, 10], stage='valid'):
        def get_hits_k(logits, labels, k):
            _, indices = logits.topk(k, dim=1)
            correct = indices.eq(labels.view(-1, 1).expand_as(indices))
            hits_k = correct.sum().item() / labels.size(0)
            return hits_k
        
        def get_mrr(logits, labels):
            _, indices = logits.sort(descending=True)
            ranks = (indices == labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
            mrr = (1.0 / ranks.float()).mean().item()
            return mrr

        # conbine results
        t_logits, t_labels = [], []
        for rec in results:
            t_logits.extend(rec['logits'])
            t_labels.extend(rec['labels'])

        # metric calculate
        idx, interval, ranks = 0, 1000, [] # out of memory
        while idx < len(t_labels):
            b_logits, b_labels= torch.stack(t_logits[idx:idx+interval]), torch.stack(t_labels[idx:idx+interval])
            _, indices = b_logits.sort(descending=True)
            ranks.extend((indices == b_labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1)
            idx += interval
        MRR =  round((1.0/torch.stack(ranks).float()).mean().item(), 4)
        Hits = {k: round(torch.mean((torch.stack(ranks) <= k).float()).item(), 4) for k in hits}

        return {
            'mrr': MRR, 
            'hits': Hits,
        }

    def get_metric_msa(self, results, stage='train'):
        preds = np.concatenate([rec['preds_regress'] for rec in results])
        labels = np.concatenate([rec['labels_regress'] for rec in results])
        mae, corr = np.mean(np.absolute(preds-labels)), np.corrcoef(preds, labels)[0][1]

        # metric after value round 
        acc_i, f1_i = {3: 0, 2: 0, 1: 0}, {3: 0, 2: 0, 1: 0}
        for clip in [3, 2, 1]: 
            pre_tmp, lab_tmp = np.round(np.clip(preds, a_min=-clip, a_max=clip)), np.round(np.clip(labels, a_min=-clip, a_max=clip))
            acc_i[clip], f1_i[clip] = self.accuracy_score(pre_tmp, lab_tmp), self.f1_score(pre_tmp, lab_tmp, average='weighted')

        # metric of binary
        acc_b, f1_b = {'w0': 0, 'w_0': 0}, {'w0': 0, 'w_0': 0} 
        for key in ['w0', 'w_0']: # w0: negative/non-negative; w_0: negative/positive
            if key == 'w0':
                zero_mark = np.arange(len(labels))
                pre_tmp, lab_tmp = preds[zero_mark] >= 0, labels[zero_mark] >= 0
            else:
                zero_mark = np.array([i for i,v in enumerate(labels) if v])
                pre_tmp, lab_tmp = preds[zero_mark] > 0, labels[zero_mark] > 0
            acc_b[key], f1_b[key] = self.accuracy_score(pre_tmp, lab_tmp), self.f1_score(pre_tmp, lab_tmp, average='weighted')
            
        score = {
            'mae': round(mae, 4),
            'corr': round(corr, 4), 
            'accuracy': {
                'integer': acc_i,
                'binary': acc_b,
            },
            'f1_score': {
                'integer': f1_i,
                'binary': f1_b,
            }
            # 'accuracy': acc_b,
            # 'f1_score': f1_b,
        }

        return score