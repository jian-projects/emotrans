import torch, os, json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, AutoModelWithLMHead
from peft import LoraConfig, get_peft_model

from transformers import logging
logging.set_verbosity_error()

from utils_model import *
from utils_rnn import DynamicLSTM
from utils_contrast import Contrast_Single
from data_loader import ERCDataset_Multi

max_seq_lens = {'meld': 128}

class ERCDataset_EmoTrans(ERCDataset_Multi):
    def label_change(self):
        new_labels, new_labels_dict = json.load(open(self.data_dir + 'label_change', 'r')), {}
        for lab, n_lab in new_labels:
            new_labels_dict[lab] = n_lab
        self.tokenizer_['labels']['e2l'] = new_labels_dict

    def prompt_utterance(self, dataset):
        for stage, convs in dataset.items():
            for conv in convs:
                spks, txts = conv['speakers'], conv['texts']
                emos = [self.tokenizer_['labels']['e2l'][emo] if emo else 'none' for emo in conv['emotions']] # 转换 emo
                emos_lab_id = [self.tokenizer_['labels']['ltoi'][emo] if emo else -1 for emo in conv['emotions']] # 获取 emo_id
                emos_token_id = [self.tokenizer.encode(e)[1] for e in emos]
                assert len(emos_lab_id) == len(emos_token_id)

                prompts = [f"{spk}: {txt} {self.tokenizer.sep_token} {spk} expresses {self.tokenizer.mask_token} {self.tokenizer.sep_token}"
                           for spk, txt in zip(spks, txts)]
                embeddings = self.tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt')
                
                conv['new_emos'] = emos
                conv['emos_lab_id'] = emos_lab_id
                conv['emos_token_id'] = emos_token_id
                conv['prompts'] = prompts
                conv['embeddings'] = embeddings

                # 记录相关信息
                self.info['num_conv_speaker'][stage].append(len(set(spks))) # conv speaker number
                self.info['num_conv_utt'][stage].append(len(spks)) # conv utterance number
                self.info['num_conv_utt_token'][stage].append(embeddings.attention_mask.sum(dim=1).tolist()) # conv utterance token number

            self.datas[stage] = convs

    def extend_sample(self, dataset, mode='online'):
        for stage, convs in dataset.items():
            samples = []
            for conv in tqdm(convs):
                conv_input_ids, conv_attention_mask = conv['embeddings'].input_ids, conv['embeddings'].attention_mask
                for ui, (emo_lab_id, emo_token_id) in enumerate(zip(conv['emos_lab_id'], conv['emos_token_id'])):
                    ## 一前一后 交替拼接，标注当前位置
                    cur_mask = [1] # 定位当前 utterance 位置
                    emo_flow_token_ids, emo_flow_token_label = [emo_token_id], [emo_lab_id]
                    input_ids_ext = conv_input_ids[ui][0:conv_attention_mask[ui].sum()].tolist()[-self.max_seq_len:]
                    # input_ids_ext_t = conv_input_ids_t[ui][0:conv_attention_mask_t[ui].sum()].tolist()[-self.max_seq_len:]
                    for i in range(1,len(conv['emotions'])):
                        if ui-i >=0:
                            tmp = conv_input_ids[ui-i][0:conv_attention_mask[ui-i].sum()].tolist()
                            if len(input_ids_ext) + len(tmp) <= self.max_seq_len:
                                cur_mask = [1] + cur_mask
                                input_ids_ext = tmp + input_ids_ext
                                # input_ids_ext_t = conv_input_ids_t[ui-i][0:conv_attention_mask_t[ui-i].sum()].tolist() + input_ids_ext_t
                                emo_flow_token_ids = [conv['emos_token_id'][ui-i]] + emo_flow_token_ids
                                emo_flow_token_label = [conv['emos_lab_id'][ui-i]] + emo_flow_token_label
                            else: break
                    
                    input_ids_ext = torch.tensor([self.tokenizer.cls_token_id] + input_ids_ext) # 增加 cls token
                    label_category = self.tokenizer_['labels']['ltoi'][conv['emotions'][ui]] if conv['emotions'][ui] else -1
                    if label_category == -1: continue
                    sample = {
                        'index':    len(samples),
                        'text':     conv['texts'][ui],
                        'speaker':  conv['speakers'][ui],
                        'emotion':  conv['emotions'][ui],
                        'prompt':   conv['prompts'][ui],
                        'input_ids':   input_ids_ext, 
                        'attention_mask':   torch.ones_like(input_ids_ext),
                        'label': label_category, 
                        'cur_mask': torch.tensor(cur_mask), 
                        'emo_flow_token_ids': torch.tensor(emo_flow_token_ids), 
                        'emo_flow_token_label': torch.tensor(emo_flow_token_label),
                    }
                    samples.append(sample)

                    # 统计一下信息
                    if conv['emotions'][ui] not in self.info['emotion_category']:
                        self.info['emotion_category'][conv['emotions'][ui]] = 0
                        self.info['emotion_category'][conv['emotions'][ui]] += 1

            # 记录相关信息
            self.info['num_samp'][stage] = len(samples)
            self.datas[stage] = samples

    def ret_samples(self, dataset, ret_num=3):
        self.flow_num = ret_num
        label_flow_matrix = pad_sequence([item['emo_flow_token_label'] for item in dataset['train']], batch_first=True, padding_value=-1)
        for desc, samples in dataset.items():
            for sample in tqdm(samples):
                sample['ret_idx'] = {}
                ret_k = label_flow_matrix.shape[1]
                for k in range(1, ret_k): # range(1,flow_num+1):
                    if k != self.flow_num: continue
                    query = sample['emo_flow_token_label'][0:k]
                    if len(query) < k: query = torch.cat([query, torch.ones([k-len(query)])*-1]).type_as(query)
                    masks = (label_flow_matrix[:,0:k]==query).sum(dim=-1)==k
                    #sample['ret_idx'][k] = [idx for idx, mask in enumerate(masks) if mask]
                    if 'ddg' in self.name and sample['label']==0: 
                        masks = torch.tensor(list(range(len(label_flow_matrix))))==sample['index']
                    sample['ret_idx'][k] = masks

    def setup(self, tokenizer, max_seq_len=128):
        self.tokenizer, self.max_seq_len = tokenizer, max_seq_len
        self.label_change() # 需要改变label or 将label加进字典, 使tokenizer后只有1位数字
        self.prompt_utterance(self.datas) # 给 utterance 增加 prompt
        self.extend_sample(self.datas) # 扩充 utterance, 构建样本
        self.ret_samples(self.datas) # 检索具有相同emo flow的样本

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col or 'flow' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            elif 'ret' in col:
                if self.flow_num: inputs[col] = torch.stack([sample[col][self.flow_num] for sample in samples])
                else: inputs[col] = torch.stack([sample[col][self.flow_num+1] for sample in samples])
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def config_for_model(args, scale='base'):
    scale = args.model['scale'] if 'scale' in args.model else scale
    args.model['plm'] = args.file['plm_dir'] + f"roberta-{scale}"
    
    args.model['data'] = f"{args.file['cache_dir']}{args.model['name']}.{scale}"


    return args
             
def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    
    ## 2. 导入数据
    data_path = args.model['data']
    if os.path.exists(data_path):
        dataset = torch.load(data_path)
    else:
        data_dir = f"{args.file['data_dir']}{args.train['tasks'][-1]}/"
        dataset = ERCDataset_EmoTrans(data_dir, args.train['batch_size'])
        tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])      
        dataset.setup(tokenizer, max_seq_len=max_seq_lens[args.train['tasks'][-1]])
        torch.save(dataset, data_path)
    
    dataset.batch_cols = {
        'index': -1,
        'input_ids': dataset.tokenizer.pad_token_id, 
        'attention_mask': 0, 
        'label': -1,
        'cur_mask': 0,
        'emo_flow_token_ids': dataset.tokenizer.pad_token_id,
        'emo_flow_token_label': -1,
        'ret_idx': -1
    }

    model = EmoSKD(
        args=args,
        dataset=dataset,
        plm=args.model['plm'],
    )
    return model, dataset


class EmoSKD(ModelForClassification):
    def __init__(self, args, dataset, plm=None):
        super().__init__() # 能继承 ModelForClassification 的属性
        self.args = args
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.mask_token_id = dataset.tokenizer.mask_token_id

        self.plm_model = AutoModel.from_pretrained(plm if plm is not None else args.model['plm'])
        self.plm_model.lm_head = AutoModelWithLMHead.from_config(self.plm_model.config).lm_head
        self.plm_pooler = PoolerAll(self.plm_model.config)  
        self.hidden_size = self.plm_model.config.hidden_size

        if self.args.model['use_lora']:
            peft_config = LoraConfig(inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1)
            self.plm_model = get_peft_model(self.plm_model, peft_config)
        
        self.lstm = DynamicLSTM(self.hidden_size, self.hidden_size, bidirectional=False)
        self.conl = Contrast_Single(temp=1, method='cos')
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)

        self.weight = args.model['weight']
        self.flow_num = args.model['flow_num']
        self.bank = np.array([None] * len(dataset.datas['train']))

    def encode(self, inputs, stage='train'):
        outputs = {'student': None, 'teacher': None }
        if self.args.model['use_lora']:
            encode_model = self.plm_model.base_model
        else: encode_model = self.plm_model
        
        # 1. encoding
        plm_outs = encode_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )

        flow_mask = inputs['input_ids'] == self.mask_token_id
        label_mask = inputs['emo_flow_token_label'] >= 0
        if self.args.model['use_mlm']:
            mlm_loss, state_sel = [], [23]
            for si,state in enumerate(plm_outs.hidden_states[1:]):
                if state_sel and si not in state_sel: continue
                logits = self.plm_model.lm_head(state[flow_mask])
                mlm_loss.append(self.loss_ce(logits, inputs['emo_flow_token_label'][label_mask]))
        else: mlm_loss = 0

        
        if self.args.model['use_rnn']:
            mask_features_total = plm_outs.last_hidden_state[flow_mask]
            flow_mask_len = flow_mask.sum(dim=-1)
            mask_features = [mask_features_total[flow_mask_len[:i].sum():flow_mask_len[:i].sum()+l] for i,l in enumerate(flow_mask_len)]
            #mask_features = [fea[-self.params.flow_num:] for fea in mask_features]
            #flow_mask_len = torch.tensor([len(fea) for fea in mask_features])
            mask_features = pad_sequence(mask_features, batch_first=True, padding_value=0)
            flow_features, _ = self.lstm(mask_features, flow_mask_len)
            features = torch.stack([flow_features[idx][pos-1] for idx, pos in enumerate(flow_mask_len)])
            features = (features + plm_outs.pooler_output)/2
        else:
            features = plm_outs.pooler_output

        rate = list(range(len(mlm_loss)+1))[1:]
        rate = [v/sum(rate) for v in rate]
        return features, sum([s*r for s,r in zip(mlm_loss, rate)])
        
    def forward(self, inputs, stage='train'):
        ## 1. encoding 
        features, loss_mlm = self.encode(inputs, stage=stage)
        logits = self.classifier(features)
        preds = torch.argmax(logits, dim=-1).cpu()
        loss = self.loss_ce(logits, inputs['label'])

        ## 2. constraints
        if stage=='train':
            if self.args.model['use_mlm']:
                loss = loss*(1-self.weight) + loss_mlm*self.weight
            if self.args.model['use_cl']:
                loss = loss + self.cl_loss(features, inputs)

        mask = inputs['label'] >= 0
        return {
            # 'fea': features,
            'loss':   loss if mask.sum() > 0 else torch.tensor(0.0).to(loss.device),
            'logits': logits,
            'preds':  preds[mask.cpu()],
            'labels': inputs['label'][mask],
        }
    
    def get_features(self, index, features, method='fetch'):
        # 根据 index 存储/获得 实例表示向量
        for bi, idx in enumerate(index):
            if method == 'store': # 存储表示
                self.bank[idx] = features[bi].detach().cpu()
            if method == 'fetch': # 取出表示
                features.append(torch.stack([fea for fea in self.bank[idx.cpu().numpy()] if fea is not None]).mean(dim=0))
        
        assert len(features) == len(index)
        return features
    
    def cl_loss(self, features, inputs):
        index, labels = inputs['index'], inputs['label']
        # 存储样本表示
        self.get_features(inputs['index'], features=features, method='store') # 存储表示
        # 获取检索样本表示
        ret_features = self.get_features(inputs['ret_idx'], features=[], method='fetch')    
        ret_features = self.dropout(torch.stack(ret_features).type_as(features))
        
        # Contrast Learning 
        eye_mask = torch.eye(features.shape[0]).type_as(features)
        con_features = torch.cat([ret_features, features]) # 原来的batch也要in-batch负采样
        sample_mask = torch.cat([torch.zeros_like(eye_mask), eye_mask], dim=-1) # 定位原本feature位置

        ## 正样本：检索正样本(1个); 负样本：检索负样本(若干)+in-batch负样本(若干)
        labels_cl = torch.arange(features.size(0)).type_as(labels)
        # loss_cl = self.contrast_single(features_tar, features_scl, labels_scl)
        sim_tar_all = self.conl(features, con_features).squeeze(1)
        sim_tar_all = sim_tar_all - sample_mask*1e8
        loss_cl = self.loss_ce(sim_tar_all, labels_cl)
        
        return loss_cl/(features.shape[-1]**0.25)