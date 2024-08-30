import json, torch
from typing import Dict
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl 
from utils_processor import get_optimizer, get_scheduler


class PoolerAll(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class ModelForClassification(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.cur_epoch = 0

    def forward(self, inputs, stage='train'):
        raise NotImplementedError

    def get_optimizer(self):
        iter_total = int(len(self.dataset.loader['train'])*self.args.train['epochs'])
        self.optimizer = get_optimizer(self)
        self.scheduler = get_scheduler(self.args, self.optimizer, iter_total)

    def configure_optimizers(self):
        optimizer = get_optimizer(self)
        return {'optimizer': optimizer}

        scheduler = get_scheduler(self.args, optimizer, len(self.dataset.train_dataloader()))
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict
    
        # weight_decay = 1e-6  # l2正则化系数
        # # 假如有两个网络，一个encoder一个decoder
        # optimizer = optim.Adam([{'encoder_params': self.encoder.parameters()}, {'decoder_params': self.decoder.parameters()}], lr=learning_rate, weight_decay=weight_decay)
        # # 同样，如果只有一个网络结构，就可以更直接了
        # optimizer = optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # # 我这里设置2000个epoch后学习率变为原来的0.5，之后不再改变
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.5)
        # optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        # return optim_dict
    
        # return torch.optim.AdamW(self.parameters(), lr=self.args.train['learning_rate'])

    def training_step(self, batch, batch_idx):
        output, cur_e = self(batch, stage='train'), self.cur_epoch,
        self.training_step_outputs.append(output)

        return {
            'loss': output['loss']
        }
    
    def on_train_epoch_end(self):
        outputs, metrics_tr = self.training_step_outputs, self.dataset.metrics.results['train']
        metrics_tr.update(self.dataset.metrics._score(outputs, stage='train'))
        metrics_tr['epoch'] = self.cur_epoch
        
        self.training_step_outputs = [] # init record
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_tr.items()})
        self.args.logger['process'].info(f"train_eval: {describe}")
    
    def validation_step(self, batch, batch_idx):
        output = self(batch, stage='valid')
        self.validation_step_outputs.append(output)

        return output

    def on_validation_end(self):
        outputs, metrics_vl = self.validation_step_outputs, self.dataset.metrics.results['valid']
        metrics = self.dataset.metrics._score(outputs, stage='valid')

        ## update best model
        mark, self.valid_update = self.dataset.metrics.base, False
        if metrics[mark] > metrics_vl[mark]: # bigger is better
            metrics_vl.update(metrics)
            metrics_vl['epoch'] = self.cur_epoch
            describe = json.dumps({k: round(float(v),4) for k,v in metrics_vl.items()})
            self.args.logger['process'].info(f"valid: {describe}")
            self.valid_update = True # execute test

            if self.args.train['save_model']: 
                if self.args.model['use_adapter']: 
                    self.save_checkpoint_peft()
                else: self.save_checkpoint() # 保存模型

        self.validation_step_outputs = [] # init record

    def save_checkpoint(self, save_path=None, mode='save'):
        if save_path is None: save_path = self.args.model['save_path']
        if mode == 'save':
            state = {
                'net': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if mode == 'load':
            state = torch.load(save_path)
            self.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optimizer'])

    def save_checkpoint_peft(self, save_path=None, mode='save'):
        if save_path is None: save_path = self.args.model['save_path']
        if mode == 'save':
            training_params_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
            state = {
                'net': training_params_dict,
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if mode == 'load':
            state = torch.load(save_path)
            self.load_state_dict(state['net'], strict=False)
            self.optimizer.load_state_dict(state['optimizer'])

    def test_step(self, batch, batch_idx):
        output = self(batch, stage='test')
        self.test_step_outputs.append(output)

    def on_test_end(self):
        outputs, metrics_te = self.test_step_outputs, self.dataset.metrics.results['test']
        metrics_te.update(self.dataset.metrics._score(outputs, stage='test'))
        metrics_te['epoch'] = self.cur_epoch
        
        self.test_step_outputs = []
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_te.items()})
        self.args.logger['process'].info(f"test: {describe}")


class ModelForGeneration(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.cur_epoch = 0

    def forward(self, inputs, stage='train'):
        raise NotImplementedError

    def get_optimizer(self):
        iter_total = int(len(self.dataset.loader['train'])*self.args.train['epochs'])
        self.optimizer = get_optimizer(self)
        self.scheduler = get_scheduler(self.args, self.optimizer, iter_total)

    def configure_optimizers(self):
        optimizer = get_optimizer(self)
        return {'optimizer': optimizer}

        scheduler = get_scheduler(self.args, optimizer, len(self.dataset.train_dataloader()))
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict
    
        # weight_decay = 1e-6  # l2正则化系数
        # # 假如有两个网络，一个encoder一个decoder
        # optimizer = optim.Adam([{'encoder_params': self.encoder.parameters()}, {'decoder_params': self.decoder.parameters()}], lr=learning_rate, weight_decay=weight_decay)
        # # 同样，如果只有一个网络结构，就可以更直接了
        # optimizer = optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # # 我这里设置2000个epoch后学习率变为原来的0.5，之后不再改变
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.5)
        # optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        # return optim_dict
    
        # return torch.optim.AdamW(self.parameters(), lr=self.args.train['learning_rate'])

    def training_step(self, batch, batch_idx):
        output, cur_e = self(batch, stage='train'), self.cur_epoch,
        self.training_step_outputs.append(output)

        return {
            'loss': output['loss']
        }
    
    def on_train_epoch_end(self):
        outputs, metrics_tr = self.training_step_outputs, self.dataset.metrics['train']
        # metrics = self.dataset._score(outputs, stage='train') # 训练时不需要评价
        # metrics_tr.update(metrics)
        metrics_tr['loss'] = np.mean([o['loss'].item() for o in outputs])
        metrics_tr['epoch'] = self.cur_epoch
        if self.dataset.met not in metrics_tr: metrics_tr[self.dataset.met] = 0

        self.training_step_outputs = [] # init record
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_tr.items()})
        self.args.logger['process'].info(f"train_eval: {describe}")
    
    def validation_step(self, batch, batch_idx):
        output = self._eval(batch, stage='valid') # 要按生成方式
        self.validation_step_outputs.append(output)

        return output

    def on_validation_end(self):
        outputs, metrics_vl = self.validation_step_outputs, self.dataset.metrics['valid']
        if self.dataset.met == 'loss':
            if None in outputs: metrics = {'loss': -1e3}
            else: metrics = {'loss': -round(np.array([o['outputs']['loss'].item() for o in outputs]).mean(), 2)}
        else:
            if None in outputs: metrics = {self.dataset.met: 0}
            else: metrics = self.dataset._score(outputs, stage='valid')

        ## update best model
        mark, self.valid_update = self.dataset.met, False
        if metrics[mark] > metrics_vl[mark]: # bigger is better
            metrics_vl.update(metrics)
            metrics_vl['epoch'] = self.cur_epoch
            describe = json.dumps({k: round(float(v),4) for k,v in metrics_vl.items()})
            self.args.logger['process'].info(f"valid: {describe}")

            self.valid_update = True # execute test
            if self.args.train['save_model']: self.save_checkpoint() # 需要保存模型

        self.validation_step_outputs = [] # init record

    def test_step(self, batch, batch_idx):
        output = self._eval(batch, stage='test') # 要按生成方式
        self.test_step_outputs.append(output)

    def on_test_end(self):
        outputs, metrics_te = self.test_step_outputs, self.dataset.metrics['test']
        metrics = self.dataset._score(outputs, stage='test')
        metrics_te.update(metrics)
        metrics_te['epoch'] = self.cur_epoch
        
        self.test_step_outputs = []
        describe = json.dumps({k: round(float(v),4) for k,v in metrics_te.items()})
        self.args.logger['process'].info(f"test: {describe}")
    

    def save_checkpoint(self, save_path=None, mode='save'):
        ## 1. if use adapter, save the trainable parameters
        if 'use_adapter' in self.args.model and self.args.model['use_adapter']:
            return self.save_checkpoint_peft(save_path, mode)
            
        ## 2. save the whole model parameters
        if save_path is None: save_path = self.args.model['save_path']
        if mode == 'save':
            state = {
                'net': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if mode == 'load':
            state = torch.load(save_path)
            self.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optimizer'])

    def save_checkpoint_peft(self, save_path=None, mode='save'):
        if save_path is None: save_path = self.args.model['save_path']
        if mode == 'save':
            trainable_params_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
            state = {
                'net': trainable_params_dict,
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if mode == 'load':
            state = torch.load(save_path)
            self.load_state_dict(state['net'], strict=False)
            self.optimizer.load_state_dict(state['optimizer'])