import torch, time, math
from tqdm import tqdm


def init_weight(model, method='xavier_uniform_'):
    if method == 'xavier_uniform_': fc = torch.nn.init.xavier_uniform_
    if method == 'xavier_normal_':  fc = torch.nn.init.xavier_normal_
    if method == 'orthogonal_':     fc = torch.nn.init.orthogonal_

    for name, param in model.named_parameters():
        if 'plm' not in name: # 跳过 plm 模型参数
            if param.requires_grad:
                if len(param.shape) > 1: fc(param) # 参数维度大于 1
                else: 
                    stdv = 1. / math.sqrt(param.shape[0])
                    torch.nn.init.uniform_(param, a=-stdv, b=stdv)

def print_trainable_parameters(args, model):
    params_all, params_train = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"): num_params = param.ds_numel
        params_all += num_params
        if param.requires_grad: params_train += num_params
    
    p_train, p_all = f"{round(params_train/1000000, 2)} M", f"{round(params_all/1000000, 2)} M"
    train_rate = round(100*params_train/params_all, 2)

    args.logger['process'].warning(f"train: {p_train} || all params: {p_all} || trainable: {train_rate} %")

class Processor():
    def __init__(self, args, model, dataset) -> None:
        self.args = args
        self.dataset = dataset
        self.model = model.to(args.train['device'])
        init_weight(self.model) # 初始化模型参数
        print_trainable_parameters(args, self.model) # 打印训练参数比重

        if self.dataset.loader: self.dataloader = self.dataset.loader
        else: self.dataloader = self.dataset.get_dataloader(self.args.train['batch_size'])
        self.log_step_rate = args.train['log_step_rate']
        self.global_step = 1
        self.log_step = int(len(self.dataloader['train']) / self.log_step_rate)
        self.model.get_optimizer() # 初始化优化器

        for k, v in vars(args).items():
            for kk, vv in v.items(): args.logger['params'].info(f"{k}.{kk}: {vv}")
        args.logger['params'].info(f"\n {'='*120} \n")

        display = ''
        for item in args.logger['display']: 
            if item in args.train: display += f"{item}: {args.train[item]}, "
            if item in args.model: display += f"{item}: {args.model[item]}, "
        args.logger['process'].warning(display)

    def train_desc(self, epoch, ttime=None):
        args, metrics = self.args, self.dataset.metrics.results
        epochs, model_name, data_name = args.train['epochs'], args.model['name'], self.dataset.name[-1]
        m = self.dataset.metrics.base
        m_tr, m_vl, m_te = round(metrics['train'][m], 3), round(metrics['valid'][m], 3), round(metrics['test'][m], 3)
        m_tr_loss = round(metrics['train']['loss'], 3)
        desc = f"eh {epoch}/{epochs} ({model_name}=>{data_name}: {str(m_tr)}/{str(m_vl)}/{str(m_te)}, loss: {str(m_tr_loss)}, time: {ttime})"
        self.tqdm_epochs.set_description(desc)
        if epoch>=0: self.tqdm_epochs.update()

    def train_stop(self, epoch=None):
        metric_valid = self.dataset.metrics.results['valid']
        early_threshold = epoch-metric_valid['epoch'] if 'epoch' in metric_valid else 0

        # 0. 达到阈值，停止训练
        if early_threshold >= self.args.train['early_stop']:
            return True
        
        # 1. 长期未更新了，增加评价次数
        if early_threshold: 
            self.log_step_rate = self.args.train['log_step_rate']+early_threshold*0.5
            self.log_step_rate = min(self.log_step_rate, 3.0)
        else: self.log_step_rate = self.args.train['log_step_rate']

    def train_batch(self, batch, bi=None):
        self.model.train() 
        if isinstance(batch, dict):     
            for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
        if isinstance(batch, list):
            for i, val in enumerate(batch): batch[i] = val.to(self.args.train['device'])
        outs = self.model.training_step(batch, bi)  
        
        outs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.train['max_grad_norm'])
        self.model.optimizer.step()
        if self.model.scheduler is not None: self.model.scheduler.step() 
        self.model.optimizer.zero_grad()        

        self.global_step += 1
        if self.global_step % self.log_step == 0:
            if self.args.train['do_valid']: self._evaluate(stage='valid')
            if self.args.train['do_test'] and self.model.valid_update: self._evaluate(stage='test')

    def _train(self):
        epochs, e_start = self.args.train['epochs'], self.args.train['e_start'] if 'e_start' in self.args.train else 0
        self.tqdm_epochs = tqdm(total=epochs, position=0) # 进度条
        self.tqdm_epochs.update(e_start)
        self.train_desc(epoch=-1) # initialize process bar
        if self.args.model['epoch_before']: self.model.epoch_deal()
        for epoch in range(e_start, epochs):
            s_time = time.time()
            self.model.cur_epoch = epoch
            if self.args.model['epoch_every']: self.model.epoch_deal(epoch)
            
            torch.cuda.empty_cache()
            if self.args.train['show']: # 显示每个epoch的进度条
                for batch in tqdm(self.dataloader['train'], smoothing=0.05):
                    self.train_batch(batch, bi=-1)
            else: 
                for bi, batch in enumerate(self.dataloader['train']):
                    self.train_batch(batch, bi)
            
            if self.args.model['epoch_after']: self.model.epoch_deal(epoch)
            self.model.on_train_epoch_end()

            self.train_desc(epoch, round(time.time()-s_time, 1))
            if self.train_stop(epoch): break 
            
        self.tqdm_epochs.close()
        return self.dataset.metrics.results

    def _evaluate(self, stage='test'):
        # for bi, batch in enumerate(self.dataloader[stage]):
        #     with torch.no_grad():
        #         for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
        #         if stage == 'valid': self.model.validation_step(batch, bi)
        #         if stage == 'test': self.model.test_step(batch, bi)
            
        # if stage == 'valid': self.model.on_validation_end()
        # if stage == 'test': self.model.on_test_end()
        # return self.dataset.metrics
        self.model.eval()
        with torch.no_grad():
            if self.args.train['show']: # 显示每个epoch的进度条
                for batch in tqdm(self.dataloader[stage], smoothing=0.05):
                    if isinstance(batch, dict):
                        for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
                    if isinstance(batch, list):
                        for i, val in enumerate(batch): batch[i] = val.to(self.args.train['device'])
                    if stage == 'valid': self.model.validation_step(batch, -1)
                    if stage == 'test': self.model.test_step(batch, -1)
            else:
                for bi, batch in enumerate(self.dataloader[stage]):
                    if isinstance(batch, dict):
                        for key, val in batch.items(): batch[key] = val.to(self.args.train['device'])
                    if isinstance(batch, list):
                        for i, val in enumerate(batch): batch[i] = val.to(self.args.train['device'])
                    if stage == 'valid': self.model.validation_step(batch, bi)
                    if stage == 'test': self.model.test_step(batch, bi)
            
        if stage == 'valid': self.model.on_validation_end()
        if stage == 'test': self.model.on_test_end()
        return self.dataset.metrics.results