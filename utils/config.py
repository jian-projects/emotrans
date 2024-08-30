import argparse, torch, os, sys
from datetime import datetime
from loguru import logger
from global_var import *

class Arguments():
    def __init__(self):
        self.file = {}

def config(task, dataset, model, opt=None, framework=None):
    if opt: dataset = opt.dataset
    args = Arguments()
    ## parameters for output
    args.file = {
        'plm_dir': plm_dir, # 预训练模型路径
        'data_dir': data_dir+f"{task}/", # 数据路径
        'save_dir': save_dir+f'{task}_{dataset}/',
        'cache_dir': cache_dir+f'{task}_{dataset}/', # 缓存路径, 加快加载

        'log': f'./logs/{dataset}/',
        'record': f'./records/{dataset}/',
    }
    sys.path.append(args.file['data_dir']) # 添加完整数据路径
    if not os.path.exists(args.file['log']): os.makedirs(args.file['log']) # 创建日志路径
    if not os.path.exists(args.file['record']): os.makedirs(args.file['record']) # 创建记录路径
    if not os.path.exists(args.file['save_dir']): os.makedirs(args.file['save_dir']) # 创建保存路径
    if not os.path.exists(args.file['cache_dir']): os.makedirs(args.file['cache_dir']) # 创建缓存路径

    ## parameters for training
    args.train = {
        'show': False,
        'tasks': [task, dataset],
        # 'models': [framework, model],

        'e_start': 0, # start epoch
        'epochs': opt.epochs if opt else 64,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'device_ids': [0],
        'do_test': True,
        'do_valid': True,
        'do_train': True,
        'early_stop': opt.early_stop if opt else 8,
        'save_model': False,
        'log_step_rate': 1.0, # 每个epoch将进行评价次数
        'log_step_rate_max': 3.0,

        'seed': 2024,
        'l2reg': 0.01,
        'data_rate': 1.0,
        'batch_size': opt.batch_size if opt else 32,
        'warmup_ratio': 0.3,
        'weight_decay': 1e-3,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 5.0,
        'learning_rate': opt.learning_rate if opt else 1e-4,
        'learning_rate_pre': opt.learning_rate_pre if opt else 1e-4,
    }

    ## parameters for model
    args.model = {
        'name': model,
        'framework': framework,
        'drop_rate': 0.3,
        'epoch_before': False,
        'epoch_every': False,
        'epoch_after': False,
    }

    ## logging
    logger.remove() # 不要在控制台输出日志
    handler_id = logger.add(sys.stdout, level="WARNING") # WARNING 级别以上的日志输出到控制台
    logDir = os.path.expanduser(args.file['log']+datetime.now().strftime("%Y%m%d_%H%M%S")) # 日志文件夹
    if not os.path.exists(logDir): os.makedirs(logDir) # 创建日志文件夹
    # logger.add(os.path.join(logDir,'loss.log'), filter=lambda record: record["extra"].get("name")=="loss") # 添加配置文件 loss
    # logger.add(os.path.join(logDir,'metric.log'), filter=lambda record: record["extra"].get("name")=="metric") # 添加配置文件 metric
    logger.add(os.path.join(logDir,'params.log'), filter=lambda record: record["extra"].get("name")=="params") # 添加配置文件 metric
    logger.add(os.path.join(logDir,'process.log'), filter=lambda record: record["extra"].get("name")=="process") # 添加配置文件 metric
    args.logger= {
        # 'loss': logger.bind(name='loss'), 
        # 'metric': logger.bind(name='metric'),
        'params': logger.bind(name='params'),
        'process': logger.bind(name='process'),  
    } # 日志记录器

    ## 展示部分参数
    args.logger['display'] = ['epochs', 'early_stop', 'batch_size', 'learning_rate_pre', 'seed']
    return args