
## cuda environment
import warnings, os, wandb, yaml, sys
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM']='false'

## import packages
from global_var import *
sys.path.append(utils_dir)

from config import config
from writer import JsonFile
from processor import Processor
from utils_processor import set_rng_seed

def run(args):
    if args.train['wandb']:
        wandb.init(
            project=f"project: {'-'.join(args.train['tasks'])}",
            name=f"{'-'.join(args.train['tasks'])}-seed-{args.train['seed']}",
        )
    set_rng_seed(args.train['seed']) # 固定随机种子

    # import model and dataset
    from Model_EmoTrans import import_model
    model, dataset = import_model(args)

    # train or eval the model
    processor = Processor(args, model, dataset)
    if args.train['inference']:
        processor.loadState()
        result = processor._evaluate(stage='test')
    else: result = processor._train()
    if args.train['wandb']: wandb.finish()

    ## 2. output results
    record = {
        'params': {
            'e':       args.train['epochs'],
            'es':      args.train['early_stop'],
            'lr':      args.train['learning_rate'],
            'lr_pre':  args.train['learning_rate_pre'],
            'bz':      args.train['batch_size'],
            'dr':      args.model['drop_rate'],
            'seed':    args.train['seed'],
            # 'ekl':     args.model['ekl'],
        },
        'metric': {
            'stop':    result['valid']['epoch'],
            'tr_mf1':  result['train']['f1'],
            'tv_mf1':  result['valid']['f1'],
            'te_mf1':  result['test']['f1'],
        },
    }
    return record


if __name__ == '__main__':
    args = config(task='', dataset='meld', framework=None, model='emotrans')

    ## 导入配置文件
    with open(f"./configs/{args.model['name']}.yaml", 'r') as f:
        run_config = yaml.safe_load(f)
    args.train.update(run_config['train'])
    args.model.update(run_config['model'])
    args.logger['display'].extend(['arch', 'scale', 'weight'])
    
    seeds = [2024,2025,2026]
    if seeds or args.train['inference']: # 按指定 seed 执行
        if not seeds: seeds = [args.train['seed']]
        recoed_path = f"{args.file['record']}{args.model['name']}_best.jsonl"
        record_show = JsonFile(recoed_path, mode_w='a', delete=True)
        for seed in seeds:
            args.train['seed'] = seed
            record = run(args)
            record_show.write(record, space=False) 


    # seeds = []
    # if seeds or args.train['inference']: # 按指定 seed 执行
    #     if not seeds: seeds = [args.train['seed']]
    #     recoed_path = f"{args.file['record']}{args.model['name']}_best.jsonl"
    #     record_show = JsonFile(recoed_path, mode_w='a', delete=True)
    #     for seed in seeds:
    #         args.train['seed'] = seed
    #         record = run(args)
    #         record_show.write(record, space=False) 
    # else: # 随机 seed 执行       
    #     recoed_path = f"{args.file['record']}{args.model['name']}_search.jsonl"
    #     record_show = JsonFile(recoed_path, mode_w='a', delete=True)
    #     for c in range(100):
    #         args.train['seed'] = random.randint(1000,9999)+c
    #         record = run(args)
    #         record_show.write(record, space=False)