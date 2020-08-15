import torch
import argparse
from model import make_model
from data import make_pretrain_dataset, make_finetune_dataset
import pretrain
from opt import OpenAIAdam
from collections import namedtuple
import utils

def main(args,pretrain_setting,finetune_setting):
    ## Build Model and push model to GPU
    device = torch.device('cuda')
    if args.do_pretrain:
        model = make_model()
    else
        model = make_model(finetune_setting.load_model_pth)
    
    model = model.to(device)

    ##Build dataset

    if args.do_pretrain:
        pretrain_dataset = make_pretrain_dataset(pretrain_setting,
                                                pretrain_setting.saved_data_pth, 
                                                pretrain_setting.raw_data_pth,
                                                pretrain_setting.processed_data_pth)
    if args.do_finetune:
        finetune_dataset = make_finetune_dataset(finetune_setting.saved_data)

    if args.do_pretrain:
        num_train_optimization_steps = len(pretrain_dataset) * pretrain_setting.epoch_num // pretrain_setting.batch_size // pretrain_setting.num_accumulation
        optimizer = OpenAIAdam(model.parameters(),
                                lr=1e-5,
                                schedule='warmup_linear',
                                warmup=0.002,
                                t_total=num_train_optimization_steps,
                                b1=0.9,
                                b2=0.999,
                                e=1e-08,
                                l2=0.01,
                                vector_l2=True,
                                max_grad_norm=1)
        pretrain.train(model,
                       dataset = pretrain_dataset,
                       optimizer = optimizer,
                       log_path = pretrain_setting.log_pth,
                       best_model_pth = pretrain_setting.best_model_pth,
                       batch_size=pretrain_setting.batch_size,
                       num_accumulation=pretrain_setting.num_accumulation,
                       epoch_num=pretrain_setting.epoch_num)

    if args.do_finetune:
        num_train_optimization_steps = len(finetune_dataset) * finetune_setting.epoch_num // finetune_setting.batch_size // finetune_setting.num_accumulation
        optimizer = OpenAIAdam(model.parameters(),
                                lr=1e-5,
                                schedule='warmup_linear',
                                warmup=0.002,
                                t_total=num_train_optimization_steps,
                                b1=0.9,
                                b2=0.999,
                                e=1e-08,
                                l2=0.01,
                                vector_l2=True,
                                max_grad_norm=1)

        pretrain.train(model,
                       dataset = finetune_dataset,
                       optimizer = optimizer,
                       log_path = finetune_setting.log_pth,
                       best_model_pth = finetune_setting.best_model_pth,
                       batch_size=finetune_setting.batch_size,
                       num_accumulation=finetune_setting.num_accumulation,
                       epoch_num=finetune_setting.epoch_num)



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--do_pretrain',action='store_true')
    parser.add_argument('--do_finetune',action='store_true')
    parser.add_argument('--pretrain_config_pth',type=str,default='pretrain_config.json')
    parser.add_argument('--finetune_config_pth',type=str,default='finetune_config.json')

    args = parser.parse_args()

    pretrain_setting = utils.load(args.pretrain_config_pth)
    finetune_setting = utils.load(args.finetune_config_pth)

    main(args,pretrain_setting,finetune_setting)

