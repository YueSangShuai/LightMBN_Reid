import data_v1
import data_v2
from loss import make_loss
from model import make_model
from optim import make_optimizer, make_scheduler

# import engine_v1
# import engine_v2
import engine_v3
import os.path as osp
from option import args
import utils.utility as utility
from utils.model_complexity import compute_model_complexity
from torch.utils.collect_env import get_pretty_env_info
import yaml
import torch


args.eval_batch_size=128
args.config="/data/yuesang/Reid/LightMBN/myselff/myself.yaml"
args.pre_train="/data/yuesang/Reid/LightMBN/myselff/lmbn_n_cuhk03_d.pth"
if args.config != "":
    with open(args.config, "r") as f:
        config = yaml.full_load(f)
    for op in config:
        setattr(args, op, config[op])



ckpt = utility.checkpoint(args)
loader = data_v2.ImageDataManager(args)
model = make_model(args, ckpt)
if args.pre_train != "":
    ckpt.load_pretrained_weights(model, args.pre_train)
optimzer = make_optimizer(args, model)
loss = None
start = -1
if args.load != "":
    start, model, optimizer = ckpt.resume_from_checkpoint(
        osp.join(ckpt.dir, "model-latest.pth"), model, optimzer
    )
    start = start - 1
if args.pre_train != "":
    ckpt.load_pretrained_weights(model, args.pre_train)

scheduler = make_scheduler(args, optimzer, start)

engine = engine_v3.Engine(args, model, optimzer, scheduler, loss, loader, ckpt)

engine.test_myself()
