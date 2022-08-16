#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Prudhvi Thirumalaraju   Line 3
# Created Date: 07/22/2022
# version ='1.0'
# ---------------------------------------------------------------------------
""" Training scripts """
# ---------------------------------------------------------------------------


import argparse
import copy
import csv
import os
import os.path as osp
import statistics
import time
from collections import OrderedDict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import helper_utils.network as network
# import helper_utils.pre_process as prep
import helper_utils.pre_process_old as prep
import yaml

from torch.utils.data import DataLoader
import helper_utils.lr_schedule as lr_schedule
from helper_utils.data_list_m import ImageList

import argparse

from helper_utils.logger import Logger
from helper_utils.sampler import ImbalancedDatasetSampler

from helper_utils.EarlyStopping import EarlyStopping
from helper_utils.tools_testing import testing_sperm_slides, validation_loss, calc_transfer_loss, Entropy







def data_setup(config):
    #set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["test"] = prep.image_test(**config["prep"]['params_test'])
    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    test_bs = data_config["test"]["batch_size"]
    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                              transform=prep_dict["test"], labelled=data_config["test"]["labelled"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=4)

    return dset_loaders


def network_setup(config):
    class_num = config["network"]["params"]["class_num"]
    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
    return base_network, schedule_param, lr_scheduler, optimizer


def test(config, dset_loaders, model_path_for_testing=None):
    if model_path_for_testing:
        model = torch.load(model_path_for_testing)
    else:

        model = torch.load(osp.join(config["model_path"], "best_model.pth.tar"))



    test_info = validation_loss(dset_loaders, model, data_name='test',dset=config['dataset'],
                                num_classes=config["network"]["params"]["class_num"],
                                logs_path=config['logs_path'], is_training=config['is_training'])

    print_msg("Final Model " + "| Test loss: " + str(test_info['val_loss']) + str("| Test Accuracy: ") +
              str(test_info['val_accuracy']) + (
                  "| 2 class acc:" + str(test_info['val_acc_2_class']) if 'val_acc_2_class' in test_info else ""),
              config["out_file"])

def print_msg(msg, outfile):
    print()
    print("=" * 50)
    print("" * 2, msg)
    print("=" * 50)
    print()

    outfile.write('\n')
    outfile.write("=" * 25)
    outfile.write(" " * 5 + msg)
    outfile.write("=" * 25)
    outfile.write('\n')
    outfile.flush()

def parge_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['train', 'test'])

    parser.add_argument('--seed', type=int)
    parser.add_argument('--dset', type=str, help="The dataset or source dataset used")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")

    parser.add_argument('--lr', type=float)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--power', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--nesterov', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--batch_size_test', type=int)
    parser.add_argument('--use_bottleneck', type=bool)
    parser.add_argument('--bottleneck_dim', type=int)

    parser.add_argument('--new_cls', type=bool)
    parser.add_argument('--no_of_classes', type=int)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--crop_size', type=int)

    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--test_interval', type=int)
    parser.add_argument('--snapshot_interval', type=int)

    parser.add_argument('--trained_model_path', type=str)
    parser.add_argument('--no_of_layers_freeze', type=int)

    parser.add_argument('--s_dset', type=str)

    parser.add_argument('--test_dset_txt', type=str)
    parser.add_argument('--s_dset_txt', type=str)
    parser.add_argument('--sampling', type=str)
    parser.add_argument('--sv_dset_txt', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.set_defaults(
        mode="test",
        seed=0,
        gpu_id="1",
        dset="Bias_study",
        s_dset_txt="Sperm Quality (1=Great, 4=Poor)",
        sampling= "stratfied",
        s_dset="D",
        lr=0.0001,
        arch="Xception",
        gamma=0.0001,
        power=0.75,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
        optimizer="SGD",
        batch_size=16,
        batch_size_test=128,
        use_bottleneck=False,
        bottleneck_dim=256,
        new_cls=True,
        no_of_classes=2,
        image_size=256,
        crop_size=224,
        trained_model_path= None,
        no_of_layers_freeze=13,
        num_iterations=200000,
        patience=500,
        test_interval=1,
        snapshot_interval=1,
        output_dir="../../reports"
    )

    args = parser.parse_args()
    return args


def set_deterministic_settings(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ####################################
    # Default Project Folders#
    ####################################


    project_root = "../../"
    data_root = project_root + "data/"
    models_root = project_root + "models/"

    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    timestamp = timestamp.replace("/", "_").replace(" ", "_").replace(":", "_").replace(".", "_")

    args = parge_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    set_deterministic_settings(seed=args.seed)
    lr = args.lr
    opt = args.optimizer
    dataset = args.dset
    trial_number = args.mode + "_" + args.s_dset_txt
    log_output_dir_root = args.output_dir + '/logs/'+args.sampling +"/"+ trial_number + '/' + str(lr)+"_seed"+str(args.seed)+"_"+ str(opt)+"_"+ args.arch + '/'
    # models_output_dir_root = args.output_dir + '/models/'+args.sampling +"/" + trial_number + '/' + str(lr)+"_seed"+str(args.seed)+"_"+ str(opt)+"_"+ args.arch + '/'

    # print(os.listdir(project_root))
    if args.mode == "train":
        is_training = True
    else:
        is_training = False

    config = {}
    no_of_classes = args.no_of_classes



    ####################################
    # Dataset Locations Setup #
    ####################################

    train_path = args.s_dset_txt
    sampling = args.sampling



    source_input = {'path': "../../data/txt_files/"+ sampling+ '/'+train_path +"/Train.txt"}
    source_valid_input = {'path': "../../data/txt_files/"+sampling+ "/"+train_path +"/Val.txt"}
    test_input = {'path': "../../data/txt_files/"+sampling+ "/"+train_path +"/Test.txt", 'labelled': True}



    model_path_for_testing = args.trained_model_path

    config['timestamp'] = timestamp
    config['trial_number'] = trial_number
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["patience"] = args.patience
    config["is_training"] = is_training

    if not is_training:
        config["num_iterations"] = 0
        best_itr = "testing"
        print("Testing:")
        config["best_itr"] = "testing"

    print("num_iterations", config["num_iterations"])

    log_output_path = log_output_dir_root
    # trial_results_path = models_output_dir_root
    # config["model_path"] = trial_results_path
    config["logs_path"] = log_output_path
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])



    config["out_file"] = open(osp.join(config["logs_path"], "log.txt"), "w")
    config["out_file2"] = open(osp.join(config["logs_path"], "log_best.txt"), "w")
    resize_size = args.image_size

    config["prep"] = {'params_source': {"resize_size": resize_size, "crop_size": args.crop_size, "dset": dataset},
                      'params_test': {"resize_size": resize_size, "crop_size": args.crop_size, "dset": dataset}}

    config["loss"] = {"trade_off": 1.0}
    config["trained_model_path"] = args.trained_model_path
    config['no_of_layers_freeze'] = args.no_of_layers_freeze

    if "Xception" in args.arch:
        config["network"] = \
            {"name": network.XceptionFc,
             "params":
                 {
                     "use_bottleneck": args.use_bottleneck,
                     "bottleneck_dim": args.bottleneck_dim,
                     "new_cls": args.new_cls}}
    elif "ResNet50" in args.arch:
        config["network"] = {"name": network.ResNetFc,
                             "params":
                                 {"resnet_name": args.arch,
                                  "use_bottleneck": args.use_bottleneck,
                                  "bottleneck_dim": args.bottleneck_dim,
                                  "new_cls": args.new_cls}}

    elif "Inception" in args.arch:
        config["network"] = {"name": network.Inception3Fc,
                             "params":
                                 {"use_bottleneck": args.use_bottleneck,
                                  "bottleneck_dim": args.bottleneck_dim,
                                  "new_cls": args.new_cls}}

    if args.optimizer == "SGD":

        config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": args.momentum,
                                                                   "weight_decay": args.weight_decay,
                                                                   "nesterov": args.nesterov},
                               "lr_type": "inv",
                               "lr_param": {"lr": args.lr, "gamma": args.gamma, "power": args.power}}

    elif args.optimizer == "Adam":
        config["optimizer"] = {"type": optim.Adam, "optim_params": {'lr': args.lr,
                                                                    "weight_decay": args.weight_decay},
                               "lr_type": "inv",
                               "lr_param": {"lr": args.lr, "gamma": args.gamma, "power": args.power}}

    config["dataset"] = dataset
    config["data"] = {"source": {"list_path": source_input['path'], "batch_size": args.batch_size},
                      "test": {"list_path": test_input['path'], "batch_size": args.batch_size_test,
                               "labelled": test_input['labelled']},
                      "valid_source": {"list_path": source_valid_input['path'], "batch_size": args.batch_size}}
    config["optimizer"]["lr_param"]["lr"] = args.lr
    config["network"]["params"]["class_num"] = no_of_classes

    config["out_file"].write(str(config))
    config["out_file"].flush()
    print("source_path", source_input)
    print("test_path", test_input)
    print('GPU', os.environ["CUDA_VISIBLE_DEVICES"], config["gpu"])

    ####################################
    # Dump arguments #
    ####################################
    with open(config["logs_path"] + "args.yml", "w") as f:
        yaml.dump(args, f)

    dset_loaders = data_setup(config)

    print()
    print("=" * 50)
    print(" " * 15, "Testing Started")
    print("=" * 50)
    print()

    test(config, dset_loaders, model_path_for_testing=model_path_for_testing)


if __name__ == "__main__":
    main()
