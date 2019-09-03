# -*- coding=utf-8 -*-
"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/UMID_REAL_400_noMalay_parall.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true") # 注意这里的用法，action 表示的是当命令中出现--resume的时候状态就置为True,
# 所以在sh脚本中不需要指定resume=True(否则会报错), 直接在sh里写上--resume就可以了。
# https://blog.csdn.net/liuweiyuxiang/article/details/82918911
parser.add_argument('--gpus', type=str, default='0', help="gpu ids default = 0")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

# https://shiyaya.github.io/2019/04/03/torch-backends-cudnn-benchmark-true-%E4%BD%BF%E7%94%A8%E6%83%85%E5%BD%A2/
cudnn.benchmark = True   # 在合适的条件下可以提高训练效率

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

# multigpus
# trainer = torch.nn.DataParallel(trainer).cuda()

# 处理gpu ids
def _get_set_gpus(gpuidsstr):
    # get gpu ids
    str_ids = gpuidsstr.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    print('---------------- Gpu ID -----------------  ', gpu_ids)

    # set gpu ids
    # if len(gpu_ids) > 0:
    #     torch.cuda.set_device(gpu_ids[0])
    return gpu_ids

print('ops.gpu', opts.gpus)

gpuids = _get_set_gpus(opts.gpus)
# print('gpuids', gpuids)
if len(gpuids) > 1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = torch.nn.DataParallel(trainer, device_ids=gpuids)
    # trainer.cuda()
    trainer.to(device)
else:
    # os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpus
    trainer.cuda()

print('displaysize: ', display_size)
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
print('dataset size: ', len(train_loader_a.dataset))
train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
# https://blog.csdn.net/zzc15806/article/details/81352742
# os.path.splitext 路径的文件名和扩展名分开
model_name = os.path.splitext(os.path.basename(opts.config))[0]
# 指定tensorboard 的存储路径
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
# iterations = trainer.resume(checkpoint_directory, hyperparameters=config)
while True:
    # zip(a,b) 返回的是一个list，并且list中每一个元素就是一个元组。list 的长度是a,b中最短的长度。
    #(0,(a,b))
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()  # 先更新学习率
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code  先训练的是鉴别器
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            # Waits for all kernels in all streams on a CUDA device to complete.
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                # test_image_outputs 是32个channel的
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations, gpuids)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

