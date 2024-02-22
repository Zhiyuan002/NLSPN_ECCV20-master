"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    main script for training and testing.
"""
from torch import nn, autograd, optim
import cv2
from config import args as args_config
import time
import random
import os
from model.torch_utils.ops import conv2d_gradfix
from data import ToFDataset
from model.Discriminator import StyleGAN2
from model.diffusion import Diffusion
from data.HyperSim import HyperSim
from itertools import cycle

os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utility
from model import get as get_model
from data import get as get_data
from loss import get as get_loss
from summary import get as get_summary
from metric import get as get_metric

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

# Minimize randomness
torch.manual_seed(args_config.seed)
np.random.seed(args_config.seed)
random.seed(args_config.seed)
torch.cuda.manual_seed_all(args_config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def visualize_tensor(img_tensor, mode, window="Image", depth_min=0, depth_max=1):  # depth_min=0.3, depth_max=0.6
    img = img_tensor.cpu().detach().numpy()

    if mode == 'rgb':
        img = img.transpose((1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if window is not None:
            cv2.imshow(window, img)
    elif mode == 'depth':
        img = img[0]
        img[img == 0] = 1000
        # print(np.max(img),np.min(img))
        # print("o",img)
        img = ((img - depth_min) * 255 / (depth_max - depth_min)).astype(np.uint8)

        # img = ((img - depth_min) * 255 / (depth_max - depth_min))
        # print("a",img)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if window is not None:
            cv2.imshow(window, img)
    elif mode=='hole':
        img = img[0]
        img = ((img - depth_min) * 255 / (depth_max - depth_min)).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    else:
        print("Not support mode!")
        return None
    return img

def check_args(args):
    if args.batch_size < args.num_gpus:
        print("batch_size changed : {} -> {}".format(args.batch_size,
                                                     args.num_gpus))
        args.batch_size = args.num_gpus

    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def train(gpu, args):
    # Initialize workers
    # NOTE : the worker with gpu=0 will do logging
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    # Prepare dataset
    data = get_data(args)

    data_train = data(args, 'train')
    data_val = data(args, 'val')
    csv_file = "./data/selected_indices.csv"
    syn_dataset_root = "/mnt/drive/Dataset/Hypersim"
    data_syn = HyperSim(csv_file, syn_dataset_root)
    # print(len(data_syn))
    sampler_train = DistributedSampler(
        data_train, num_replicas=args.num_gpus, rank=gpu)
    sampler_val = DistributedSampler(
        data_val, num_replicas=args.num_gpus, rank=gpu)
    sampler_syn = DistributedSampler(
        data_syn, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size // args.num_gpus

    loader_train = DataLoader(
        dataset=data_train, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)
    loader_syn = DataLoader(
        dataset=data_syn, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_syn,
        drop_last=True)
    loader_val = DataLoader(
        dataset=data_val, batch_size=1, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_val,
        drop_last=False)

    # Network
    model = get_model(args)
    discr = StyleGAN2(size=32)
    discr = discr.cuda(gpu)
    diffuse = Diffusion().cuda()

    net = model(args)
    net.cuda(gpu)

    if gpu == 0:
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict(checkpoint['net'])

            print('Load network parameters from : {}'.format(args.pretrain))
            # To do: Discriminator

    # Loss
    loss = get_loss(args)
    loss = loss(args)
    loss.cuda(gpu)

    # Optimizer
    optimizer, scheduler = utility.make_optimizer_scheduler(args, net)
    D_optimizer = torch.optim.Adam(list(discr.parameters()), lr=1e-4, betas=(0, 0.99), eps=1e-8)
    net = apex.parallel.convert_syncbn_model(net)
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level,
                                    verbosity=0)

    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    amp.load_state_dict(checkpoint['amp'])

                    print('Resume optimizer, scheduler and amp '
                          'from : {}'.format(args.pretrain))
                except KeyError:
                    print('State dicts for resume are not saved. '
                          'Use --save_full argument')

            del checkpoint

    net = DDP(net)

    metric = get_metric(args)
    metric = metric(args)
    summary = get_summary(args)

    if gpu == 0:
        utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
        except OSError:
            pass

    if gpu == 0:
        writer_train = summary(args.save_dir, 'train', args,
                               loss.loss_name, metric.metric_name)
        writer_val = summary(args.save_dir, 'val', args,
                             loss.loss_name, metric.metric_name)

        with open(args.save_dir + '/args.json', 'w') as args_json:
            json.dump(args.__dict__, args_json, indent=4)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    for epoch in range(1, args.epochs+1):
        # Train
        net.train()

        sampler_train.set_epoch(epoch)

        if gpu == 0:
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            list_lr = []
            for g in optimizer.param_groups:
                list_lr.append(g['lr'])

            print('=== Epoch {:5d} / {:5d} | Lr : {} | {} | {} ==='.format(
                epoch, args.epochs, list_lr, current_time, args.save_dir
            ))

        num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0
        train_batch_count = len(loader_train)
        for batch, sample in enumerate(zip(loader_train, cycle(loader_syn))):

            step = (epoch - 1) * train_batch_count + batch + 1
            sample_syn = sample[1]
            sample = sample[0]
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                      if val is not None}
            sample_syn = {key: val.cuda(gpu) for key, val in sample_syn.items()
                      if val is not None}

            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] \
                                 * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()

            output = net(sample)
            output_clone = output['pred'].clone().detach()
            result_dir = os.path.join(args.save_dir, "intermediate_results")
            os.makedirs(result_dir, exist_ok=True)

            if batch % 5000 == 0:
                result_file_name = "train" + str(int(batch/5000)) + '_epoch' + str(epoch) + ".png"
                # print(output['pred'].size())

                final_pred = visualize_tensor(output['pred'][0], mode='depth', window=None, depth_min=0,
                                              depth_max=10)
                gt_depth = visualize_tensor(sample['gt'][0], mode='depth', window=None, depth_min=0,
                                            depth_max=10)
                syn_data = visualize_tensor(sample_syn['depth_image'][0], mode='depth', window=None, depth_min=0,
                                            depth_max=10)
                vis_full = np.vstack((final_pred, gt_depth, syn_data))
                cv2.imwrite(os.path.join(result_dir, result_file_name), vis_full)

            d_step = 1

            for _ in range(d_step):
                # add_depth_save = add_depth
                add_depth_save = sample_syn['depth_image']
                pred_depth_D_save = output_clone
                pred_depth_save = output['pred']
                # vis = torch.cat([add_depth_save[0], pred_depth_D_save[0]],dim=1)
                # visualize_tensor(vis,'depth','Syn',depth_min=0,depth_max=10)
                # cv2.waitKey(2000)
                d_regularize = batch % args.d_reg_every == 0
                add_depth = add_depth_save.requires_grad_(d_regularize)
                real_diffuse, fake_diffuse, G_diffuse, real_t = diffuse(add_depth, pred_depth_D_save, pred_depth_save)

                # Dmain: Minimize logits for generated images.

                # fake_diffuse, fake_t = diffuse(pred_depth_D_save)
                D_fake_score = discr(fake_diffuse)
                D_fake_score = D_fake_score.squeeze()
                D_fake_loss = torch.nn.functional.softplus(D_fake_score).mean()

                discr.zero_grad()
                D_fake_loss.backward()
                D_optimizer.step()

                # Dmain: Maximize logits for real images.
                # Dr1: Apply R1 regularization.
                # print(real_diffuse.size())
                D_real_score = discr(real_diffuse)
                # D_real = D_real_score.clone().detach()
                # D_real_score = D_real_score.squeeze()
                D_real_loss = torch.nn.functional.softplus(-D_real_score)


                if d_regularize:
                    with conv2d_gradfix.no_weight_gradients():
                        # print(D_real_score.sum().size(),add_depth.size())
                        r1_grads=autograd.grad(outputs=[D_real_score.sum()],inputs=[add_depth],
                                               create_graph=True,only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    r1_loss = r1_penalty * (args.r1 / 2)

                    discr.zero_grad()
                    ( r1_loss + D_real_loss + 0 * D_real_score[0]).mean().backward()
                    D_optimizer.step()

                real_t = real_t.float().mean()
                writer_train.add_scalar('train/real_t', real_t, step)
                writer_train.add_scalar('train/D_fake_loss', D_fake_loss, step)
                writer_train.add_scalar('train/D_real_loss', D_real_loss.mean(), step)
                writer_train.add_scalar('train/D_r1_loss', r1_loss.mean(), step)
                # writer_train.add_scalar('train/D_train_loss', D_train_loss, step)

            G_result = (discr(G_diffuse)).squeeze()
            G_train_loss = torch.nn.functional.softplus(-G_result).mean()

            if batch % 5000 == 0:
                img_real = torch.cat([add_depth_save[0], real_diffuse[0]],dim=1)
                img_fake = torch.cat([pred_depth_D_save[0],fake_diffuse[0]],dim=1)
                img_G = torch.cat([pred_depth_save[0], G_diffuse[0]],dim=1)
                img = torch.cat([img_real, img_fake, img_G], dim=2)
                writer_train.add_image('train/before_D', img, global_step=step)

            ada_interval = 4
            ada_kimg = 100  # try first, then observe
            if batch % ada_interval == 0:  #
                # print("update T")
                C = (loader_train.batch_size * ada_interval) / (ada_kimg * 1000)
                result = D_real_score
                adjust = torch.sign(result) * C
                adjust = adjust.mean().item()
                diffuse.p = clip(diffuse.p + adjust, 0, 1)
                diffuse.update_T()

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_train.batch_size
            loss_val = loss_val / loader_train.batch_size

            loss_sum += 0.001 * G_train_loss
            with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            if gpu == 0:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_train.add(loss_val, metric_val)
                writer_train.add_scalar('train/L1_loss', loss_val[0][0], step)
                writer_train.add_scalar('train/L2_loss', loss_val[0][1], step)
                writer_train.add_scalar('train/G_train_loss', G_train_loss, step)
                writer_train.add_scalar('train/real_score', D_real_score.mean().item(), step)
                writer_train.add_scalar('train/fake_score', D_fake_score.mean().item(), step)
                writer_train.add_scalar('train/diffusion_p', diffuse.p, step)
                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'Train', current_time, log_loss / log_cnt)

                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str,
                                                              list_lr)

                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            writer_train.update(epoch, sample, output)

            if args.save_full or epoch == args.epochs:
                state = {
                    'net': net.module.state_dict(),
                    'net_D': discr.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'amp': amp.state_dict(),
                    'args': args
                }
            else:
                state = {
                    'net': net.module.state_dict(),
                    'net_D': discr.state_dict(),
                    'args': args
                }

            torch.save(state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))

        # Val
        torch.set_grad_enabled(False)
        net.eval()

        num_sample = len(loader_val) * loader_val.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_val):
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                      if val is not None}

            output = net(sample)


            if batch < 20:
                result_file_name = str(batch)+ '_epoch' + str(epoch) + ".png"
                # print(output['pred'].size())
                input_depth_min = output['pred'][0].min().item()
                input_depth_max = output['pred'][0].max().item()
                final_pred = visualize_tensor(output['pred'][0], mode='depth', window=None, depth_min=input_depth_min,
                                              depth_max=input_depth_max)
                gt_depth = visualize_tensor(sample['gt'][0], mode='depth', window=None, depth_min=input_depth_min,
                                            depth_max=input_depth_max)
                vis_full = np.vstack((final_pred, gt_depth))
                cv2.imwrite(os.path.join(result_dir, result_file_name), vis_full)

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_val.batch_size
            loss_val = loss_val / loader_val.batch_size

            if gpu == 0:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_val.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'Val', current_time, log_loss / log_cnt)
                pbar.set_description(error_str)
                pbar.update(loader_val.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            writer_val.update(epoch, sample, output)
            print('')

            writer_val.save(epoch, batch, sample, output)

        torch.set_grad_enabled(True)

        scheduler.step()


def test(args):
    # Prepare dataset
    data = get_data(args)

    data_test = data(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    # Network
    model = get_model(args)
    net = model(args)
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError

    net = nn.DataParallel(net)

    metric = get_metric(args)
    metric = metric(args)
    summary = get_summary(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
    except OSError:
        pass

    writer_test = summary(args.save_dir, 'test', args, None, metric.metric_name)

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if val is not None}

        t0 = time.time()
        output = net(sample)
        t1 = time.time()

        t_total += (t1 - t0)

        metric_val = metric.evaluate(sample, output, 'train')

        writer_test.add(None, metric_val)

        # Save data for analysis
        if args.save_image:
            writer_test.save(args.epochs, batch, sample, output)

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        pbar.set_description(error_str)
        pbar.update(loader_test.batch_size)

    pbar.close()

    writer_test.update(args.epochs, sample, output)

    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))


def main(args):
    if not args.test_only:
        if args.no_multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

            args.pretrain = '{}/model_{:05d}.pt'.format(args.save_dir,
                                                        args.epochs)

    test(args)


if __name__ == '__main__':
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)
