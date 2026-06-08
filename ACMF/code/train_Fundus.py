import argparse
import logging
import os
import random
import shutil
import sys
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from networks.unet_model import UNet
from dataloaders.dataloader import FundusSegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.cuda.amp import autocast, GradScaler
import contextlib
import matplotlib.pyplot as plt 

from torch.optim.lr_scheduler import LambdaLR
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fundus')
parser.add_argument("--save_name", type=str, default="debug", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument('--num_eval_iter', type=int, default=250)
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--load',action='store_true')
parser.add_argument('--load_path',type=str,default='../model/lb1_ratio0.2/iter_6000.pth')
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

parser.add_argument("--label_bs", type=int, default=2, help="labeled_batch_size per gpu")
parser.add_argument("--unlabel_bs", type=int, default=2)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=5)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument("--cutmix_prob", default=1.0, type=float)
parser.add_argument("--LB", default=0.01, type=float)
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cycle(iterable: Iterable):
    """Make an iterator returning elements from the iterable.

    .. note::
        **DO NOT** use `itertools.cycle` on `DataLoader(shuffle=True)`.\n
        Because `itertools.cycle` saves a copy of each element, batches are shuffled only at the first epoch. \n
        See https://docs.python.org/3/library/itertools.html#itertools.cycle for more details.
    """
    while True:
        for x in iterable:
            yield x

def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        
        
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np

def low_freq_mutate_np( amp_src, amp_trg, L=0.1, degree=1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    # ratio = random.randint(1,10)/10
    ratio = random.uniform(0, degree)

    a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * (1-ratio) + a_trg[:,h1:h2,w1:w2] * ratio
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def source_to_target_freq( src_img, amp_trg, L=0.1, degree=1 ):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img #.transpose((2, 0, 1))
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L, degree=degree)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg #.transpose(1, 2, 0)



part = ['cup', 'disc']
dataset = FundusSegmentation
n_part = len(part)
dice_calcu = {'fundus':metrics.dice_coeff_2label}

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, ema=True):
    model.eval()
    model_name = 'ema' if ema else 'stu'
    val_loss = 0.0
    val_dice = [0.0] * n_part
    domain_metrics = []
    domain_num = len(test_dataloader)
    ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    softmax, sigmoid, multi = False, True, True
    dice_loss = losses.DiceLossWithMask(2)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        dc = -1
        domain_val_loss = 0.0
        domain_val_dice = [0.0] * n_part
        for batch_num,sample in enumerate(cur_dataloader):
            dc = sample['dc'][0].item()
            data = sample['image'].cuda()
            mask = sample['label'].cuda()

            cup_mask = mask.eq(0).float()
            disc_mask = mask.le(128).float()
            mask = torch.cat((cup_mask.unsqueeze(1), disc_mask.unsqueeze(1)),dim=1)

            output,_ = model(data)
            loss_seg = ce_loss(output, mask).mean() + \
                        dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
            dice = dice_calcu[args.dataset](np.asarray(torch.sigmoid(output.cpu()))>=0.5,mask.clone().cpu())

            domain_val_loss += loss_seg.item()
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]

        domain_val_loss /= len(cur_dataloader)
        val_loss += domain_val_loss
        writer.add_scalar('{}_val/domain{}/loss'.format(model_name, dc), domain_val_loss, epoch)
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
        for n, p in enumerate(part):
            writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name, dc, p), domain_val_dice[n], epoch)
        text = 'domain%d epoch %d : loss : %f' % (dc, epoch, domain_val_loss)
        for n, p in enumerate(part):
            text += ' val_%s_dice: %f' % (p, domain_val_dice[n])
            if n != n_part-1:
                text += ','
        logging.info(text)
        
    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('{}_val/val_{}_dice'.format(model_name, p), val_dice[n], epoch)
    text = 'epoch %d : loss : %f' % (epoch, val_loss)
    for n, p in enumerate(part):
        text += ' val_%s_dice: %f' % (p, val_dice[n])
        if n != n_part-1:
            text += ','
    logging.info(text)
    return val_dice

def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr

    num_channels = 3
    patch_size = 256
    num_classes = 2
    args.label_bs = 2
    args.unlabel_bs = 2
    min_v, max_v = 0.5, 1.5
    fillcolor = 255
    args.max_iterations = 30000
    if args.domain_num >=4:
        args.domain_num = 4

    max_iterations = args.max_iterations
    weak = transforms.Compose([tr.RandomScaleCrop(patch_size),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform(),
            ])
    
    strong = transforms.Compose([
            tr.Brightness(min_v, max_v),
            tr.Contrast(min_v, max_v),
            tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
    ])

    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_num = args.domain_num
    domain = list(range(1,domain_num+1))
    domain_len = [50, 99, 320, 320]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain-1]
    lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor)
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
        test_dataset.append(cur_dataset)
    lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    def create_model(ema=False):
        # Network definition
        if args.model == 'unet':
            model = UNet(n_channels = num_channels, n_classes = num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model()
    ema_model = create_model(ema=True)

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # set to train

    ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    softmax, sigmoid, multi = False, True, True

    dice_loss = losses.DiceLossWithMask(num_classes)

    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    max_epoch = max_iterations // args.num_eval_iter
    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part
    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] *n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part

    iter_num = int(iter_num)
    threshold = args.threshold


    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        ema_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')
        for i_batch in range(1, args.num_eval_iter+1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'], ulb_sample['strong_aug'], ulb_sample['label']
            lb_dc, ulb_dc = lb_sample['dc'].cuda(), ulb_sample['dc'].cuda()
            
            move_transx = []
            for i in range(len(lb_x_w)):
                amp_trg = extract_amp_spectrum((ulb_x_w[i].numpy()+1)*127.5)
                img_freq = source_to_target_freq(((lb_x_w[i]+1)*127.5).numpy(), amp_trg, L=args.LB, degree=iter_num/max_iterations)
                img_freq = np.clip(img_freq, 0, 255).astype(np.float32)
                move_transx.append(img_freq)
            move_transx = torch.tensor(np.array(move_transx), dtype=torch.float32)
            move_transx = move_transx/127.5 -1
            move_transx = move_transx.cuda()

            move_transx_un = []
            for i in range(len(ulb_x_w)):
                amp_trg_un = extract_amp_spectrum((lb_x_w[i].numpy() + 1) * 127.5)
                img_freq_un = source_to_target_freq(((ulb_x_w[i] + 1) * 127.5).numpy(), amp_trg_un, L=args.LB,
                                                    degree=iter_num / max_iterations)
                img_freq_un = np.clip(img_freq_un, 0, 255).astype(np.float32)
                move_transx_un.append(img_freq_un)
            move_transx_un = torch.tensor(np.array(move_transx_un), dtype=torch.float32)
            move_transx_un = move_transx_un / 127.5 - 1
            move_transx_un = move_transx_un.cuda()


            lb_x_w, lb_y, ulb_x_w, ulb_x_s, ulb_y = lb_x_w.cuda(), lb_y.cuda(), ulb_x_w.cuda(), ulb_x_s.cuda(), ulb_y.cuda()
            lb_cup_label = lb_y.eq(0).float() # == 0
            lb_disc_label = lb_y.le(128).float()  # <= 128
            lb_mask = torch.cat((lb_cup_label.unsqueeze(1), lb_disc_label.unsqueeze(1)),dim=1)
            ulb_cup_label = ulb_y.eq(0).float()
            ulb_disc_label = ulb_y.le(128).float()
            ulb_mask = torch.cat((ulb_cup_label.unsqueeze(1), ulb_disc_label.unsqueeze(1)),dim=1)


            with amp_cm():
                with torch.no_grad():

                    logits_ulb_x_w,_ = ema_model(ulb_x_w)


                    prob = logits_ulb_x_w.sigmoid()
                    pseudo_label = prob.ge(0.5).float()
                    mask = prob.ge(threshold).float() + prob.le(1-threshold).float()
                    # mask = torch.ones_like(prob).float()

                # labeled_image_pre, unlabeled_image_pre = model.forward_pair_Ablation(
                #     lb_x_w, ulb_x_s)
                labeled_image_pre, unlabeled_image_pre, labeled_image_pre_trans, unlabeled_image_pre_trans, *_ = model.forward_pair(
                    lb_x_w, ulb_x_s, move_transx, move_transx_un)

                sup_loss = ce_loss( labeled_image_pre, lb_mask).mean() + \
                            dice_loss( labeled_image_pre, lb_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
                sup_loss_trans = ce_loss(labeled_image_pre_trans, lb_mask).mean() + \
                           dice_loss(labeled_image_pre_trans, lb_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid,
                                     multi=multi)
                
                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))



                unsup_loss_s = (ce_loss(unlabeled_image_pre, pseudo_label) * mask.squeeze(1)).mean() + \
                                dice_loss(unlabeled_image_pre, pseudo_label.unsqueeze(1), mask=mask, softmax=softmax, sigmoid=sigmoid, multi=multi)
                unsup_loss_s_trans = (ce_loss(unlabeled_image_pre_trans, pseudo_label) * mask.squeeze(1)).mean() + \
                                dice_loss(unlabeled_image_pre_trans, pseudo_label.unsqueeze(1), mask=mask, softmax=softmax, sigmoid=sigmoid, multi=multi)

                loss = sup_loss+sup_loss_trans + consistency_weight * ( unsup_loss_s+unsup_loss_s_trans)
                # loss = sup_loss  + consistency_weight * (unsup_loss_s)

            optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # update ema model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('train/mask', mask.mean(), iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/sup_loss', sup_loss.item(), iter_num)
            writer.add_scalar('train/unsup_loss_s', unsup_loss_s.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/bi_consistency_weight', consistency_weight**2, iter_num)
            if p_bar is not None:
                p_bar.update()
            p_bar.set_description('iteration %d: loss:%.4f,sup_loss:%.4f, cons_w:%.4f,mask_ratio:%.4f'
                                    % (iter_num, loss.item(), sup_loss.item(), consistency_weight, mask.mean()))

        if p_bar is not None:
            p_bar.close()


        logging.info('test ema model')
        text = ''
        val_dice= test(args, ema_model, test_dataloader, epoch_num+1, writer)
        for n, p in enumerate(part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]
                best_dice_iter[n] = iter_num
            text += 'val_%s_best_dice: %f at %d iter' % (p, best_dice[n], best_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best_avg_dice:
            best_avg_dice = sum(val_dice) / len(val_dice)
            best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                dice_of_best_avg[n] = val_dice[n]
        text += 'val_best_avg_dice: %f at %d iter' % (best_avg_dice, best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best_avg[n])
        logging.info(text)
        logging.info('test stu model')
        stu_val_dice = test(args, model, test_dataloader, epoch_num+1, writer, ema=False)
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "{}_avg_dice_best_model.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg model to {}'.format(save_best))
            torch.save(model.state_dict(), save_best)
        text += '*******val_best_avg_dice*******: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)
        text = 'checkpoint.pth'
        checkpoint_path = os.path.join(snapshot_path, text)
        util.save_osmancheckpoint(epoch_num+1, ema_model, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter, checkpoint_path)
        logging.info('save checkpoint to {}'.format(checkpoint_path))

        
    writer.close()


if __name__ == "__main__":
    snapshot_path = "../model/" + args.dataset + "/" + args.save_name + "/"
    train_data_path = '../data/Fundus/'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    train(args, snapshot_path)
