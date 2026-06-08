import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.unet_model import UNet
from dataloaders.dataloader import FundusSegmentation, MNMSSegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from medpy.metric import binary

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'MNMS'])
parser.add_argument("--save_name", type=str, default="debug", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--eval', type=bool, default=True)

parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)

parser.add_argument('--save_img', action='store_true')

# === save prediction folder ===
parser.add_argument('--save_pred', action='store_true', help='save input/pred/gt into a folder')
parser.add_argument('--pred_dir', type=str, default=None,
                    help='where to save predictions (default: snapshot_path/predictions)')

args = parser.parse_args()

def _mkdir(p):
    os.makedirs(p, exist_ok=True)


def _to_uint8_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - vmin) / (vmax - vmin + 1e-8)
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_gray_png(path, arr_hw_uint8: np.ndarray):
    Image.fromarray(arr_hw_uint8, mode="L").save(path)


def save_rgb_png(path, arr_hwc_uint8: np.ndarray):
    Image.fromarray(arr_hwc_uint8, mode="RGB").save(path)


def tensor_to_uint8_image(t_img: torch.Tensor) -> np.ndarray:
    t = t_img.detach().float().cpu()
    if t.dim() == 2:
        arr = t.numpy()
        return _to_uint8_minmax(arr)
    if t.dim() == 3:
        C, H, W = t.shape
        if C == 3:
            chs = []
            for c in range(3):
                chs.append(_to_uint8_minmax(t[c].numpy()))
            return np.stack(chs, axis=-1)
        else:
            return _to_uint8_minmax(t[0].numpy())
    raise ValueError(f"Unexpected image tensor shape: {tuple(t.shape)}")


def save_label_png(path, label_hw: np.ndarray, num_classes: int):
    label_hw = label_hw.astype(np.int32)
    if num_classes <= 1:
        out = np.zeros_like(label_hw, dtype=np.uint8)
    else:
        out = (label_hw * (255 // (num_classes - 1))).astype(np.uint8)
    save_gray_png(path, out)


def save_binary_png(path, mask_hw: np.ndarray):
    m = (mask_hw > 0).astype(np.uint8) * 255
    save_gray_png(path, m)


def save_fundus_trimap_png(path, cup_mask_hw, disc_mask_hw):
    cup = (cup_mask_hw > 0)
    disc = (disc_mask_hw > 0) | cup
    disc_only = disc & (~cup)
    out = np.zeros(cup.shape, dtype=np.uint8)
    out[disc_only] = 128
    out[cup] = 255
    save_gray_png(path, out)


def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return ""


def get_sample_id(sample: dict, j: int, fallback: str):
    if not isinstance(sample, dict):
        return fallback

    cand_keys = ["name", "id", "case", "uid", "img_name", "path", "image_path"]
    for k in cand_keys:
        if k in sample:
            v = sample[k]
            try:
                if isinstance(v, (list, tuple)):
                    vv = v[j]
                else:
                    vv = v
                s = _safe_str(vv)
                if len(s) == 0:
                    continue
                if "/" in s or "\\" in s:
                    s = os.path.basename(s)
                s = s.replace(" ", "_")
                s = os.path.splitext(s)[0]
                if len(s) > 0:
                    return s
            except Exception:
                pass
    return fallback


def to_2d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    temp_prob = input_tensor == torch.ones_like(input_tensor)
    tensor_list.append(temp_prob)
    temp_prob2 = input_tensor > torch.zeros_like(input_tensor)
    tensor_list.append(temp_prob2)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def to_3d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(1, 4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


# -------------------------
# Dataset settings
# -------------------------
if args.dataset == 'fundus':
    part = ['cup', 'disc']
    dataset = FundusSegmentation
elif args.dataset == 'MNMS':
    part = ['lv', 'myo', 'rv']
    dataset = MNMSSegmentation

n_part = len(part)
dice_calcu = {
    'fundus': metrics.dice_coeff_2label,
    'MNMS': metrics.dice_coeff_3label
}


# -------------------------
# Test Function
# -------------------------
@torch.no_grad()
def test(args, model, test_dataloader, out_root=None):
    model.eval()
    domain_num = len(test_dataloader)
    num = 0
    global_dice_collector = []
    global_dc_collector = []
    global_jc_collector = []
    global_hd_collector = []
    global_asd_collector = []

    # === prepare prediction folder ===
    if args.save_pred:
        if out_root is None:
            out_root = "./predictions"
        _mkdir(out_root)

    for dom_idx in range(domain_num):
        cur_dataloader = test_dataloader[dom_idx]
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [
            0.0] * n_part
        domain_code = dom_idx + 1

        # domain folder
        if args.save_pred:
            domain_dir = os.path.join(out_root, f"domain_{domain_code:02d}")
            img_dir = os.path.join(domain_dir, "images")
            pred_dir = os.path.join(domain_dir, "pred")
            gt_dir = os.path.join(domain_dir, "gt")
            _mkdir(img_dir)
            _mkdir(pred_dir)
            _mkdir(gt_dir)

        for batch_num, sample in enumerate(cur_dataloader):
            data = sample['image'].cuda(non_blocking=True)
            mask = sample['label'].cuda(non_blocking=True)

            if args.dataset == 'fundus':
                cup_mask = mask.eq(0).float()
                disc_mask = mask.le(128).float()
                mask = torch.cat((cup_mask.unsqueeze(1), disc_mask.unsqueeze(1)), dim=1)
            elif args.dataset == 'MNMS':
                mask_ = mask[..., 0].eq(255).float()
                mask_[mask[..., 1].eq(255)] = 2
                mask_[mask[..., 2].eq(255)] = 3
                mask = mask_.long()

            output, x5 = model(data)

            mask_cpu = mask.cpu()
            output_cpu = output.cpu()
            data_cpu = data.cpu()

            if args.dataset == 'fundus':
                pred_label = torch.sigmoid(output_cpu).ge(0.5)
                pred_onehot = pred_label.clone()
                mask_onehot = mask_cpu.clone()
            elif args.dataset == 'MNMS':
                pred_label = torch.max(torch.softmax(output_cpu, dim=1), dim=1)[1]
                pred_onehot = to_3d(pred_label)
                mask_onehot = to_3d(mask_cpu)

            dice = dice_calcu[args.dataset](np.asarray(pred_label), mask_cpu)
            avg_dice = sum(dice) / len(dice)

            if args.save_pred:
                bs = data_cpu.shape[0]
                for j in range(bs):
                    sid = get_sample_id(sample, j, fallback=f"dom{domain_code:02d}_b{batch_num:04d}_i{j:02d}")

                    img_u8 = tensor_to_uint8_image(data_cpu[j])
                    if isinstance(img_u8, np.ndarray) and img_u8.ndim == 3 and img_u8.shape[-1] == 3:
                        save_rgb_png(os.path.join(img_dir, f"{sid}_img.png"), img_u8)
                    else:
                        save_gray_png(os.path.join(img_dir, f"{sid}_img.png"), img_u8)

                    if args.dataset == "fundus":
                        pred_oh = pred_onehot[j].numpy().astype(np.uint8)
                        gt_oh = mask_onehot[j].numpy().astype(np.uint8)
                        save_fundus_trimap_png(
                            os.path.join(pred_dir, f"{sid}_pred.png"),
                            cup_mask_hw=pred_oh[0],
                            disc_mask_hw=pred_oh[1]
                        )
                        save_fundus_trimap_png(
                            os.path.join(gt_dir, f"{sid}_gt.png"),
                            cup_mask_hw=gt_oh[0],
                            disc_mask_hw=gt_oh[1]
                        )
                    elif args.dataset == "prostate":
                        pred_hw = pred_label[j].numpy().astype(np.int32)
                        gt_hw = mask_cpu[j].numpy().astype(np.int32)
                        save_label_png(os.path.join(pred_dir, f"{sid}_pred.png"), pred_hw, num_classes=2)
                        save_label_png(os.path.join(gt_dir, f"{sid}_gt.png"), gt_hw, num_classes=2)
                    elif args.dataset == "MNMS":
                        pred_hw = pred_label[j].numpy().astype(np.int32)
                        gt_hw = mask_cpu[j].numpy().astype(np.int32)
                        save_label_png(os.path.join(pred_dir, f"{sid}_pred.png"), pred_hw, num_classes=4)
                        save_label_png(os.path.join(gt_dir, f"{sid}_gt.png"), gt_hw, num_classes=4)


            if args.eval and args.save_img:
                for j in range(len(data_cpu)):
                    num += 1
                    util.draw_mask_and_save(
                        data_cpu[j], pred_onehot[j],
                        './img/save/{}_{}_{}.png'.format(domain_code, num, round(avg_dice, 4))
                    )


            dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
            for j in range(len(data_cpu)):
                for p_idx, p in enumerate(part):
                    dc[p_idx] += binary.dc(np.asarray(pred_onehot[j, p_idx], dtype=bool),
                                           np.asarray(mask_onehot[j, p_idx], dtype=bool))
                    jc[p_idx] += binary.jc(np.asarray(pred_onehot[j, p_idx], dtype=bool),
                                           np.asarray(mask_onehot[j, p_idx], dtype=bool))
                    if pred_onehot[j, p_idx].float().sum() < 1e-4:
                        hd[p_idx] += 100
                        asd[p_idx] += 100
                    else:
                        hd[p_idx] += binary.hd95(np.asarray(pred_onehot[j, p_idx], dtype=bool),
                                                 np.asarray(mask_onehot[j, p_idx], dtype=bool))
                        asd[p_idx] += binary.asd(np.asarray(pred_onehot[j, p_idx], dtype=bool),
                                                 np.asarray(mask_onehot[j, p_idx], dtype=bool))

            for p_idx, p in enumerate(part):
                dc[p_idx] /= len(data_cpu)
                jc[p_idx] /= len(data_cpu)
                hd[p_idx] /= len(data_cpu)
                asd[p_idx] /= len(data_cpu)

            for p_idx in range(len(domain_val_dice)):
                domain_val_dice[p_idx] += dice[p_idx]
                domain_val_dc[p_idx] += dc[p_idx]
                domain_val_jc[p_idx] += jc[p_idx]
                domain_val_hd[p_idx] += hd[p_idx]
                domain_val_asd[p_idx] += asd[p_idx]

        # domain average
        for p_idx in range(len(domain_val_dice)):
            domain_val_dice[p_idx] /= len(cur_dataloader)
            domain_val_dc[p_idx] /= len(cur_dataloader)
            domain_val_jc[p_idx] /= len(cur_dataloader)
            domain_val_hd[p_idx] /= len(cur_dataloader)
            domain_val_asd[p_idx] /= len(cur_dataloader)


            global_dice_collector.append(domain_val_dice[p_idx])
            global_dc_collector.append(domain_val_dc[p_idx])
            global_jc_collector.append(domain_val_jc[p_idx])
            global_hd_collector.append(domain_val_hd[p_idx])
            global_asd_collector.append(domain_val_asd[p_idx])

        text = 'domain%d Test Average Results :' % (domain_code)
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dice: %f, ' % (p, domain_val_dice[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dc: %f, ' % (p, domain_val_dc[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_jc: %f, ' % (p, domain_val_jc[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_hd: %f, ' % (p, domain_val_hd[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_asd: %f, ' % (p, domain_val_asd[n])
        logging.info(text)

    model.train()

    if len(global_dice_collector) > 0:
        overall_dice = sum(global_dice_collector) / len(global_dice_collector)
        overall_jc = sum(global_jc_collector) / len(global_jc_collector)
        overall_hd = sum(global_hd_collector) / len(global_hd_collector)
        overall_asd = sum(global_asd_collector) / len(global_asd_collector)

        logging.info('====================================================================================')
        logging.info('All-Domain Overall Average Metrics :')
        logging.info('\tDice: %.4f,  JC: %.4f' % (overall_dice, overall_jc))
        logging.info('\tHD95: %.2f,  ASD: %.2f' % (overall_hd, overall_asd))
        logging.info('====================================================================================')

    if args.save_pred:
        logging.info(f"[SavePred] Predictions saved to: {out_root}")

    return

def main(args, snapshot_path, train_data_path):
    if args.dataset == 'fundus':
        num_channels = 3
        num_classes = 2
        if args.domain_num >= 4:
            args.domain_num = 4
    elif args.dataset == 'MNMS':
        num_channels = 1
        num_classes = 4
        if args.domain_num >= 4:
            args.domain_num = 4

    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_num = args.domain_num
    test_dataset = []
    test_dataloader = []

    for i in range(1, domain_num + 1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1,
                              domain=[i], normal_toTensor=normal_toTensor)
        test_dataset.append(cur_dataset)

    for i in range(0, domain_num):
        cur_dataloader = DataLoader(
            test_dataset[i],
            batch_size=args.test_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        test_dataloader.append(cur_dataloader)

    def create_model(ema=False):
        if args.model == 'unet':
            model = UNet(n_channels=num_channels, n_classes=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model()

    if args.eval:
        path = f"../model/{args.dataset}/{args.save_name}/unet_avg_dice_best_model.pth"

        state = torch.load(path, map_location="cpu", weights_only=True)
        state.pop("channel_mask.running_S", None)
        model.load_state_dict(state, strict=False)

        out_root = args.pred_dir
        if out_root is None:
            out_root = os.path.join(snapshot_path, "predictions")

        test(args, model, test_dataloader, out_root=out_root)
        return


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    snapshot_path = "../model/" + args.dataset + "/" + args.save_name + "/"
    _mkdir(snapshot_path)

    if args.dataset == 'fundus':
        train_data_path = '/data/Fundus/'
    elif args.dataset == 'MNMS':
        train_data_path = "/data/MNMS/"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    main(args, snapshot_path, train_data_path)
