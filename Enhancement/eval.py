import os
import random
import time
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
import utils
from natsort import natsorted
from glob import glob
from PIL import Image
from skimage import img_as_ubyte
import lpips
from basicsr.models import build_model
from basicsr.utils.options import parse
from basicsr.bayesian import set_prediction_type
from basicsr.utils.histogram import compute_histograms
from basicsr.metrics import calculate_niqe
from basicsr.metrics import getUCIQE, getUIQM
from torchmetrics.multimodal import CLIPImageQualityAssessment




parser = argparse.ArgumentParser(description='Image Enhancement')

parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--input_dir', default='', type=str, help='Directory for inputs')
parser.add_argument('--target_dir', default='', type=str, help='Directory for targets')
parser.add_argument('--opt', type=str, default='youryaml.yaml', help='Path to option YAML file.')
parser.add_argument('--cond_opt', type=str, default='your condition_yaml.yaml', help='Path to option YAML file.')
parser.add_argument('--weights', default='yourweight.pth', type=str, help='Path to weights')
parser.add_argument('--cond_weights', default='yourweight.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='yourdataset', type=str, help='Name of dataset')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--num_samples', default=200, type=int, help='Number of random samples')
parser.add_argument('--Monte_Carlo', action='store_true', help='use Monte Carlo Simulation, i.e., averaging the outcome of random samples. \
                    When the smaple number is very large, Monte Carlo Simulation is equal to the deterministic model')
parser.add_argument('--psnr_weight', default=1.0, type=float, help='Balance between PSNR and SSIM')
parser.add_argument('--no_ref', default='', type=str, choices=['clip', 'niqe', 'uiqm_uciqe'], help='no reference image quality evaluator. \
                    Support CLIP-IQA and NIQE')
parser.add_argument('--uiqm_weight', default=1.0, type=float, help='Balance between UIQM and UICIQE')
parser.add_argument('--lpips', action='store_true', help='True to compute LPIPS')
parser.add_argument('--deterministic', action='store_true', help='Use deterministic mode')
parser.add_argument('--parallel_num', default=1, type=int, help='Acceleartion by increasing the parallel processing samples. \
                    Adjust this to 1 if you encounter CUDA OOM issues')
parser.add_argument('--seed', default=287128, type=int, help='fix random seed to reproduce consistent resutls')
parser.add_argument('--clip_prompts',nargs='+',
    default=['brightness', 'noisiness', 'quality'],
    help="A list of CLIP prompts to use with CLIP-IQA when 'no_ref' is set to 'clip'. \
        Recommended prompts include 'brightness', 'noisiness', and 'quality'. You can specify any one or more prompts separated by spaces. \
        If not specified, the default is ['brightness', 'noisiness', 'quality']."
)

args = parser.parse_args()

#-------------------Set random seed----------------
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#-------------------Load yaml----------------------
yaml_file = args.opt
weights = args.weights
cond_weights = args.cond_weights
print(f"dataset {args.dataset}")

opt = parse(args.opt, is_train=False)
opt['dist'] = False
cond_opt = parse(args.cond_opt, is_train=False)
cond_opt['dist'] = False

net = build_model(opt).net_g
set_prediction_type(net, deterministic=args.deterministic)
cond_net = build_model(cond_opt).net_g

checkpoint = torch.load(weights)
cond_checkpoint = torch.load(cond_weights)

scale_factor = opt['condition'].get('scale_down', 0) + opt['condition'].get('hist_patch_size', 0)

if args.deterministic:
    args.num_samples = 1
if args.num_samples == 1:
    args.parallel_num = 1
args.num_samples = args.num_samples - (args.num_samples % args.parallel_num)

net.load_state_dict(checkpoint['params'])
print("Loaded weights from", weights)
cond_net.load_state_dict(cond_checkpoint['params'])
print("Loaded weights from", cond_weights)


net.cuda()
net = nn.DataParallel(net)
net.eval()

cond_net.cuda()
cond_net = nn.DataParallel(cond_net)
cond_net.eval()

dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
os.makedirs(result_dir, exist_ok=True)


if args.no_ref =='clip':
    clip_metric = CLIPImageQualityAssessment(prompts=tuple(args.clip_prompts)).cuda()

psnr = []
ssim = []
lpips_ = []
niqe = []
uiqm = []
uciqe = []

if args.input_dir != '':
    input_dir = args.input_dir
    target_dir = args.target_dir
else:
    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
if args.GT_mean and target_dir =='':
    raise ValueError('GT_mean is only available when GT is provided')
input_paths = natsorted( glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.bmp')) + glob(os.path.join(input_dir, '*.tif')) )
if target_dir != '':
    target_paths = natsorted( glob(os.path.join(target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')) + glob(os.path.join(target_dir, '*.bmp')) + glob(os.path.join(target_dir, '*.tif')))

if args.lpips:
    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    loss_fn.cuda()
def _padimg_np(inp, factor):
    h, w = inp.shape[0], inp.shape[1]
    hp, wp = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = hp - h if h % factor != 0 else 0
    padw = wp - w if w % factor != 0 else 0
    if padh != 0 or padw !=0:
        inp = np.pad(inp, ((0, padh), (0, padw), (0, 0)), 'reflect')
    return inp

if len(input_paths) == 0:
    raise ValueError('No input images found')
mc_psnr = []
mc_ssim = []
start_time = time.perf_counter()
with torch.inference_mode():
    for p_idx, inp_path in tqdm(enumerate(input_paths), total=len(input_paths)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path)) / 255.
        h, w = img.shape[0], img.shape[1]
        if target_dir != '':
            target = np.float32(utils.load_img(target_paths[p_idx])) / 255.
            target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).cuda()
        img_pad = _padimg_np(img, factor=4*scale_factor) # Paddings
        hp, wp = img_pad.shape[0], img_pad.shape[1]
        input_ = torch.from_numpy(img_pad).permute(2, 0, 1).unsqueeze(0).cuda()

        if opt['condition']['type'] == 'mean':
            img_down_pad = cv2.resize(img_pad, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LINEAR)
            img_down_pad = torch.from_numpy(img_down_pad).permute(2, 0, 1).unsqueeze(0).cuda()
        elif opt['condition']['type'] == 'histogram':
            hist_lq_pad = compute_histograms(img_pad, opt['condition']['hist_patch_size'], opt['condition']['num_bins'])
            hist_lq_pad = hist_lq_pad.permute(1 ,2, 3, 0)
            hist_lq_pad = hist_lq_pad.reshape(hist_lq_pad.shape[0], hist_lq_pad.shape[1], -1).numpy()
            hist_lq_pad = torch.from_numpy(hist_lq_pad).permute(2, 0, 1).unsqueeze(0).cuda()

        one_pred_list = []
        one_psnr_list = []
        one_ssim_list = []
        one_niqe_list = []
        one_clip_list =[]
        one_uiqm_list = []
        one_uciqe_list = []
        one_pred_cond_list = []
        if args.Monte_Carlo:
            mc_pred_list = []

        if opt['condition']['type'] == 'mean':
            input_cond = img_down_pad
        elif opt['condition']['type'] == 'histogram':
            input_cond = hist_lq_pad

        for i in range(args.num_samples):
            pred_cond = net(input_cond)[-1]
            pred_cond = torch.clamp(pred_cond, 0, 1)
            if args.GT_mean and opt['condition']['type'] != 'histogram':
                mean_pred = pred_cond.mean(dim=(2,3), keepdims=True)
                mean_target = target_tensor.mean(dim=(2,3), keepdims=True)
                pred_cond = torch.clamp(pred_cond * (mean_target / mean_pred), 0, 1)

            # if opt['condition']['type'] == 'histogram':
            noise_level = cond_opt['condition'].get('noise_level', 0)
            pred_cond = pred_cond + torch.randn_like(pred_cond) * noise_level

            one_pred_cond_list.append(pred_cond.cpu())

        torch.cuda.empty_cache()
        one_pred_conds = torch.cat(one_pred_cond_list, dim=0)

        one_preds = []
        input_expended = input_.expand(args.parallel_num, *input_.shape[1:])
        for i in range(args.num_samples // args.parallel_num):
            sub_one_pred_conds = one_pred_conds[i*args.parallel_num: (i+1)*args.parallel_num].cuda()
            sub_one_pred_conds = F.interpolate(sub_one_pred_conds, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            one_preds.append(cond_net(torch.cat([input_expended, sub_one_pred_conds], dim=1))[-1].cpu())
        one_preds_tensor = torch.cat(one_preds, dim=0)[:, :, :h, :w]
        one_preds = one_preds_tensor.detach().permute(0, 2, 3, 1)
        if args.Monte_Carlo:
            mc_pred = torch.clamp(torch.mean(one_preds, dim=0), 0, 1).numpy()
        one_preds_tensor = torch.clamp(one_preds_tensor, 0, 1)
        one_preds_numpy = torch.clamp(one_preds, 0, 1).numpy()

        if args.no_ref == 'clip':
            for i in range(args.num_samples // args.parallel_num):
                vs = clip_metric(one_preds_tensor[i*args.parallel_num:(i+1)*args.parallel_num].cuda())
                if isinstance(vs, torch.Tensor):
                    one_clip_list.append(vs.cpu().numpy())
                else:
                    if 'noisiness' in vs:
                        vs['noisiness'] = vs['noisiness'] * 1
                    if 'brightness' in vs:
                        # scaling down this to avoid outputting an over-exposed result
                        vs['brightness'] = vs['brightness'] * 0.7
                    vs = {key: value[None] if len(value.shape) == 0 else value for key, value in vs.items()}
                    vs_matrix = torch.stack(list(vs.values()))
                    vs = vs_matrix.mean(dim=0)
                    one_clip_list.extend(vs.cpu().numpy())
        for i in range(args.num_samples):
            pred = one_preds_numpy[i]
            if args.GT_mean:
                mean_pred = pred.mean(axis=(0,1), keepdims=True)
                mean_target = target.mean(axis=(0,1), keepdims=True)
                pred = np.clip(pred * (mean_target / mean_pred), 0, 1)
            one_pred_list.append(pred)
            if args.no_ref == 'clip':
                pass # running CLIP in parallel, so skip the loop here
            elif args.no_ref == 'niqe':
                one_niqe_list.append(calculate_niqe(pred*255, crop_border=0))
            elif args.no_ref == 'uiqm_uciqe':
                img_RGB = np.array(
                    Image.fromarray(img_as_ubyte(pred)).resize((256, int(256 / pred.shape[1] * pred.shape[0])))
                )
                one_uiqm_list.append(getUIQM(img_RGB))
                one_uciqe_list.append(getUCIQE(img_as_ubyte(pred)))
            else:
                if target_dir != '':
                    one_psnr_list.append(utils.calculate_psnr(target, pred))
                    one_ssim_list.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(pred)))
        torch.cuda.empty_cache()
        #------------------------------------------------------------------------------------------

        if args.no_ref in ['clip', 'niqe', 'uiqm_uciqe']:
            # No-Reference Inference--------------------------
            if args.no_ref == 'clip':
                _idx = one_clip_list.index(max(one_clip_list))
            elif args.no_ref == 'niqe':
                _v = min(one_niqe_list)
                _idx = one_niqe_list.index(_v)
                niqe.append(_v)
            elif args.no_ref == 'uiqm_uciqe':
                best_one_list = (args.uiqm_weight * np.array(one_uiqm_list) / max(one_uiqm_list) + (1-args.uiqm_weight)* np.array(one_uciqe_list) / max(one_uciqe_list)).tolist()
                _idx = best_one_list.index(max(best_one_list))
                uiqm.append(one_uiqm_list[_idx])
                uciqe.append(one_uciqe_list[_idx])
            best_one_pred = one_pred_list[_idx]
            if target_dir != '':
                psnr.append(utils.calculate_psnr(target, best_one_pred))
                ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(best_one_pred)))
        else:
            # Full-Reference Inference-----------------------
            if target_dir != '':
                best_one_list = (args.psnr_weight * np.array(one_psnr_list) / max(one_psnr_list)  + (1 - args.psnr_weight) * np.array(one_ssim_list) / max(one_ssim_list)).tolist()
                _idx = best_one_list.index(max(best_one_list))
                best_one_pred = one_pred_list[_idx]
                best_one_psnr = one_psnr_list[_idx]
                best_one_ssim = one_ssim_list[_idx]
                psnr.append(best_one_psnr)
                ssim.append(best_one_ssim)
            else:
                # Deterministic Mode with no reference-------
                best_one_pred = one_pred_list[0]



        if target_dir != '':
            if args.lpips:
                ex_p0 = lpips.im2tensor(img_as_ubyte(best_one_pred)).cuda()
                ex_ref = lpips.im2tensor(img_as_ubyte(target)).cuda()
                score_lpips = loss_fn.forward(ex_ref, ex_p0).item()
                lpips_.append(score_lpips)

            if args.Monte_Carlo:
                if args.GT_mean:
                    mean_mc_pred = cv2.cvtColor(mc_pred.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                    mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                    mc_pred = np.clip(mc_pred * (mean_target / mean_mc_pred), 0, 1)
                mc_psnr.append(utils.calculate_psnr(target, mc_pred))
                mc_ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(mc_pred)))

        # one_rank_list = one_clip_list
        # # one_rank_list = one_niqe_list
        # sorted_one_rank_list = sorted(one_rank_list, reverse=True)
        # best_score = sorted_one_rank_list[-1]
        # # sorted_one_rank_list = sorted_one_rank_list[0::args.num_samples//4]
        # for _i in range(len(sorted_one_rank_list)):
        #     _idx2 = one_rank_list.index(sorted_one_rank_list[_i])
        #     utils.save_img((os.path.join(result_dir, '{:.2f}.png'.format(sorted_one_rank_list[_i]))), img_as_ubyte(one_pred_list[_idx2]))

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(best_one_pred))

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"running time: {execution_time:.4f} sec")

with open(os.path.join(result_dir, 'result.txt'), 'w') as f:
    if target_dir != '':
        psnr = np.mean(np.array(psnr))
        ssim = np.mean(np.array(ssim))
        print("Best_PSNR: {:.4f} dB".format(psnr))
        print("Best_SSIM: {:.4f}".format(ssim))
        f.write("Best_PSNR: {:.4f} dB \n".format(psnr))
        f.write("Best_SSIM: {:.4f} \n".format(ssim))
        if args.lpips:
            lpips_ = np.mean(np.array(lpips_))
            print("Best_lpips: {:.4f}".format(lpips_))
            f.write("Best_lpips: {:.4f} \n".format(lpips_))

    if args.no_ref == 'niqe':
        niqe = np.mean(np.array(niqe))
        print("Best_NIQE: {:.4f}".format(niqe))
        f.write("Best_NIQE: {:.4f} \n".format(niqe))

    if args.no_ref == 'uiqm_uciqe':
        uiqm = np.mean(np.array(uiqm))
        uciqe = np.mean(np.array(uciqe))
        print("Best_UIQM: {:.4f}".format(uiqm))
        print("Best_UCIQE: {:.4f}".format(uciqe))
        f.write("Best_UIQM: {:.4f} \n".format(uiqm))
        f.write("Best_UCIQE: {:.4f} \n".format(uciqe))

    if args.Monte_Carlo and target_dir != '':
        mc_psnr = np.mean(np.array(mc_psnr))
        mc_ssim = np.mean(np.array(mc_ssim))
        print("MC_PSNR: {:.4f} dB".format(mc_psnr))
        print("MC_SSIM: {:.4f}".format(mc_ssim))
        f.write("MC_PSNR: {:.4f} dB \n".format(mc_psnr))
        f.write("MC_SSIM: {:.4f} \n".format(mc_ssim))