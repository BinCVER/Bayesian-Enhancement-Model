import os
import sys
import random
import time
import argparse
from tqdm import tqdm
import numpy as np
import json
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
import pyiqa
from basicsr.models import build_model
from basicsr.utils.gaussian_downsample import downsample_8x
from basicsr.utils.poisson_gaussian import add_poisson_gaussian_noise
from basicsr.utils.options import parse
from basicsr.bayesian import set_prediction_type
from basicsr.metrics import calculate_niqe
from basicsr.metrics import getUCIQE, getUIQM
from torchmetrics.multimodal import CLIPImageQualityAssessment




parser = argparse.ArgumentParser(description='Image Enhancement')

parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--input_dir', default='', type=str, help='Directory for inputs')
parser.add_argument('--target_dir', default='', type=str, help='Directory for targets')
parser.add_argument('--weights', default='', type=str, help='Path to weights of the first stage')
parser.add_argument('--opt', type=str, default='youryaml.yaml', help='Path to option YAML file of the first stage.')
parser.add_argument('--cond_weights', default='', type=str, help='Path to weights of the second stage; Emty means running a Single-Stage Model')
parser.add_argument('--cond_opt', type=str, default='', help='Path to option YAML file of the second stage.')
parser.add_argument('--dataset', default='yourdataset', type=str, help='Name of dataset')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--num_samples', default=200, type=int, help='Number of random samples')
parser.add_argument('--Monte_Carlo', action='store_true', help='use Monte Carlo Simulation, i.e., averaging the outcome of random samples. \
                    When the smaple number is very large, Monte Carlo Simulation is equal to the deterministic model')
parser.add_argument('--psnr_weight', default=1.0, type=float, help='Balance between PSNR and SSIM')
parser.add_argument('--no_ref', default='', type=str, choices=['clip', 'niqe', 'uiqm_uciqe', 'brisque'], help='no reference image quality evaluator. \
                    Support CLIP-IQA and NIQE')
parser.add_argument('--uiqm_weight', default=1.0, type=float, help='Balance between UIQM and UICIQE')
parser.add_argument('--lpips', action='store_true', help='True to compute LPIPS')
parser.add_argument('--deterministic', action='store_true', help='Use deterministic mode')
parser.add_argument('--parallel_num', default=1, type=int, help='Acceleartion by increasing the parallel processing samples. \
                    Adjust this to 1 if you encounter CUDA OOM issues')
parser.add_argument('--clip_prompts',nargs='+', default=['brightness', 'noisiness', 'quality'],
    help="A list of CLIP prompts to use with CLIP-IQA when 'no_ref' is set to 'clip'. \
        Recommended prompts include 'brightness', 'noisiness', and 'quality'. You can specify any one or more prompts separated by spaces. \
        If not specified, the default is ['brightness', 'noisiness', 'quality']."
)
# parser.add_argument('--seed', default=287128, type=int, help='fix random seed to reproduce consistent resutls')

def process_large_image(model, img, tile_size=512, overlap=32):
    """
    Process a large image in small tiles to avoid memory overflow with smooth blending in overlap regions.

    Args:
        model (torch.nn.Module): The deep learning model.
        img (torch.Tensor): Input image (C, H, W).
        tile_size (int): The size of each tile.
        overlap (int): The amount of overlap between tiles.

    Returns:
        torch.Tensor: Processed image (C, H, W).
    """
    _, h, w = img.shape
    stride = tile_size - overlap

    output = torch.zeros_like(img[:3], dtype=torch.float32)
    weight_map = torch.zeros_like(img[:3], dtype=torch.float32)

    # Create a weight map with smooth edges
    tile_weight = torch.ones((1, tile_size, tile_size), dtype=torch.float32)
    for i in range(overlap):
        tile_weight[:, i, :] *= i / overlap
        tile_weight[:, :, i] *= i / overlap
        tile_weight[:, -i-1, :] *= i / overlap
        tile_weight[:, :, -i-1] *= i / overlap
    tile_weight = tile_weight.unsqueeze(0).cuda()  # Add channel and batch dims and move to GPU

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(y_end - tile_size, 0)
            x_start = max(x_end - tile_size, 0)

            tile = img[:, y_start:y_end, x_start:x_end].unsqueeze(0).cuda()
            tile_padded = F.pad(tile, (0, tile_size - tile.size(3), 0, tile_size - tile.size(2)), 'reflect')

            with torch.no_grad():
                processed_tile = model(tile_padded)[-1][:, :, :tile.size(2), :tile.size(3)]

            weight = tile_weight[:, :, :processed_tile.size(2), :processed_tile.size(3)]

            output[:, y_start:y_end, x_start:x_end] += (processed_tile.squeeze(0) * weight.squeeze(0))
            weight_map[:, y_start:y_end, x_start:x_end] += weight.squeeze(0)

    # Normalize the output by the weight map
    output /= weight_map
    return output


args = parser.parse_args()
# #-------------------Set random seed----------------
# seed = args.seed
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#-------------------Load yaml----------------------
yaml_file = args.opt
weights = args.weights
cond_weights = args.cond_weights
print(f"dataset {args.dataset}")

opt = parse(args.opt, is_train=False)
opt['dist'] = False
net = build_model(opt).net_g
set_prediction_type(net, deterministic=args.deterministic)
if args.cond_opt:
    cond_opt = parse(args.cond_opt, is_train=False)
    cond_opt['dist'] = False
    netII = build_model(cond_opt).net_g

checkpoint = torch.load(weights)
if cond_weights:
    cond_checkpoint = torch.load(cond_weights)

scale_factor = 8

best_seeds = None
seed_file = os.path.join('Options/', '{}_seeds.json'.format(args.dataset))
# if os.path.exists(seed_file):
#     with open(seed_file, 'r') as f:
#         best_seeds = json.load(f)
#     print('Best seeds provided')

if args.deterministic:
    args.num_samples = 1
if args.num_samples == 1:
    args.parallel_num = 1
if best_seeds is not None:
    args.num_samples = args.parallel_num = 1
new_best_seeds = []
args.num_samples = args.num_samples - (args.num_samples % args.parallel_num)

net.load_state_dict(checkpoint['params'])
print("Loaded weights from", weights)
if cond_weights:
    netII.load_state_dict(cond_checkpoint['params'])
    print("Loaded weights from", cond_weights)


net.cuda()
net = nn.DataParallel(net)
net.eval()
if cond_weights:
    netII.cuda()
    netII = nn.DataParallel(netII)
    netII.eval()

dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
os.makedirs(result_dir, exist_ok=True)


if args.no_ref =='clip':
    clip_metric = CLIPImageQualityAssessment(model_name_or_path='clip_iqa',
                                            prompts=tuple(args.clip_prompts),
                                            data_range=255).cuda()
elif args.no_ref =='brisque':
    brisque_fn = pyiqa.create_metric('brisque', device='cuda')
psnr = []
ssim = []
lpips_ = []
niqe = []
uiqm = []
uciqe = []
brisque = []

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
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
        h, w = img.shape[0], img.shape[1]
        if target_dir != '':
            target = np.float32(utils.load_img(target_paths[p_idx])) / 255.
            target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).cuda()
        img_pad = _padimg_np(img, factor=8*scale_factor) # Paddings
        hp, wp = img_pad.shape[0], img_pad.shape[1]
        input_ = torch.from_numpy(img_pad).permute(2, 0, 1).unsqueeze(0).cuda()



        one_pred_list = []
        one_psnr_list = []
        one_ssim_list = []
        one_niqe_list = []
        one_clip_list =[]
        one_brisque_list = []
        one_uiqm_list = []
        one_uciqe_list = []
        one_pred_cond_list = []
        if args.Monte_Carlo:
            mc_pred_list = []

        sub_seed_list = []
        stageI_pred_imgs = []
        for i in range(args.num_samples):
            if best_seeds:
                torch.manual_seed(best_seeds[p_idx])
            else:
                sub_seed = random.randint(0, sys.maxsize - 1)
                sub_seed_list.append(sub_seed)
                torch.manual_seed(sub_seed)
            pred_cond, _, stageI_pred_img = net(input_)
            stageI_pred_img = torch.clamp(stageI_pred_img, 0, 1)
            stageI_pred_imgs.append(stageI_pred_img)
            if args.GT_mean:
                mean_pred = pred_cond.mean(dim=(2,3), keepdims=True)
                mean_target = (target_tensor / (img_tensor + 0.1)).mean(dim=(2,3), keepdims=True)
                pred_cond = pred_cond * (mean_target / mean_pred)
            if cond_weights:
                noise_level = cond_opt['condition'].get('noise_level', 0)
            one_pred_cond_list.append(pred_cond)


        torch.cuda.empty_cache()
        one_pred_conds = torch.cat(one_pred_cond_list, dim=0)
        if (not args.deterministic) or cond_weights:
            one_pred_conds = one_pred_conds + torch.randn_like(one_pred_conds) * noise_level
        one_pred_conds = F.interpolate(one_pred_conds, scale_factor=8, mode='bilinear', align_corners=False)
        stageI_pred_imgs = torch.cat(stageI_pred_imgs, dim=0)
        stageI_pred_imgs_unpad = stageI_pred_imgs[:, :, :h, :w]

        if args.Monte_Carlo:
            best_predI = one_pred_conds.mean(0)
        else:
            if args.no_ref == 'clip':
                for i in range(args.num_samples // args.parallel_num):
                    vs = clip_metric(stageI_pred_imgs_unpad[i*args.parallel_num:(i+1)*args.parallel_num].cuda())
                    if isinstance(vs, torch.Tensor):
                        one_clip_list.append(vs.cpu().numpy())
                    else:
                        if 'noisiness' in vs:
                            vs['noisiness'] = vs['noisiness'] * 1
                        if 'brightness' in vs:
                            # scaling down this to avoid outputting an over-exposed result
                            vs['brightness'] = vs['brightness'] * 1
                        vs = {key: value[None] if len(value.shape) == 0 else value for key, value in vs.items()}
                        vs_matrix = torch.stack(list(vs.values()))
                        vs = vs_matrix.mean(dim=0)
                        one_clip_list.extend(vs.cpu().numpy())
            elif args.no_ref == 'brisque':
                for i in range(args.num_samples // args.parallel_num):
                    brisque_scores = brisque_fn(stageI_pred_imgs_unpad[i*args.parallel_num:(i+1)*args.parallel_num].cuda())
                    one_brisque_list.extend(brisque_scores.cpu().numpy())

            stageI_pred_imgs_unpad = stageI_pred_imgs_unpad.permute(0,2,3,1)
            for i in range(args.num_samples):
                predI = stageI_pred_imgs_unpad[i].cpu().numpy()
                if args.no_ref in ['clip', 'brisque']:
                    pass # running CLIP/BRISQUE in parallel, so skip the loop here
                elif args.no_ref == 'niqe':
                    one_niqe_list.append(calculate_niqe(predI*255, crop_border=0))
                elif args.no_ref == 'uiqm_uciqe':
                    img_RGB = np.array(
                        Image.fromarray(img_as_ubyte(predI)).resize((256, int(256 / predI.shape[1] * predI.shape[0])))
                    )
                    one_uiqm_list.append(getUIQM(img_RGB))
                    one_uciqe_list.append(getUCIQE(img_as_ubyte(predI)))
                else:
                    if target_dir != '':
                        one_psnr_list.append(utils.calculate_psnr(target, predI))
                        one_ssim_list.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(predI)))
            torch.cuda.empty_cache()
            #------------------------------------------------------------------------------------------

            if args.no_ref in ['clip', 'niqe', 'uiqm_uciqe', 'brisque']:
                # No-Reference Inference--------------------------
                if args.no_ref == 'clip':
                    _idx = one_clip_list.index(max(one_clip_list))
                    # np.save('clip.npy', np.array(one_clip_list))
                elif args.no_ref == 'niqe':
                    _v = min(one_niqe_list)
                    _idx = one_niqe_list.index(_v)
                    niqe.append(_v)
                elif args.no_ref == 'uiqm_uciqe':
                    best_one_list = (args.uiqm_weight * np.array(one_uiqm_list) / max(one_uiqm_list) + (1-args.uiqm_weight)* np.array(one_uciqe_list) / max(one_uciqe_list)).tolist()
                    _idx = best_one_list.index(max(best_one_list))
                    uiqm.append(one_uiqm_list[_idx])
                    uciqe.append(one_uciqe_list[_idx])
                elif args.no_ref == 'brisque':
                    _v = min(one_brisque_list)
                    _idx = one_brisque_list.index(_v)
                    brisque.append(_v)
                best_predI = one_pred_conds[_idx]

            else:
                # Full-Reference Inference-----------------------
                if target_dir != '':
                    best_one_list = (args.psnr_weight * np.array(one_psnr_list) / max(one_psnr_list)  + (1 - args.psnr_weight) * np.array(one_ssim_list) / max(one_ssim_list)).tolist()
                    _idx = best_one_list.index(max(best_one_list))
                    best_predI = one_pred_conds[_idx]
                else:
                    # Deterministic Mode with no reference-------
                    best_predI = one_pred_conds[0]


        # if (not best_seeds) and (not args.deterministic):
        #     new_best_seeds.append(sub_seed_list[_idx])

        # Stage II-------------------------------------------
        input_ = torch.cat([input_, best_predI[None]], dim=1)
        # input_ = input_ + add_poisson_gaussian_noise(input_, poisson_scale=30.0, gaussian_std=0.01)
        if input_.shape[1] > 1500 or input_.shape[2] > 1500:
            predII = process_large_image(netII, input_[0], tile_size=1500, overlap=100)[None]
        else:
            predII = netII(input_)[-1]
        predII = torch.clamp(predII, 0, 1)
        predII = predII[:, :, :h, :w].permute(0,2,3,1).squeeze().cpu().numpy()

        if args.GT_mean:
            mean_pred = predII.mean(axis=(0,1), keepdims=True)
            mean_target = target.mean(axis=(0,1), keepdims=True)
            predII = np.clip(predII * (mean_target / mean_pred), 0, 1)

        if target_dir != '':
            psnr.append(utils.calculate_psnr(target, predII))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(predII)))
            if args.lpips:
                ex_p0 = lpips.im2tensor(img_as_ubyte(predII)).cuda()
                ex_ref = lpips.im2tensor(img_as_ubyte(target)).cuda()
                score_lpips = loss_fn.forward(ex_ref, ex_p0).item()
                lpips_.append(score_lpips)

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(predII))

    # with open(seed_file, 'w') as f:
    #     json.dump(new_best_seeds, f)
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

    if args.no_ref == 'brisque':
        brisque = np.mean(np.array(brisque))
        print("Best_BRISQUE: {:.4f}".format(brisque))
        f.write("Best_BRISQUE: {:.4f} \n".format(brisque))


