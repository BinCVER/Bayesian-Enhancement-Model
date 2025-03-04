import os
import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import os
import random
import torch.nn.functional as F
from functools import partial
import glob
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.gaussian_downsample import downsample_fft
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.bayesian import convert2bnnv2, convert2bnn_selective, get_kl_loss, set_prediction_type, set_prediction_type
from basicsr.utils.mixing_augment import Mixing_Augment
loss_module = importlib.import_module('basicsr.losses')
metric_module = importlib.import_module('basicsr.metrics')
try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False


@MODEL_REGISTRY.register()
class ConditionGeneratorV2(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ConditionGeneratorV2, self).__init__(opt)

        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        # if self.use_amp:
        #     print('Using Automatic Mixed Precision')
        # else:
        #     print('Not using Automatic Mixed Precision')

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = build_network(opt['network_g'])
        bnn_config = {
                "sigma_init": opt.get('sigma_init', 0.05),
                "decay": 0.998,
                "pretrain": False,
        }
        if opt.get('selective', False):
            convert2bnn_selective(self.net_g, bnn_config)
        else:
            convert2bnnv2(self.net_g, bnn_config)
        self.net_g = self.model_to_device(self.net_g)
        # self.deterministic = True
        # set_prediction_type(self.net_g, deterministic=True)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key=self.opt["path"].get("param_key", "params"),
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        normal_params = []
        custom_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if "impfusion" in k:
                    custom_params.append(v)
                else:
                    normal_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

            optim_params = [
                {'params': normal_params, 'lr_mult': 1, 'name': "normal_params"},
                {'params': custom_params, 'lr_mult': 1, 'decay_mult': 0, 'name': "custom_params"},
                ]

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)


    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        self.mask = data['mask'].to(self.device) if 'mask' in data else None

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):

        self.lq = data['lq'].to(self.device)
        # self.lq = self.lq + add_poisson_gaussian_noise(self.lq, poisson_scale=500.0, gaussian_std=0.01)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            raise NotImplementedError('condtions\' generation needs gt inforamtion!')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        with autocast(enabled=self.use_amp):
            if current_iter > self.opt['train']['scheduler']['periods'][0]:
                self.mask = None
            coarse, coarse_up, preds = self.net_g(self.lq, mask=self.mask)

            l_total = 0
            loss_dict = OrderedDict()

            l_kl = get_kl_loss(self.net_g)
            loss_dict['l_kl'] = l_kl
            l_total += 0.01 * l_kl / self.opt['datasets']['train']['mini_batch_sizes'][0]

            alpha = 0.0238
            cal_z_gt = lambda x, y, alpha: (-x + (x**2 + 4 * alpha * y)**0.5) / (2 * alpha)
            coarse_gt = downsample_fft(cal_z_gt(self.lq, self.gt, alpha), sigma=1)

            l_coarse = self.cri_pix(coarse, coarse_gt)
            l_total += l_coarse
            loss_dict['l_pix'] = l_coarse / self.opt['train']['pixel_opt'].get('loss_weight', 1)

            if current_iter % 100 == 0:
                visual_train = tensor2img([preds[0]], rgb2bgr=True)
                visual_train_gt = tensor2img([self.gt[0]], rgb2bgr=True)
                imwrite(visual_train,  osp.join(self.opt['path']['visualization'], f'train.png'))
                imwrite(visual_train_gt,  osp.join(self.opt['path']['visualization'], f'train_gt.png'))

        self.amp_scaler.scale(l_total).backward()
        self.amp_scaler.unscale_(self.optimizer_g)
        # l_pix.backward()

        if self.opt['train']['max_grad_norm']:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt['train']['max_grad_norm'])
        else:
            total_grad_norm = get_grad_norm_(self.net_g.parameters())
        # self.optimizer_g.step()
        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        return total_grad_norm

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        fea_map = self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        self.coarse = self.coarse[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        return fea_map

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            # if self.deterministic == False:
            set_prediction_type(self.net_g, deterministic=True)
            with torch.no_grad():
                pred = self.net_g_ema(img)
            # if self.deterministic == False:
            set_prediction_type(self.net_g, deterministic=False)
            if isinstance(pred, list):
                coarse, pred = pred[-2], pred[-1]
            self.output = pred
            self.coarse = coarse
        else:
            self.net_g.eval()
            # if self.deterministic == False:
            set_prediction_type(self.net_g, deterministic=True)
            with torch.no_grad():
                pred = self.net_g(img)
            # if self.deterministic == False:
            set_prediction_type(self.net_g, deterministic=False)
            if isinstance(pred, list) or isinstance(pred, tuple):
                coarse, pred = pred[-2], pred[-1]
                self.output = pred
                self.coarse = coarse
            else:
                self.output = pred
            self.net_g.train()
            return pred[1:-1]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt["val"]["metrics"].keys()
            }
        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = OrderedDict()
            visuals['lq'] = self.lq.detach().cpu()
            visuals['result'] = self.output.detach().cpu()
            coarse = self.coarse.detach().cpu()
            coarse = (coarse - coarse.min()) / (coarse.max() - coarse.min())
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            coarse_img = tensor2img([coarse], rgb2bgr=rgb2bgr)
            if hasattr(self, 'gt'):
                visuals['gt'] = self.gt.detach().cpu()
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img and self.opt['condition']['type']:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_gt.png')
                    save_coarse_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_c.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt.png')
                    save_coarse_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_c.png')
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
                imwrite(coarse_img, save_coarse_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result']*255, visuals['gt']*255, **opt_)

            cnt += 1

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return self.metric_results['psnr']

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network(
                [self.net_g, self.net_g_ema],
                "net_g",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)


@torch.no_grad()
def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor([0.0])
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad, norm_type) for p in parameters]),
        norm_type
    )
    return total_norm