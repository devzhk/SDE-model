import argparse
import os
import psutil
import random
import time
import numpy as np
import yaml
from tqdm import tqdm
import h5py

import torch
from torch.optim import Adam
import torchvision.utils as tvu


from torchdiffeq import odeint_adjoint

from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
from score_sde import sde_lib

from utils.helper import Logger, dict2namespace


def _extract_into_tensor(arr_or_func, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array or a func.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if callable(arr_or_func):
        res = arr_or_func(timesteps).float()
    else:
        res = arr_or_func.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']


class VPODE(torch.nn.Module):
    def __init__(self, model, score_type='guided_diffusion', beta_min=0.1, beta_max=20, N=1000,
                 img_shape=(3, 256, 256), model_kwargs=None):
        """Construct a Variance Preserving SDE.
        Args:
          model: diffusion model
          score_type: [guided_diffusion, score_sde, ddpm]
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        super().__init__()
        self.model = model
        self.score_type = score_type
        self.model_kwargs = model_kwargs
        self.img_shape = img_shape

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t)
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1. / torch.sqrt(1. - self.alphas_cumprod_cont(t))

    def _scale_timesteps(self, t):
        assert torch.all(t <= 1) and torch.all(t >= 0), f't has to be in [0, 1], but get {t} with shape {t.shape}'
        return (t.float() * self.N).long()

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def ode_fn(self, t, x):
        """Create the drift and diffusion functions for the reverse SDE"""
        drift, diffusion = self.vpsde_fn(t, x)

        assert x.ndim == 2 and np.prod(self.img_shape) == x.shape[1], x.shape
        x_img = x.view(-1, *self.img_shape)

        if self.score_type == 'score_sde':
            # model output is epsilon
            sde = sde_lib.VPSDE(beta_min=self.beta_0, beta_max=self.beta_1, N=self.N)
            score_fn = mutils.get_score_fn(sde, self.model, train=False, continuous=True)
            score = score_fn(x_img, t)
            assert x_img.shape == score.shape, f'{x_img.shape}, {score.shape}'
            score = score.view(x.shape[0], -1)

        else:
            raise NotImplementedError(f'Unknown score type in RevVPSDE: {self.score_type}!')

        ode_coef = drift - 0.5 * diffusion[:, None] ** 2 * score
        return ode_coef

    def forward(self, t, states):
        x = states[0]

        t = t.expand(x.shape[0])  # (batch_size, )
        dx_dt = self.ode_fn(t, x)
        assert dx_dt.shape == x.shape

        return dx_dt,


class OdeDiffusion(torch.nn.Module):
    def __init__(self, args, config, device=None, data_parallel=True):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # load model
        if config.data.dataset == 'CIFAR10':
            img_shape = (3, 32, 32)
            model_dir = 'ckpts'
            print(f'model_config: {config}')
            model = mutils.create_model(config)

            optimizer = Adam(model.parameters())
            ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
            state = dict(step=0, optimizer=optimizer, model=model, ema=ema)
            restore_checkpoint(f'{model_dir}/checkpoint_8.pth', state, device)
            ema.copy_to(model.parameters())

        else:
            raise NotImplementedError(f'Unknown dataset {config.data.dataset}!')

        model.eval().to(self.device)

        # data parallel
        ngpus = torch.cuda.device_count()
        if ngpus > 1 and data_parallel:
            model = torch.nn.DataParallel(model).eval()

        self.model = model
        self.vpode = VPODE(model=model, score_type=args.score_type, img_shape=img_shape,
                           model_kwargs=None).to(self.device)
        self.betas = self.vpode.discrete_betas.float().to(self.device)

        self.atol, self.rtol = 1e-3, 1e-3
        self.method = 'euler'

        self.img_shape = img_shape

        print(f'method: {self.method}, atol: {self.atol}, rtol: {self.rtol}, step_size: {self.args.step_size}')

    def uncond_sample(self, batch_size, bs_id=-1, tag=None):
        shape = (batch_size, *self.img_shape)

        start_time = time.time()
        print("Start sampling")
        if tag is None:
            tag = 'rnd' + str(random.randint(0, 10000))
        out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

        if -1 < bs_id < 2:
            os.makedirs(out_dir, exist_ok=True)

        e = torch.FloatTensor(*shape).normal_(0, 1).to(self.device)
        x = e

        if -1 < bs_id < 2:
            tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init.png'), nrow=int(np.sqrt(batch_size)))

        epsilon = 1e-5
        t0, t1 = 1., epsilon
        t_size = self.args.t_size
        ts = torch.linspace(t0, t1, t_size).to(self.device)

        x_ = x.view(batch_size, -1)  # (batch_size, state_size)
        states = (x_, )

        # ODE solver
        odeint = odeint_adjoint
        state_t = odeint(
            self.vpode,
            states,
            ts,
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
            options=None if self.method != 'euler' else dict(step_size=self.args.step_size)  # only used for fixed-point method
        )  # 'euler', 'dopri5'

        # x0_ = state_t[0][-1]
        # x0 = x0_.view(x.shape)  # (batch_size, c, h, w)

        state_t = [x_i.view(x.shape).detach().cpu() for x_i in state_t[0]]
        assert len(state_t) == t_size, len(state_t)

        if -1 < bs_id < 2:
            x0 = state_t[-1][:100]
            tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples0.png'), nrow=int(np.sqrt(100)))
            xmid = state_t[t_size//2][:100]
            tvu.save_image((xmid + 1) * 0.5, os.path.join(out_dir, f'samples{t_size//2}.png'), nrow=int(np.sqrt(100)))
            xinit = state_t[0][:100]
            tvu.save_image((xinit + 1) * 0.5, os.path.join(out_dir, f'init.png'), nrow=int(np.sqrt(100)))

        minutes, seconds = divmod(time.time() - start_time, 60)
        print("Sampling time: {:0>2}:{:05.2f}".format(int(minutes), seconds))
        state_t = torch.stack(state_t, dim=1)
        return state_t


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, default='configs/cifar/ddpm.yaml', help='Path to the config file')
    parser.add_argument('--exp', type=str, default='exp/cifar_sample', help='Path for saving running related data.')
    parser.add_argument('--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--score_type', type=str, default='score_sde', help='[guided_diffusion, score_sde]')
    parser.add_argument('--step_size', type=float, default=1e-3, help='step size for ODE Euler method')
    parser.add_argument('--seed', type=int, default=12301, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--t_size', type=int, default=33, help='number of states in ode to save')
    parser.add_argument('--gpu_ids', type=str, default='0')
    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    print(config)
    new_config = dict2namespace(config)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    log_dir = os.path.join(args.image_folder, 'seed' + str(args.seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    data_save_path = f'{args.log_dir}/ode_data_sd{args.seed}.h5'
    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    ngpus = torch.cuda.device_count()
    print(f'ngpus: {ngpus}')
    num_samples = args.num_samples
    batch_size = args.batch_size * ngpus

    n_batches = num_samples // batch_size
    if n_batches * batch_size < num_samples:
        n_batches += 1

    print(f'n_batches: {n_batches} with batch size: {batch_size}')

    ode_diffusion = OdeDiffusion(args, new_config, device=new_config.device, data_parallel=True)

    with h5py.File(data_save_path, 'a') as f:
        dset = f.create_dataset(f'data_t{args.t_size}',
                                (batch_size, args.t_size, 3, 32, 32),
                                maxshape=(None, args.t_size, 3, 32, 32))
        for i in tqdm(range(n_batches)):
            images_batch = ode_diffusion.uncond_sample(batch_size, bs_id=i)
            if i == 0:
                dset[:] = images_batch.cpu().numpy()
            else:
                dset.resize(size=(i + 1) * batch_size, axis=0)
                dset[-batch_size:] = images_batch.cpu().numpy()
            ram_usage = psutil.Process().memory_info().rss / (1024 * 1024)
            print(f'Memory usage: {ram_usage} MB.')

    logger.close()