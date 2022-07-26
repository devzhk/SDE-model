import torch
import re


def interpolate_model(model_ema, model, beta=0.999):
    for p_ema, p in zip(model_ema.parameters(), model.parameters()):
        p_ema.copy_(p.lerp(p_ema, beta))
    for b_ema, b in zip(model_ema.buffers(), model.buffers()):
        b_ema.copy_(b)


def save_ckpt(path,
              model,
              model_ema,
              optim=None,
              args=None):
    '''
    saves checkpoint and configurations to dir/name
    :param args: dict of configuration
    :param g_ema: moving average

    :param optim:
    '''
    ckpt_path = path
    if args.distributed:
        model_ckpt = model.module
    else:
        model_ckpt = model
    state_dict = {
        'model': model_ckpt.state_dict(),
        'ema': model_ema.state_dict(),
        'args': args
    }

    if optim is not None:
        state_dict['optim'] = optim.state_dict()

    torch.save(state_dict, ckpt_path)
    print(f'checkpoint saved at {ckpt_path}')


def load_ddpm_ckpt(model, state_dict, prefix=''):
    dirct_copy_list = ['GroupNorm_', 'Dense_', 'NIN_']
    with torch.no_grad():
        for name, param in model.named_parameters():
            if re.match(r'\w*all_modules\.[0-1]\.[a-z]+', name):
                param.copy_(state_dict[prefix + name])
            elif 'all_modules.2.proj' in name:
                key = prefix + name.replace('.proj', '')
                param.copy_(state_dict[key])
            elif 'all_modules.2.Dense_0' in name:
                continue
            elif any(True for module in dirct_copy_list if module in name):
                key = prefix + name
                param.copy_(state_dict[key])
            elif 'Conv_' in name:
                key = prefix + name
                param.copy_(state_dict[key].reshape(param.shape))
            else:
                key = prefix + name
                # print(state_dict[key].shape)            
                param.copy_(state_dict[key].reshape(param.shape))
