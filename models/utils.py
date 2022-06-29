import torch


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
