import os
import math
import torch


def eval(model, dataloader, criterion,
         device, config, epoch=-1):
    t_dim = config['data']['t_dim']
    t_step = config['data']['t_step']
    num_t = math.ceil(t_dim / t_step)
    logname = config['log']['logname']
    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    t0, t1 = 1., config['data']['epsilon']
    timesteps = torch.linspace(t0, t1, num_t, device=device)
    model.eval()

    with torch.no_grad():
        test_err = 0
        for states in dataloader:
            states = states.to(device)
            in_state = states[:, :, 0]

            pred = model(in_state, timesteps)
            loss = criterion(pred, states)

            test_err += loss
    test_err /= len(dataloader)
    if device == 0:
        print(f'MSE of the whole trajectory: {test_err}')
        save_image(pred[:, :, -1] * 0.5 + 0.5,
                   f'{save_img_dir}/pred_test_{epoch}.png')
        save_image(states[:, :, -1] * 0.5 + 0.5,
                   f'{save_img_dir}/truth_test_{epoch}.png')
    return test_err
