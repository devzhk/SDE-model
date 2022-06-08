**Generate trajectory of the probability flow**
```bash
python3 sample_image.py --config configs/cifar/ddpm.yaml --num_samples 10000 --batchsize 100 --t_size 65
```

`generate_data.py` generates training set for operator

`train_operator.py` trains the operator mapping from latent code to image

