**Generate trajectory of the probability flow**
```bash
python3 sample_image.py --config configs/cifar/ddpm.yaml --num_samples 10000 --batchsize 100 --t_size 65
```


## Dataset setup
First, download the latest gdrive release from github
```bash
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
```
Unzip the archive.
```bash
tar -xvf gdrive_2.1.1_linux_386.tar.gz
```
Perform authentication
```bash
./gdrive about
```
Run python script
```bash
python3 download_data.py
```

## Train

```bash
python3 train_image.py
```