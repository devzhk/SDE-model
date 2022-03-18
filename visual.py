import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import myODE


dataset = myODE('data/ode_data_sd1.pt')

ys = dataset.images.numpy().reshape(1000, -1).T
xs = list(range(1000))
pos = [10, 30, 40, 50, 65, 100]
num_lines = 6


for i in range(num_lines):
    line, = plt.plot(xs, ys[pos[i]])
plt.legend()
plt.ylabel('Pixel value')
plt.xlabel('Time step')
plt.savefig('figs/time.png')