from pathlib import Path
import random
from PIL import Image
import numpy as np

random.seed(0)
path = Path('/home/hefeng/data1/HSI-SR/DataSet/CAVE')
all = []

for p in path.iterdir():
    a='/home/hefeng/data1/HSI-SR/DataSet/CAVE/watercolors_ms/watercolors_ms/watercolors_ms_25.png'
    img = Image.open(a)
    data = np.array(img)
    # print(len(data.shape))
    print(data[:,:,0]-data[:,:,1])
    break
    # for p1 in p.iterdir():
    #     a = [i for i in p1.iterdir() if i.match('*.png')]

    #     break
    # break

# img = Image.open(all[1])

# data = np.array(img)

# print(data)