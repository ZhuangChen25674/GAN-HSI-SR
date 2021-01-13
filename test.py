from data import LoadData
from G import *
import torch
from matplotlib import pyplot as plt
import numpy as np

import torch.nn.functional as F

a = torch.tensor([3.0,4.0])

print(torch.pow((a*a).sum(),0.5))
