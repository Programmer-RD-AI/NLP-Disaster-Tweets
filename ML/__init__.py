import math
import os
import random
import threading
from typing import *
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchinfo
import torchvision
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import *
from tqdm import tqdm
from wandb import *
from torch.nn import *
from torchvision.models import *
import torchtext
from torchtext.transforms import *
from torchtext.models import *
from sklearn.metrics import *
from torch.hub import *
import torchtext.functional as F
import warnings
import torch.multiprocessing

print(torch.__version__, torchvision.__version__, torchtext.__version__)
torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["WANDB_SILENT"] = "true"
PROJECT_NAME = "NLP-Disaster Tweets"
device = torch.device("cuda")
BATCH_SIZE = 16
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

from ML.dataset import *
from ML.helper_functions import *
from ML.modelling import *
