import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import sys
import torch 
import src.settings.config as cfg
import src.data.Element
import src.network.Network as net
import src.network.training as train
from src.data.Element import VoltageDataset
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from src.data.dataset.dataset import generate_dataset
from logs.log import Log
from torch.utils.data import Subset, ConcatDataset
import preprocess as pre
# ---------------------------------------README----------------------------------------------- 
# The Dataset from the in situ can be found in the src/data/sensor_data/measurement/raw folder'
# The size of the dataset means it needs to be windowed (split) into accurate training data. 
# This can be done via dataset.py file (see documentation there)
# The model is a CNN, which is trained on the dataset. The model can be found in the src/network folder.

#generate_dataset(cfg.manual_paths, cfg.target_path, indices=cfg.indices, stride= 100000, kernel_size=  100000)
#raise RuntimeError("Dataset generated, please remove this line to continue")
# paths = open("src/data/dataset/list_path.txt", "r").read().split('\n')[:-1]
# set = VoltageDataset(paths, "src/data/dataset/labels/labels.csv")
logger = Log("src/logs/log_general.txt", "src/logs/log_detailed.txt")
#set.info(log = logger)
model = net.WindowedCNN(100000, 1)
model.info()

#create training data
    

fig, axs = plt.subplots(2, 1, figsize=(8, 6))



train_dataset, val_dataset = pre.get_dataset("src/data/dataset/list_path.txt", 10, label_path="src/data/dataset/labels/labels.csv",remove_duplicates=True, info=False)
test_dataset, test_val = pre.get_dataset("src/data/dataset/secondaries.txt", 0,label_path="src/data/dataset/labels/labels_secondary.csv", remove_duplicates=False, info=False)
dataloader_train = data.DataLoader(train_dataset, batch_size=cfg.BATCH, shuffle=True)
dataloader_test = data.DataLoader(val_dataset, batch_size=cfg.BATCH, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001) #0001 0.0000001
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_function = torch.nn.MSELoss()
epochs = 500
model.to(cfg.DEVICE)
for y in range(0, 100):
    train.train(model, dataset_train=dataloader_train, dataset_validate=dataloader_test, loss_function=loss_function, optimizer=optimizer, step=[], losses=[], axs=axs, epochs=100, visualization=True, plot_data=None, secondary_info=False)
    #dataloader_train = data.DataLoader(test_dataset, batch_size=cfg.BATCH, shuffle=True)
    #train.validate(dataset=dataloader_train, model=model, accuracy=[], loss_function=loss_function, primary_info=True, secondary_info=True)
plt.show()

